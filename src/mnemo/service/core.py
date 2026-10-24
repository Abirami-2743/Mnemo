"""Session orchestration: one user conversation processed by several memory managers in lockstep."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..agents.explainer import stream_explanation
from ..agents.extractor import HeuristicExtractor, LLMExtractor
from ..agents.llm import GroqClient
from ..agents.qa import QAAgent
from ..benchmark import BenchmarkGenerator, Episode
from ..catalog import Catalog, load_catalog
from ..config import ExperimentConfig, Settings, load_experiment_config
from ..domain import Action, Extraction
from ..features import build_observation
from ..metrics import score_state, to_metrics
from ..operations import apply_action, begin_turn, plan_all, valid_mask
from ..policies import Policy, RulePolicy, RuleThresholds, StoreAllPolicy
from ..qa import run_probes
from ..store import MemoryStore, create_store

log = logging.getLogger(__name__)
SYSTEMS_ORDER = ("store-all", "rule-based", "mnemo-ppo")


class SessionNotFound(KeyError):
    pass


class SessionLimit(RuntimeError):
    pass


@dataclass
class Session:
    id: str
    episode: Episode | None
    day: int = 0
    turn: int = 0
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def user_id(self, system: str) -> str:
        return f"{self.id}:{system}"


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class MemoryService:
    def __init__(self, settings: Settings, store: MemoryStore | None = None) -> None:
        self.settings = settings
        self.cfg: ExperimentConfig = load_experiment_config(settings.config_path)
        self.catalog: Catalog = load_catalog(settings.catalog_path)
        self.gen = BenchmarkGenerator(self.catalog, self.cfg.benchmark)
        self.store: MemoryStore = store or create_store(settings)
        self.sessions: dict[str, Session] = {}
        self.client: GroqClient | None = None
        heuristic = HeuristicExtractor(self.catalog)
        if settings.llm_enabled:
            assert settings.groq_api_key is not None
            self.client = GroqClient(
                settings.groq_api_key.get_secret_value(),
                model=settings.groq_model,
                base_url=settings.groq_base_url,
                timeout_s=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
            self.extractor: Any = LLMExtractor(self.client, self.catalog, heuristic)
        else:
            self.extractor = heuristic
        self.qa = QAAgent(self.catalog, self.client)
        self.policies: dict[str, Policy] = self._load_policies()

    # ------------------------------------------------------------------ setup
    def _load_policies(self) -> dict[str, Policy]:
        thresholds = RuleThresholds()
        results = self.settings.results_dir / "results.json"
        if results.exists():
            try:
                info = json.loads(results.read_text())["meta"]["rule_thresholds"]
                thresholds = RuleThresholds(info["state"], info["change"], info["retract"])
            except (KeyError, ValueError):
                log.warning("could not read tuned rule thresholds from %s", results)
        policies: dict[str, Policy] = {
            "store-all": StoreAllPolicy(),
            "rule-based": RulePolicy(self.catalog, thresholds),
        }
        if self.settings.policy_path.exists():
            from ..rl.numpy_policy import NumpyPolicy

            policies["mnemo-ppo"] = NumpyPolicy(self.settings.policy_path)
        elif self.settings.model_path.exists():
            try:
                from ..rl.policy import PPOPolicy

                policies["mnemo-ppo"] = PPOPolicy.load(self.settings.model_path)
            except Exception:
                log.exception("failed to load PPO model; mnemo-ppo disabled")
        else:
            log.warning("no PPO policy found (%s); mnemo-ppo disabled", self.settings.policy_path)
        return policies

    @property
    def systems(self) -> list[str]:
        return [s for s in SYSTEMS_ORDER if s in self.policies]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "store_backend": self.store.backend,
            "store_reachable": self.store.ping(),
            "llm": self.settings.groq_model if self.client else None,
            "extractor": self.extractor.source,
            "systems": self.systems,
            "sessions": len(self.sessions),
        }

    # --------------------------------------------------------------- sessions
    def _gc(self) -> None:
        now = time.time()
        for sid in [s for s, x in self.sessions.items() if now - x.touched > self.settings.session_ttl_s]:
            self.delete_session(sid)

    def create_session(self, scenario_seed: int | None) -> Session:
        self._gc()
        if len(self.sessions) >= self.settings.max_sessions:
            raise SessionLimit("too many active sessions")
        episode = None
        if scenario_seed is not None:
            lo, hi = self.cfg.benchmark.splits["test"]
            episode = self.gen.generate(lo + scenario_seed % (hi - lo))
        sess = Session(id=uuid.uuid4().hex[:12], episode=episode)
        self.sessions[sess.id] = sess
        return sess

    def get(self, sid: str) -> Session:
        try:
            sess = self.sessions[sid]
        except KeyError as exc:
            raise SessionNotFound(sid) from exc
        sess.touched = time.time()
        return sess

    def delete_session(self, sid: str) -> None:
        sess = self.sessions.pop(sid, None)
        if sess:
            for system in self.systems:
                self.store.reset_user(sess.user_id(system))

    def memory_view(self, sess: Session) -> dict[str, Any]:
        out = {}
        for system in self.systems:
            records = self.store.get_records(sess.user_id(system))
            live = [r for r in records if r.live(sess.day) or r.status.value == "historical"]
            out[system] = {
                "records": [r.public() for r in sorted(live, key=lambda r: r.seq)],
                "tokens": sum(r.token_cost for r in live),
            }
        return out

    def scenario_info(self, sess: Session) -> dict[str, Any]:
        ep = sess.episode
        return {
            "session_id": sess.id,
            "scenario": None
            if ep is None
            else {"episode_id": ep.episode_id, "profile": ep.profile, "turns": ep.horizon},
            "budget_tokens": self.cfg.memory.budget_tokens,
            "systems": self.systems,
            "extractor": self.extractor.source,
        }

    # ----------------------------------------------------------------- turns
    async def _extract(self, text: str) -> Extraction | None:
        res = self.extractor.extract(text)
        return await res if asyncio.iscoroutine(res) else res

    async def run_turn(self, sess: Session, text: str | None, advance_days: int) -> AsyncIterator[str]:
        """Process one turn (scripted if `text` is None) and yield SSE frames."""
        async with sess.lock:
            gt_turn = None
            if text is None:
                if sess.episode is None:
                    raise ValueError("this session has no scripted scenario; provide text")
                if sess.turn >= sess.episode.horizon:
                    yield sse("done", {"finished": True})
                    return
                gt_turn = sess.episode.turns[sess.turn]
                text, sess.day = gt_turn.text, gt_turn.day
            else:
                sess.day += advance_days
            budget = self.cfg.memory.budget_tokens
            yield sse("turn", {"turn": sess.turn, "day": sess.day, "text": text})
            extraction = await self._extract(text)
            yield sse("extraction", {"extraction": extraction.model_dump(mode="json") if extraction else None})

            scores: dict[str, Any] = {}
            outcomes = {}
            for system in self.systems:
                policy = self.policies[system]
                uid = sess.user_id(system)
                state = begin_turn(self.store, uid, sess.day)
                plans = plan_all(
                    state,
                    extraction,
                    turn=sess.turn,
                    budget=budget,
                    catalog=self.catalog,
                    mem_cfg=self.cfg.memory,
                    overflow=policy.overflow,
                )
                mask = valid_mask(plans)
                action = Action.IGNORE
                if extraction is not None:
                    obs = build_observation(
                        self.store,
                        state,
                        extraction,
                        plans,
                        mask,
                        budget=budget,
                        catalog=self.catalog,
                        mem_cfg=self.cfg.memory,
                    )
                    from ..benchmark.generator import Truth
                    from ..domain import TurnKind
                    from ..engine import Decision

                    dec = Decision(
                        sess.turn, sess.day, text, Truth(TurnKind.CHITCHAT), state, extraction, plans, mask, obs
                    )
                    action = policy.act(dec)
                outcome = apply_action(
                    self.store,
                    state,
                    plans,
                    action,
                    extraction,
                    turn=sess.turn,
                    catalog=self.catalog,
                    extra_event={"system": system, "text": text},
                )
                outcomes[system] = outcome
                row = {
                    "system": system,
                    "requested": outcome.requested.name,
                    "effective": outcome.effective.name,
                    "reason": outcome.reason,
                    "changes": outcome.changes(),
                    "tokens": outcome.tokens_after,
                    "records": [
                        r.public() for r in outcome.records_after if r.live(sess.day) or r.status.value == "historical"
                    ],
                }
                if gt_turn is not None:
                    m = to_metrics(
                        score_state(outcome.records_after, gt_turn.snapshot, sess.day, self.catalog, self.cfg.reward)
                    )
                    row["metrics"] = m.as_dict()
                    scores[system] = m.as_dict()
                yield sse("action", row)

            # narration of the MNEMO decision (streamed token by token)
            focus = "mnemo-ppo" if "mnemo-ppo" in outcomes else self.systems[-1]
            async for tok in stream_explanation(self.client, extraction, outcomes[focus], focus):
                yield sse("explain_token", {"system": focus, "text": tok})
            yield sse("explain_end", {"system": focus})

            if scores:
                yield sse("scores", {"turn": sess.turn, "scores": scores})
            if gt_turn is not None and sess.episode is not None:
                for ck in sess.episode.checkpoints:
                    if ck.turn_index == sess.turn:
                        for ev in await self._checkpoint_events(sess, ck):
                            yield ev
            sess.turn += 1
            yield sse("done", {"finished": sess.episode is not None and sess.turn >= sess.episode.horizon})

    async def _checkpoint_events(self, sess: Session, ck) -> list[str]:
        events = []
        for system in self.systems:
            records = self.store.get_records(sess.user_id(system))
            results = run_probes(records, ck.probes, ck.day)
            events.append(
                sse(
                    "qa_result",
                    {
                        "system": system,
                        "accuracy": float(np.mean([s for _, _, s in results])),
                        "probes": [
                            {
                                "question": p.question,
                                "kind": p.kind,
                                "expected": sorted(p.expected) if isinstance(p.expected, frozenset) else p.expected,
                                "answer": sorted(a) if isinstance(a, frozenset) else a,
                                "score": s,
                            }
                            for p, a, s in results
                        ],
                    },
                )
            )
        return events

    async def ask(self, sess: Session, question: str) -> dict[str, Any]:
        async with sess.lock:
            out = {}
            for system in self.systems:
                records = self.store.get_records(sess.user_id(system))
                answer, source = await self.qa.answer_free(question, records, sess.day)
                out[system] = {"answer": answer, "source": source}
            return {"question": question, "answers": out}

    async def aclose(self) -> None:
        if self.client:
            await self.client.aclose()
        close = getattr(self.store, "close", None)
        if close:
            close()
