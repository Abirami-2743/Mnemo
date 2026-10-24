"""Stepwise episode engine shared by the Gym environment, the evaluator and baselines.

The engine walks through a benchmark episode. Turns without an extracted claim need no
decision and are skipped (time still passes and records still expire). For every
claim it exposes a `Decision` (state, extraction, validity mask, observation) and
accepts an `Action`; it applies it through `mnemo.operations` and scores the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .benchmark.generator import Episode, Truth
from .catalog import Catalog
from .config import ExperimentConfig
from .domain import Action, Extraction
from .features import build_observation
from .metrics import StateMetrics, harmful_removals, score_state, to_metrics
from .operations import (
    ActionPlan,
    Outcome,
    Overflow,
    TurnState,
    apply_action,
    begin_turn,
    plan_all,
    valid_mask,
)
from .qa import run_probes
from .store.base import MemoryStore
from .store.memory import InMemoryStore

USER_ID = "episode"


@dataclass(slots=True)
class Decision:
    turn: int
    day: int
    text: str
    truth: Truth
    state: TurnState
    extraction: Extraction
    plans: dict[Action, ActionPlan]
    mask: np.ndarray
    obs: np.ndarray


@dataclass(slots=True)
class StepResult:
    reward: float
    reward_parts: dict[str, float]
    outcome: Outcome
    done: bool
    harmful_removals: int


@dataclass(slots=True)
class CheckpointScore:
    turn: int
    accuracy: float
    n_probes: int
    by_kind: dict[str, tuple[float, int]]


@dataclass(slots=True)
class EpisodeTrace:
    """Everything recorded while an episode ran."""

    per_turn: list[StateMetrics] = field(default_factory=list)
    checkpoints: list[CheckpointScore] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    n_rejected: int = 0
    n_decisions: int = 0
    harmful: int = 0
    total_reward: float = 0.0
    final_records: list = field(default_factory=list)


class EpisodeEngine:
    def __init__(
        self,
        catalog: Catalog,
        cfg: ExperimentConfig,
        *,
        budget: int | None = None,
        overflow: Overflow = "reject",
        reward_cfg=None,
    ) -> None:
        self.catalog = catalog
        self.cfg = cfg
        self.budget = budget or cfg.memory.budget_tokens
        self.overflow: Overflow = overflow
        self.reward_cfg = reward_cfg or cfg.reward
        self.store: MemoryStore = InMemoryStore()
        self.episode: Episode | None = None
        self.extractions: list[Extraction | None] = []
        self.trace = EpisodeTrace()
        self.decision: Decision | None = None
        self._cursor = 0
        self._pending_qa = 0.0
        self._ckpt_by_turn: dict[int, int] = {}

    # ------------------------------------------------------------------ control
    def start(self, episode: Episode, extractions: list[Extraction | None]) -> Decision | None:
        self.store.reset_user(USER_ID)
        self.episode = episode
        self.extractions = extractions
        self.trace = EpisodeTrace()
        self._ckpt_by_turn = {c.turn_index: i for i, c in enumerate(episode.checkpoints)}
        self._cursor = 0
        self._pending_qa = 0.0
        self.decision = None
        self._advance()
        return self.decision

    @property
    def done(self) -> bool:
        return self.decision is None

    def _norm(self) -> float:
        return self.catalog.max_importance

    def _record_turn(self, turn: int, records) -> None:
        assert self.episode is not None
        t = self.episode.turns[turn]
        sc = score_state(records, t.snapshot, t.day, self.catalog, self.reward_cfg)
        self.trace.per_turn.append(to_metrics(sc))
        self.trace.final_records = records
        idx = self._ckpt_by_turn.get(turn)
        if idx is not None:
            ck = self.episode.checkpoints[idx]
            results = run_probes(records, ck.probes, ck.day)
            acc = float(np.mean([s for _, _, s in results]))
            by_kind: dict[str, list[float]] = {}
            for probe, _, s in results:
                by_kind.setdefault(probe.kind, []).append(s)
            self.trace.checkpoints.append(
                CheckpointScore(turn, acc, len(results), {k: (float(np.mean(v)), len(v)) for k, v in by_kind.items()})
            )
            self._pending_qa += self.reward_cfg.qa_weight * acc

    def _advance(self) -> None:
        """Move to the next turn that carries a claim, skipping (and scoring) idle turns."""
        assert self.episode is not None
        ep = self.episode
        while self._cursor < ep.horizon:
            turn = ep.turns[self._cursor]
            state = begin_turn(self.store, USER_ID, turn.day)
            ex = self.extractions[self._cursor]
            if ex is None:
                self._record_turn(self._cursor, state.records)
                self._cursor += 1
                continue
            plans = plan_all(
                state,
                ex,
                turn=turn.index,
                budget=self.budget,
                catalog=self.catalog,
                mem_cfg=self.cfg.memory,
                overflow=self.overflow,
            )
            mask = valid_mask(plans)
            obs = build_observation(
                self.store,
                state,
                ex,
                plans,
                mask,
                budget=self.budget,
                catalog=self.catalog,
                mem_cfg=self.cfg.memory,
            )
            self.decision = Decision(turn.index, turn.day, turn.text, turn.truth, state, ex, plans, mask, obs)
            return
        self.decision = None

    # --------------------------------------------------------------------- step
    def step(self, action: Action) -> StepResult:
        dec = self.decision
        assert dec is not None and self.episode is not None, "episode is finished"
        turn = self.episode.turns[dec.turn]
        rc = self.reward_cfg
        norm = self._norm()

        before = score_state(dec.state.records, turn.snapshot, dec.day, self.catalog, rc)
        outcome = apply_action(
            self.store,
            dec.state,
            dec.plans,
            action,
            dec.extraction,
            turn=dec.turn,
            catalog=self.catalog,
        )
        after = score_state(outcome.records_after, turn.snapshot, dec.day, self.catalog, rc)
        r_state = after.value(rc, norm) - before.value(rc, norm)
        r_tokens = -rc.token_cost * (outcome.tokens_after - outcome.tokens_before) / self.budget
        harm = harmful_removals(outcome.deleted + outcome.evicted, turn.snapshot)

        self.trace.actions.append(outcome.effective)
        self.trace.n_decisions += 1
        self.trace.n_rejected += int(outcome.rejected)
        self.trace.harmful += harm
        self._cursor = dec.turn + 1
        self._record_turn(dec.turn, outcome.records_after)
        self._advance()

        r_qa = self._pending_qa
        self._pending_qa = 0.0
        reward = r_state + r_tokens + r_qa
        self.trace.total_reward += reward
        return StepResult(
            reward=float(reward),
            reward_parts={"state": r_state, "tokens": r_tokens, "qa": r_qa},
            outcome=outcome,
            done=self.done,
            harmful_removals=harm,
        )
