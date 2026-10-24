"""Memory-management policies compared in the study (all act on the same `Decision`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .catalog import Catalog
from .config import ExperimentConfig
from .domain import CUE_TO_RECORD_KIND, Action, Cue
from .engine import Decision
from .metrics import score_state
from .operations import Overflow


class Policy(Protocol):
    name: str
    overflow: Overflow

    def reset(self) -> None: ...

    def act(self, d: Decision) -> Action: ...


@dataclass
class StoreAllPolicy:
    """Appends every claim; a FIFO window evicts the oldest records when the budget is full."""

    name: str = "store-all"
    overflow: Overflow = "evict_oldest"

    def reset(self) -> None:
        return None

    def act(self, d: Decision) -> Action:
        return Action.ADD


@dataclass
class RuleThresholds:
    state: float = 0.5
    change: float = 0.5  # change / temporary / plan
    retract: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {"state": self.state, "change": self.change, "retract": self.retract}


@dataclass
class RulePolicy:
    """Hand-written, confidence-gated rules. Thresholds are tuned on validation episodes."""

    catalog: Catalog
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    name: str = "rule-based"
    overflow: Overflow = "reject"

    def reset(self) -> None:
        return None

    def act(self, d: Decision) -> Action:
        ex = d.extraction
        t = self.thresholds
        if ex.cue in (Cue.HYPOTHETICAL, Cue.THIRD_PARTY):
            return Action.IGNORE
        if ex.cue is Cue.RETRACT:
            ok = ex.confidence >= t.retract and d.mask[Action.DELETE]
            return Action.DELETE if ok else Action.IGNORE
        gate = t.state if ex.cue is Cue.STATE else t.change
        if ex.confidence < gate:
            return Action.IGNORE
        if self.catalog.is_multi(ex.attribute):
            return Action.ADD if d.mask[Action.ADD] and self._new_value(d) else Action.IGNORE
        if d.mask[Action.UPDATE]:
            return Action.UPDATE
        if d.mask[Action.ADD] and self._new_value(d):
            return Action.ADD
        return Action.IGNORE

    def _new_value(self, d: Decision) -> bool:
        kind = CUE_TO_RECORD_KIND[d.extraction.cue]
        value = self.catalog.canonical_value(d.extraction.attribute, d.extraction.value or "")
        return not any(
            r.attribute == d.extraction.attribute and r.kind is kind and r.value == value and r.live(d.day)
            for r in d.state.records
        )


@dataclass
class OraclePolicy:
    """Myopic oracle: picks the valid action with the best immediate reward using ground truth.

    It is a reference *ceiling for one-step decisions*, not an achievable system, because it
    reads the hidden ground-truth snapshot.
    """

    catalog: Catalog
    cfg: ExperimentConfig
    snapshots: dict[int, object] = field(default_factory=dict)
    name: str = "oracle"
    overflow: Overflow = "reject"

    def reset(self) -> None:
        return None

    def bind(self, episode) -> None:
        self.snapshots = {t.index: t.snapshot for t in episode.turns}

    def act(self, d: Decision) -> Action:
        snap = self.snapshots[d.turn]
        best, best_v = Action.IGNORE, -1e9
        for a in (Action.IGNORE, Action.UPDATE, Action.ADD, Action.DELETE):
            plan = d.plans[a]
            if not plan.ok:
                continue
            removed = {r.id for r in plan.delete} | {r.id for r in plan.evict}
            sup = {r.id for r in plan.supersede}
            recs = [r for r in d.state.records if r.id not in removed and r.id not in sup]
            recs += plan.supersede + ([plan.insert] if plan.insert else [])
            sc = score_state(recs, snap, d.day, self.catalog, self.cfg.reward)
            v = (
                sc.value(self.cfg.reward, self.catalog.max_importance)
                - self.cfg.reward.token_cost * (plan.tokens_after - d.state.tokens) / self.cfg.memory.budget_tokens
            )
            if v > best_v + 1e-9:
                best, best_v = a, v
        return best


class RandomMaskedPolicy:
    """Uniform over valid actions (sanity-check floor)."""

    name = "random"
    overflow: Overflow = "reject"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        return None

    def act(self, d: Decision) -> Action:
        valid = np.flatnonzero(d.mask)
        return Action(int(self.rng.choice(valid)))
