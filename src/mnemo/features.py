"""Observation features: what the policy sees for one extracted claim."""

from __future__ import annotations

import numpy as np

from .catalog import Catalog
from .config import MemoryConfig
from .domain import CUE_TO_RECORD_KIND, CUES, N_ACTIONS, Action, Extraction
from .operations import ActionPlan, TurnState
from .store.base import MemoryStore

FEATURE_NAMES: tuple[str, ...] = (
    *(f"cue_{c.value}" for c in CUES),
    "confidence",
    "importance",
    "is_multi",
    "target_same_kind",
    "target_same_value",
    "other_kind_same_attr",
    "same_value_other_kind",
    "n_live_attr",
    "history_exists",
    "equals_history",
    "duration",
    "starts_in",
    "budget_fill",
    "add_cost",
    "headroom_after_add",
    "records_norm",
    "prior_seen",
    "prior_applied_frac",
    "target_age",
    "target_confidence",
    *(f"valid_{a.name.lower()}" for a in Action),
)
OBS_DIM = len(FEATURE_NAMES)
# Feature groups used by ablations.
GROUPS: dict[str, tuple[str, ...]] = {
    "confidence": ("confidence", "target_confidence"),
    "support": ("prior_seen", "prior_applied_frac"),
    "budget": ("budget_fill", "add_cost", "headroom_after_add"),
}


def feature_mask(ablate: tuple[str, ...] = ()) -> np.ndarray:
    """1.0 for kept features, 0.0 for ablated ones."""
    drop = {n for g in ablate for n in GROUPS[g]}
    return np.array([0.0 if n in drop else 1.0 for n in FEATURE_NAMES], dtype=np.float32)


def build_observation(
    store: MemoryStore,
    state: TurnState,
    extraction: Extraction | None,
    plans: dict[Action, ActionPlan],
    mask: np.ndarray,
    *,
    budget: int,
    catalog: Catalog,
    mem_cfg: MemoryConfig,
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    day = state.day
    used = state.tokens
    obs[idx["budget_fill"]] = min(used / budget, 2.0)
    obs[idx["records_norm"]] = min(len(state.records) / mem_cfg.max_records_norm, 2.0)
    for a in Action:
        obs[idx[f"valid_{a.name.lower()}"]] = float(mask[a])
    if extraction is None:
        return obs

    ex = extraction
    obs[idx[f"cue_{ex.cue.value}"]] = 1.0
    obs[idx["confidence"]] = ex.confidence
    obs[idx["importance"]] = catalog.importance(ex.attribute) / catalog.max_importance
    obs[idx["is_multi"]] = float(catalog.is_multi(ex.attribute))
    obs[idx["duration"]] = min((ex.duration_days or 0) / 30.0, 1.0)
    obs[idx["starts_in"]] = min((ex.starts_in_days or 0) / 30.0, 1.0)

    kind = CUE_TO_RECORD_KIND[ex.cue]
    value = catalog.canonical_value(ex.attribute, ex.value) if ex.value is not None else None
    attr_live = [r for r in state.records if r.attribute == ex.attribute and r.live(day)]
    same_kind = [r for r in attr_live if r.kind is kind]
    obs[idx["n_live_attr"]] = min(len(attr_live) / 3.0, 1.0)
    obs[idx["target_same_kind"]] = float(bool(same_kind))
    obs[idx["target_same_value"]] = float(any(r.value == value for r in same_kind))
    obs[idx["other_kind_same_attr"]] = float(any(r.kind is not kind for r in attr_live))
    obs[idx["same_value_other_kind"]] = float(any(r.kind is not kind and r.value == value for r in attr_live))
    history = [r for r in state.records if r.attribute == ex.attribute and not r.live(day)]
    obs[idx["history_exists"]] = float(bool(history))
    obs[idx["equals_history"]] = float(any(r.value == value for r in history))
    if same_kind:
        newest = max(same_kind, key=lambda r: r.seq)
        obs[idx["target_age"]] = min((day - newest.updated_day) / 60.0, 1.0)
        obs[idx["target_confidence"]] = newest.confidence

    add_plan = plans[Action.ADD]
    cost = add_plan.insert.token_cost if add_plan.insert is not None else 0
    obs[idx["add_cost"]] = min(cost / budget, 1.0)
    obs[idx["headroom_after_add"]] = float(np.clip((budget - used - cost) / budget, -1.0, 1.0))
    if value is not None:
        seen, applied = store.claim_stats(state.user_id, ex.attribute, value)
        obs[idx["prior_seen"]] = min(seen / 5.0, 1.0)
        obs[idx["prior_applied_frac"]] = applied / seen if seen else 0.0
    return obs


assert N_ACTIONS == 4
