"""Semantics of the four memory actions, the token budget and temporal expiry.

This module is the single definition of what ADD / UPDATE / IGNORE / DELETE *do*.
It is backend-agnostic (works through the `MemoryStore` protocol) and is used by
the RL environment, the baselines, the evaluation harness and the API, so every
system under comparison is bound by exactly the same rules.

Design rules
------------
* ADD always appends; it never checks for duplicates or contradictions. Deciding
  *not* to add a duplicate is the policy's job.
* UPDATE supersedes the live record of the same attribute and kind. A superseded
  permanent record is kept as *history* (`status=historical`); a superseded
  temporary/planned record is simply replaced. Only the `history_depth` most
  recent history records per attribute are retained.
* DELETE removes the live record(s) matching the claim's attribute (and value).
* IGNORE does nothing.
* The budget is hard. History is the first thing evicted to make room. If a
  new record still does not fit, the action is infeasible ("over_budget") under
  the `reject` overflow mode; the `evict_oldest` mode (a FIFO window, used by the
  store-all baseline) evicts the oldest live records instead.
* Temporary and planned records expire automatically at `valid_to`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .catalog import Catalog
from .config import MemoryConfig
from .domain import (
    CUE_TO_RECORD_KIND,
    N_ACTIONS,
    Action,
    Cue,
    Extraction,
    MemoryRecord,
    RecordKind,
    RecordStatus,
)
from .store.base import MemoryStore
from .tokens import record_token_cost

Overflow = Literal["reject", "evict_oldest"]


@dataclass(slots=True)
class TurnState:
    """Memory as the policy sees it at decision time (expired records already removed)."""

    user_id: str
    day: int
    records: list[MemoryRecord]
    expired: list[MemoryRecord]

    @property
    def tokens(self) -> int:
        return sum(r.token_cost for r in self.records)


@dataclass(slots=True)
class ActionPlan:
    """What an action would do, computed without touching the store."""

    action: Action
    ok: bool
    reason: str
    insert: MemoryRecord | None = None
    supersede: list[MemoryRecord] = field(default_factory=list)  # new historical versions
    delete: list[MemoryRecord] = field(default_factory=list)
    evict: list[MemoryRecord] = field(default_factory=list)
    tokens_after: int = 0


@dataclass(slots=True)
class Outcome:
    """Result of applying an action to the store."""

    requested: Action
    effective: Action
    applied: bool
    rejected: bool
    reason: str
    inserted: MemoryRecord | None
    superseded: list[MemoryRecord]
    deleted: list[MemoryRecord]
    evicted: list[MemoryRecord]
    expired: list[MemoryRecord]
    records_after: list[MemoryRecord]
    tokens_before: int
    tokens_after: int

    def changes(self) -> dict[str, list[dict[str, Any]]]:
        def brief(rs: list[MemoryRecord]) -> list[dict[str, Any]]:
            return [{"id": r.id, "attribute": r.attribute, "value": r.value} for r in rs]

        return {
            "inserted": brief([self.inserted] if self.inserted else []),
            "superseded": brief(self.superseded),
            "deleted": brief(self.deleted),
            "evicted": brief(self.evicted),
            "expired": brief(self.expired),
        }


# --------------------------------------------------------------------------- helpers
def begin_turn(store: MemoryStore, user_id: str, day: int) -> TurnState:
    """Load the user's memory and physically remove records that expired by `day`."""
    records = store.get_records(user_id)
    expired = [r for r in records if r.expired(day)]
    if expired:
        store.delete_many(user_id, [r.id for r in expired])
        gone = {r.id for r in expired}
        records = [r for r in records if r.id not in gone]
    return TurnState(user_id=user_id, day=day, records=records, expired=expired)


def build_record(
    extraction: Extraction,
    *,
    user_id: str,
    day: int,
    turn: int,
    mem_cfg: MemoryConfig,
    catalog: Catalog,
) -> MemoryRecord:
    """Record that a claim would create. Its kind is decided only by the extracted cue."""
    assert extraction.value is not None
    value = catalog.canonical_value(extraction.attribute, extraction.value)
    kind = CUE_TO_RECORD_KIND[extraction.cue]
    valid_to: int | None
    if kind is RecordKind.TEMPORARY:
        valid_from = day
        valid_to = day + (extraction.duration_days or mem_cfg.default_temp_ttl)
    elif kind is RecordKind.PLANNED:
        start = extraction.starts_in_days if extraction.starts_in_days is not None else mem_cfg.default_plan_start
        valid_from = day + start
        valid_to = valid_from + (extraction.duration_days or mem_cfg.default_plan_duration)
    else:
        valid_from, valid_to = day, None
    cost = record_token_cost(extraction.attribute, value, kind, RecordStatus.CURRENT, valid_from, valid_to)
    return MemoryRecord(
        id="",
        user_id=user_id,
        seq=0,
        attribute=extraction.attribute,
        value=value,
        kind=kind,
        status=RecordStatus.CURRENT,
        valid_from=valid_from,
        valid_to=valid_to,
        created_day=day,
        updated_day=day,
        source_turn=turn,
        confidence=extraction.confidence,
        token_cost=cost,
    )


def live_targets(records: list[MemoryRecord], attribute: str, kind: RecordKind, day: int) -> list[MemoryRecord]:
    return [r for r in records if r.live(day) and r.attribute == attribute and r.kind is kind]


def _fit(
    survivors: list[MemoryRecord],
    new: MemoryRecord,
    budget: int,
    overflow: Overflow,
) -> tuple[bool, list[MemoryRecord]]:
    """Can `new` fit next to `survivors`? Returns (fits, records evicted to make room)."""
    used = sum(r.token_cost for r in survivors) + new.token_cost
    need = used - budget
    if need <= 0:
        return True, []
    evicted: list[MemoryRecord] = []
    history = sorted((r for r in survivors if r.status is RecordStatus.HISTORICAL), key=lambda r: r.seq)
    for rec in history:
        if need <= 0:
            break
        evicted.append(rec)
        need -= rec.token_cost
    if need > 0 and overflow == "evict_oldest":
        live = sorted((r for r in survivors if r.status is RecordStatus.CURRENT), key=lambda r: r.seq)
        for rec in live:
            if need <= 0:
                break
            evicted.append(rec)
            need -= rec.token_cost
    return (need <= 0), evicted


def _prune_history(
    records: list[MemoryRecord], new_history: list[MemoryRecord], attribute: str, depth: int
) -> list[MemoryRecord]:
    """History records of `attribute` beyond the `depth` most recent ones."""
    old = [r for r in records if r.attribute == attribute and r.status is RecordStatus.HISTORICAL]
    combined = sorted(old + new_history, key=lambda r: (r.valid_to or 0, r.seq))
    if len(combined) <= depth:
        return []
    drop = combined[: len(combined) - depth]
    new_ids = {r.id for r in new_history}
    return [r for r in drop if r.id not in new_ids]


# --------------------------------------------------------------------------- planning
def plan_all(
    state: TurnState,
    extraction: Extraction | None,
    *,
    turn: int,
    budget: int,
    catalog: Catalog,
    mem_cfg: MemoryConfig,
    overflow: Overflow = "reject",
) -> dict[Action, ActionPlan]:
    """Plan every action for the current claim. `plan.ok` is the action-validity mask."""
    used = state.tokens
    plans: dict[Action, ActionPlan] = {Action.IGNORE: ActionPlan(Action.IGNORE, True, "ignored", tokens_after=used)}
    if extraction is None:
        for action in (Action.ADD, Action.UPDATE, Action.DELETE):
            plans[action] = ActionPlan(action, False, "no_extraction", tokens_after=used)
        return plans

    day = state.day
    records = state.records

    # ---------------------------------------------------------------- ADD
    if extraction.cue is Cue.RETRACT:
        plans[Action.ADD] = ActionPlan(Action.ADD, False, "retraction_cannot_be_added", tokens_after=used)
    else:
        new = build_record(extraction, user_id=state.user_id, day=day, turn=turn, mem_cfg=mem_cfg, catalog=catalog)
        fits, evict = _fit(records, new, budget, overflow)
        if fits:
            after = used + new.token_cost - sum(r.token_cost for r in evict)
            plans[Action.ADD] = ActionPlan(Action.ADD, True, "ok", insert=new, evict=evict, tokens_after=after)
        else:
            plans[Action.ADD] = ActionPlan(Action.ADD, False, "over_budget", tokens_after=used)

    # ---------------------------------------------------------------- UPDATE
    plans[Action.UPDATE] = _plan_update(state, extraction, turn, budget, catalog, mem_cfg, overflow)

    # ---------------------------------------------------------------- DELETE
    value = catalog.canonical_value(extraction.attribute, extraction.value) if extraction.value is not None else None
    targets = [
        r
        for r in records
        if r.live(day) and r.attribute == extraction.attribute and (value is None or r.value == value)
    ]
    if targets:
        plans[Action.DELETE] = ActionPlan(
            Action.DELETE,
            True,
            "ok",
            delete=targets,
            tokens_after=used - sum(r.token_cost for r in targets),
        )
    else:
        plans[Action.DELETE] = ActionPlan(Action.DELETE, False, "no_target", tokens_after=used)
    return plans


def _plan_update(
    state: TurnState,
    extraction: Extraction,
    turn: int,
    budget: int,
    catalog: Catalog,
    mem_cfg: MemoryConfig,
    overflow: Overflow,
) -> ActionPlan:
    used = state.tokens
    if extraction.cue is Cue.RETRACT:
        return ActionPlan(Action.UPDATE, False, "retraction_cannot_update", tokens_after=used)
    if catalog.is_multi(extraction.attribute):
        return ActionPlan(Action.UPDATE, False, "update_not_applicable", tokens_after=used)
    day = state.day
    kind = CUE_TO_RECORD_KIND[extraction.cue]
    targets = live_targets(state.records, extraction.attribute, kind, day)
    if not targets:
        return ActionPlan(Action.UPDATE, False, "no_target", tokens_after=used)
    new = build_record(extraction, user_id=state.user_id, day=day, turn=turn, mem_cfg=mem_cfg, catalog=catalog)
    if all(t.value == new.value for t in targets):
        return ActionPlan(Action.UPDATE, False, "no_change", tokens_after=used)

    target_ids = {t.id for t in targets}
    supersede: list[MemoryRecord] = []
    delete: list[MemoryRecord] = []
    if kind is RecordKind.PERMANENT:
        for t in targets:
            supersede.append(
                t.clone(
                    status=RecordStatus.HISTORICAL,
                    valid_to=day,
                    updated_day=day,
                    token_cost=record_token_cost(
                        t.attribute, t.value, t.kind, RecordStatus.HISTORICAL, t.valid_from, day
                    ),
                )
            )
        delete.extend(_prune_history(state.records, supersede, extraction.attribute, mem_cfg.history_depth))
    else:
        delete.extend(targets)

    removed_ids = target_ids | {r.id for r in delete}
    survivors = [r for r in state.records if r.id not in removed_ids] + supersede
    fits, evict = _fit(survivors, new, budget, overflow)
    if not fits:
        return ActionPlan(Action.UPDATE, False, "over_budget", tokens_after=used)
    evicted_ids = {r.id for r in evict}
    # An evicted record may be one we just superseded; do not both update and delete it.
    supersede = [r for r in supersede if r.id not in evicted_ids]
    after = sum(r.token_cost for r in survivors if r.id not in evicted_ids) + new.token_cost
    return ActionPlan(
        Action.UPDATE,
        True,
        "ok",
        insert=new,
        supersede=supersede,
        delete=delete,
        evict=evict,
        tokens_after=after,
    )


def valid_mask(plans: dict[Action, ActionPlan]) -> np.ndarray:
    return np.array([plans[Action(i)].ok for i in range(N_ACTIONS)], dtype=bool)


# --------------------------------------------------------------------------- applying
def apply_action(
    store: MemoryStore,
    state: TurnState,
    plans: dict[Action, ActionPlan],
    action: Action,
    extraction: Extraction | None,
    *,
    turn: int,
    catalog: Catalog,
    extra_event: dict[str, Any] | None = None,
) -> Outcome:
    """Apply `action`. An infeasible action is rejected and becomes a no-op (IGNORE)."""
    plan = plans[action]
    rejected = not plan.ok
    effective = Action.IGNORE if rejected else action
    if rejected:
        plan = plans[Action.IGNORE]

    removed = {r.id for r in plan.delete} | {r.id for r in plan.evict}
    if removed:
        store.delete_many(state.user_id, sorted(removed))
    for hist in plan.supersede:
        store.update(hist)
    inserted = store.insert(plan.insert) if plan.insert is not None else None

    superseded_ids = {r.id for r in plan.supersede}
    kept = [r for r in state.records if r.id not in removed and r.id not in superseded_ids]
    records_after = kept + list(plan.supersede) + ([inserted] if inserted else [])
    records_after.sort(key=lambda r: r.seq)

    applied = effective is not Action.IGNORE
    reason = plans[action].reason if rejected else ("ok" if applied else "ignored")
    outcome = Outcome(
        requested=action,
        effective=effective,
        applied=applied,
        rejected=rejected,
        reason=reason,
        inserted=inserted,
        superseded=list(plan.supersede),
        deleted=list(plan.delete),
        evicted=list(plan.evict),
        expired=list(state.expired),
        records_after=records_after,
        tokens_before=state.tokens,
        tokens_after=sum(r.token_cost for r in records_after),
    )

    event: dict[str, Any] = {
        "turn": turn,
        "day": state.day,
        "attribute": extraction.attribute if extraction else None,
        "value": (
            catalog.canonical_value(extraction.attribute, extraction.value)
            if extraction and extraction.value is not None
            else None
        ),
        "cue": extraction.cue.value if extraction else None,
        "confidence": extraction.confidence if extraction else None,
        "requested": action.name,
        "effective": effective.name,
        "applied": applied,
        "rejected": rejected,
        "reason": reason,
        "changes": outcome.changes(),
    }
    if extra_event:
        event.update(extra_event)
    store.append_event(state.user_id, event)
    return outcome
