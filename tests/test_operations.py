from __future__ import annotations

import pytest

from mnemo.domain import Action, Cue, Extraction, RecordKind, RecordStatus
from mnemo.operations import apply_action, begin_turn, plan_all


def ex(attr="location", value="Chennai", cue=Cue.STATE, **kw):
    return Extraction(attribute=attr, value=value, cue=cue, confidence=0.9, **kw)


def run(store, cfg, catalog, e, action, *, day=1, turn=0, budget=500, overflow="reject"):
    state = begin_turn(store, "u", day)
    plans = plan_all(state, e, turn=turn, budget=budget, catalog=catalog, mem_cfg=cfg.memory, overflow=overflow)
    return apply_action(store, state, plans, action, e, turn=turn, catalog=catalog)


def live(store, day):
    return [r for r in store.get_records("u") if r.live(day)]


def test_add_and_ignore(store, cfg, catalog):
    out = run(store, cfg, catalog, ex(), Action.ADD)
    assert out.applied and out.inserted.value == "Chennai"
    out = run(store, cfg, catalog, ex(value="Pune"), Action.IGNORE, turn=1)
    assert not out.applied and len(store.get_records("u")) == 1


def test_update_permanent_keeps_history(store, cfg, catalog):
    run(store, cfg, catalog, ex(), Action.ADD)
    out = run(store, cfg, catalog, ex(value="Bangalore", cue=Cue.CHANGE), Action.UPDATE, day=5, turn=1)
    assert out.applied
    recs = {r.value: r for r in store.get_records("u")}
    assert recs["Chennai"].status is RecordStatus.HISTORICAL and recs["Chennai"].valid_to == 5
    assert recs["Bangalore"].status is RecordStatus.CURRENT


def test_history_depth_is_bounded(store, cfg, catalog):
    for i, city in enumerate(["Chennai", "Pune", "Delhi", "Kochi"]):
        act = Action.ADD if i == 0 else Action.UPDATE
        run(store, cfg, catalog, ex(value=city, cue=Cue.CHANGE), act, day=i + 1, turn=i)
    hist = [r for r in store.get_records("u") if r.status is RecordStatus.HISTORICAL]
    assert len(hist) == cfg.memory.history_depth and hist[0].value == "Delhi"


def test_update_without_target_is_rejected_and_noop(store, cfg, catalog):
    out = run(store, cfg, catalog, ex(cue=Cue.CHANGE), Action.UPDATE)
    assert out.rejected and out.reason == "no_target" and out.effective is Action.IGNORE
    assert store.get_records("u") == []


def test_update_same_value_is_noop(store, cfg, catalog):
    run(store, cfg, catalog, ex(), Action.ADD)
    out = run(store, cfg, catalog, ex(), Action.UPDATE, turn=1)
    assert out.rejected and out.reason == "no_change"


def test_multi_attribute_cannot_update(store, cfg, catalog):
    run(store, cfg, catalog, ex("hobby", "chess"), Action.ADD)
    out = run(store, cfg, catalog, ex("hobby", "painting"), Action.UPDATE, turn=1)
    assert out.rejected and out.reason == "update_not_applicable"


def test_delete_removes_matching_value_only(store, cfg, catalog):
    run(store, cfg, catalog, ex("hobby", "chess"), Action.ADD)
    run(store, cfg, catalog, ex("hobby", "painting"), Action.ADD, turn=1)
    out = run(store, cfg, catalog, ex("hobby", "chess", Cue.RETRACT), Action.DELETE, turn=2)
    assert out.applied and [r.value for r in live(store, 1)] == ["painting"]


def test_retraction_cannot_be_added(store, cfg, catalog):
    out = run(store, cfg, catalog, ex(cue=Cue.RETRACT), Action.ADD)
    assert out.rejected and out.reason == "retraction_cannot_be_added"


def test_temporary_records_expire_and_are_purged(store, cfg, catalog):
    run(store, cfg, catalog, ex(cue=Cue.TEMPORARY, duration_days=3), Action.ADD, day=10)
    rec = store.get_records("u")[0]
    assert rec.kind is RecordKind.TEMPORARY and rec.valid_to == 13
    state = begin_turn(store, "u", 13)
    assert [r.id for r in state.expired] == [rec.id] and store.get_records("u") == []


def test_planned_record_window(store, cfg, catalog):
    run(store, cfg, catalog, ex(cue=Cue.PLAN, duration_days=4, starts_in_days=6), Action.ADD, day=2)
    rec = store.get_records("u")[0]
    assert (rec.valid_from, rec.valid_to) == (8, 12)


def test_temporary_update_replaces_without_history(store, cfg, catalog):
    run(store, cfg, catalog, ex(cue=Cue.TEMPORARY, duration_days=5), Action.ADD, day=1)
    run(store, cfg, catalog, ex(value="Pune", cue=Cue.TEMPORARY, duration_days=5), Action.UPDATE, day=2, turn=1)
    recs = store.get_records("u")
    assert [(r.value, r.status) for r in recs] == [("Pune", RecordStatus.CURRENT)]


def test_budget_rejects_when_full_and_evicts_history_first(store, cfg, catalog):
    run(store, cfg, catalog, ex(), Action.ADD, budget=40)
    out = run(store, cfg, catalog, ex("employer", "Zoho"), Action.ADD, budget=12, turn=1)
    assert out.rejected and out.reason == "over_budget"
    run(store, cfg, catalog, ex(value="Pune", cue=Cue.CHANGE), Action.UPDATE, day=2, turn=2, budget=40)
    out = run(store, cfg, catalog, ex("employer", "Zoho"), Action.ADD, day=3, turn=3, budget=17)
    assert out.applied and out.evicted and out.evicted[0].status is RecordStatus.HISTORICAL


def test_evict_oldest_overflow_mode(store, cfg, catalog):
    for i, v in enumerate(["chess", "painting", "hiking", "gaming"]):
        out = run(store, cfg, catalog, ex("hobby", v), Action.ADD, turn=i, budget=24, overflow="evict_oldest")
        assert out.applied
    values = [r.value for r in store.get_records("u")]
    assert "chess" not in values and "gaming" in values
    assert sum(r.token_cost for r in store.get_records("u")) <= 24


def test_event_log_and_claim_stats(store, cfg, catalog):
    run(store, cfg, catalog, ex(), Action.ADD)
    run(store, cfg, catalog, ex(), Action.IGNORE, turn=1)
    assert store.claim_stats("u", "location", "Chennai") == (2, 1)
    assert len(store.events("u")) == 2


@pytest.mark.parametrize("action", list(Action))
def test_plans_never_exceed_budget(store, cfg, catalog, action):
    run(store, cfg, catalog, ex(), Action.ADD, budget=30)
    state = begin_turn(store, "u", 1)
    plans = plan_all(state, ex("employer", "Zoho"), turn=1, budget=30, catalog=catalog, mem_cfg=cfg.memory)
    if plans[action].ok:
        assert plans[action].tokens_after <= 30
