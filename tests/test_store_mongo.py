from __future__ import annotations

import mongomock
import pytest

from mnemo.domain import Action, Cue, Extraction
from mnemo.operations import apply_action, begin_turn, plan_all
from mnemo.store import InMemoryStore, MongoStore


def _script(store, cfg, catalog):
    steps = [
        (Extraction(attribute="location", value="Chennai", cue=Cue.STATE, confidence=0.9), Action.ADD, 1),
        (Extraction(attribute="location", value="Pune", cue=Cue.CHANGE, confidence=0.9), Action.UPDATE, 3),
        (Extraction(attribute="hobby", value="chess", cue=Cue.STATE, confidence=0.9), Action.ADD, 4),
        (
            Extraction(attribute="employer", value="Zoho", cue=Cue.TEMPORARY, confidence=0.9, duration_days=2),
            Action.ADD,
            5,
        ),
        (Extraction(attribute="hobby", value="chess", cue=Cue.RETRACT, confidence=0.9), Action.DELETE, 9),
    ]
    snaps = []
    for turn, (e, a, day) in enumerate(steps):
        st = begin_turn(store, "u1", day)
        pl = plan_all(st, e, turn=turn, budget=200, catalog=catalog, mem_cfg=cfg.memory)
        apply_action(store, st, pl, a, e, turn=turn, catalog=catalog)
        snaps.append([(r.seq, r.attribute, r.value, r.kind, r.status, r.valid_to) for r in store.get_records("u1")])
    return snaps


def test_mongo_store_matches_in_memory_store(cfg, catalog):
    mongo = MongoStore(mongomock.MongoClient(), "t")
    mongo.ensure_indexes() if False else None  # index DDL is exercised against a real server in CI
    assert _script(mongo, cfg, catalog) == _script(InMemoryStore(), cfg, catalog)
    assert mongo.claim_stats("u1", "location", "Chennai") == (1, 1)
    assert mongo.ping()


def test_mongo_reset_and_isolation(cfg, catalog):
    mongo = MongoStore(mongomock.MongoClient(), "t")
    _script(mongo, cfg, catalog)
    assert mongo.get_records("other") == []
    mongo.reset_user("u1")
    assert mongo.get_records("u1") == [] and mongo.events("u1") == []


def test_mongo_update_unknown_record_raises(cfg, catalog):
    mongo = MongoStore(mongomock.MongoClient(), "t")
    _script(mongo, cfg, catalog)
    rec = mongo.get_records("u1")[0].clone(id="u1:999")
    with pytest.raises(KeyError):
        mongo.update(rec)


@pytest.mark.mongo
def test_live_mongodb_roundtrip(cfg, catalog):
    """Runs only when MONGODB_URI points at a real server (CI service container)."""
    import os

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI not set")
    store = MongoStore.from_uri(uri, "mnemo_test", 2000)
    store.ensure_indexes()
    store.reset_user("u1")
    assert _script(store, cfg, catalog) == _script(InMemoryStore(), cfg, catalog)
    store.reset_user("u1")
    store.close()
