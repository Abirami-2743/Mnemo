from __future__ import annotations

import numpy as np

from mnemo.benchmark import BenchmarkGenerator, ExtractorNoise
from mnemo.domain import KIND_TO_CUE, RecordStatus, TurnKind
from mnemo.metrics import score_state, to_metrics
from mnemo.operations import begin_turn
from mnemo.qa import answer_probe, score_answer


def test_generation_is_deterministic(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    ids = g.episode_ids("test", 5)
    assert [g.generate(i) for i in ids] == [g.generate(i) for i in ids]
    assert g.generate(ids[0]) != g.generate(ids[1])


def test_splits_are_disjoint(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    seen = {s: set(g.episode_ids(s, 50)) for s in ("train", "val", "test", "ood")}
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not seen[a] & seen[b]


def test_ground_truth_follows_turn_semantics(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    for eid in g.episode_ids("val", 40):
        ep = g.generate(eid)
        for t in ep.turns:
            k = t.truth.kind
            items = {(i.attribute, i.value, i.kind.value, i.status.value) for i in t.snapshot.items}
            if k in (TurnKind.HYPOTHETICAL, TurnKind.THIRD_PARTY) and t.truth.attribute:
                ever = t.snapshot.ever_true.get(t.truth.attribute, frozenset())
                held = (t.truth.attribute, t.truth.value, "permanent", "current") in items
                assert held == (t.truth.value in ever and held)  # never *adds* a fact
            if k is TurnKind.CHANGE:
                assert (t.truth.attribute, t.truth.value, "permanent", "current") in items
            if k is TurnKind.RETRACT and not catalog.is_multi(t.truth.attribute):
                assert not any(i[0] == t.truth.attribute and i[2] == "permanent" and i[3] == "current" for i in items)
        assert ep.checkpoints and ep.checkpoints[-1].turn_index == ep.horizon - 1


def test_all_turn_kinds_occur(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    kinds = {t.truth.kind for i in g.episode_ids("train", 60) for t in g.generate(i).turns}
    assert kinds == set(TurnKind)


def test_noise_zero_is_perfect_and_deterministic(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    ep = g.generate(g.episode_ids("test", 1)[0])
    clean = ExtractorNoise(cfg.noise, catalog, 0.0).extract_episode(ep)
    for t, e in zip(ep.turns, clean, strict=True):
        cue = KIND_TO_CUE[t.truth.kind]
        assert (e is None) == (cue is None)
        if e:
            assert (e.cue, e.attribute, e.value) == (cue, t.truth.attribute, t.truth.value)
    noisy = ExtractorNoise(cfg.noise, catalog, 2.0)
    assert noisy.extract_episode(ep) == noisy.extract_episode(ep)


def test_perfect_memory_scores_one_and_junk_lowers_it(catalog, cfg):
    from mnemo.domain import MemoryRecord, RecordKind

    g = BenchmarkGenerator(catalog, cfg.benchmark)
    ep = next(e for e in (g.generate(i) for i in g.episode_ids("test", 30)) if len(e.turns[-1].snapshot.items) > 3)
    snap = ep.turns[-1].snapshot
    recs = [
        MemoryRecord(
            f"r{n}",
            "u",
            n,
            it.attribute,
            it.value,
            it.kind,
            it.status,
            0,
            None if it.kind is RecordKind.PERMANENT else 10**6,
            0,
            0,
            0,
            0.9,
            8,
        )
        for n, it in enumerate(snap.items)
    ]
    m = to_metrics(score_state(recs, snap, ep.turns[-1].day, catalog, cfg.reward))
    assert np.isclose(m.retention, 1.0) and np.isclose(m.precision, 1.0)
    assert m.stale_rate == 0 and m.contradiction_rate == 0
    junk = recs[0].clone(id="j", seq=99, value="__never_true__")
    m2 = to_metrics(score_state([*recs, junk], snap, ep.turns[-1].day, catalog, cfg.reward))
    assert m2.precision < 1.0 and m2.junk_rate > 0
    dup = recs[0].clone(id="d", seq=100)
    assert to_metrics(score_state([*recs, dup], snap, ep.turns[-1].day, catalog, cfg.reward)).redundancy_rate > 0


def test_history_is_not_counted_as_stale(store, cfg, catalog):
    from mnemo.domain import Action, Cue, Extraction
    from mnemo.operations import apply_action, plan_all

    def go(e, a, day, turn):
        st = begin_turn(store, "u", day)
        pl = plan_all(st, e, turn=turn, budget=500, catalog=catalog, mem_cfg=cfg.memory)
        return apply_action(store, st, pl, a, e, turn=turn, catalog=catalog)

    go(Extraction(attribute="location", value="Chennai", cue=Cue.STATE, confidence=1), Action.ADD, 1, 0)
    out = go(Extraction(attribute="location", value="Pune", cue=Cue.CHANGE, confidence=1), Action.UPDATE, 2, 1)
    assert any(r.status is RecordStatus.HISTORICAL for r in out.records_after)
    from mnemo.benchmark import GTItem, GTSnapshot
    from mnemo.domain import RecordKind

    items = (
        GTItem("location", "Pune", RecordKind.PERMANENT, RecordStatus.CURRENT),
        GTItem("location", "Chennai", RecordKind.PERMANENT, RecordStatus.HISTORICAL),
    )
    snap = GTSnapshot(day=2, items=items, ever_true={"location": frozenset({"Chennai", "Pune"})})
    m = to_metrics(score_state(out.records_after, snap, 2, catalog, cfg.reward))
    assert m.stale_rate == 0.0 and m.contradiction_rate == 0.0 and np.isclose(m.f1, 1.0)
    # history the ground truth does not list is neutral, not an error
    bare = GTSnapshot(day=2, items=items[:1], ever_true=snap.ever_true)
    m2 = to_metrics(score_state(out.records_after, bare, 2, catalog, cfg.reward))
    assert m2.stale_rate == 0.0 and m2.junk_rate == 0.0


def test_qa_reader_and_scoring(store, cfg, catalog):
    from mnemo.benchmark.generator import Probe
    from mnemo.domain import Action, Cue, Extraction
    from mnemo.operations import apply_action, plan_all

    def go(e, a, day, turn):
        st = begin_turn(store, "u", day)
        pl = plan_all(st, e, turn=turn, budget=500, catalog=catalog, mem_cfg=cfg.memory)
        apply_action(store, st, pl, a, e, turn=turn, catalog=catalog)

    go(Extraction(attribute="location", value="Chennai", cue=Cue.STATE, confidence=1), Action.ADD, 1, 0)
    go(Extraction(attribute="location", value="Pune", cue=Cue.CHANGE, confidence=1), Action.UPDATE, 2, 1)
    go(Extraction(attribute="hobby", value="chess", cue=Cue.STATE, confidence=1), Action.ADD, 2, 2)
    recs = store.get_records("u")
    assert answer_probe(recs, Probe("current", "location", "Pune", ""), 2) == "Pune"
    assert answer_probe(recs, Probe("previous", "location", "Chennai", ""), 2) == "Chennai"
    assert answer_probe(recs, Probe("set", "hobby", frozenset({"chess"}), ""), 2) == frozenset({"chess"})
    assert score_answer(None, None) == 1.0 and score_answer("a", None) == 0.0
    assert score_answer(frozenset({"a"}), frozenset({"a", "b"})) == 0.5
    assert np.isclose(score_answer(frozenset(), frozenset()), 1.0)
