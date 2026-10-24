"""Deterministic memory reader and probe scorer.

Downstream answer accuracy is measured with a reader that only sees the stored
records (never the conversation), so the score reflects memory quality and not
the reader's intelligence. The LLM QA agent in `mnemo.agents` plays the same
role in the live demo; both are scored against the same ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable

from .benchmark.generator import Probe
from .domain import MemoryRecord, RecordKind, RecordStatus

Answer = str | frozenset[str] | None


def answer_probe(records: Iterable[MemoryRecord], probe: Probe, day: int) -> Answer:
    """Answer a probe using only the stored records ("latest evidence wins")."""
    pool = [r for r in records if r.attribute == probe.attribute]
    if probe.kind == "set":
        return frozenset(r.value for r in pool if r.live(day) and r.kind is RecordKind.PERMANENT)
    if probe.kind == "previous":
        hist = [r for r in pool if r.status is RecordStatus.HISTORICAL]
        return max(hist, key=lambda r: (r.valid_to or 0, r.seq)).value if hist else None
    wanted = {
        "current": RecordKind.PERMANENT,
        "temporary": RecordKind.TEMPORARY,
        "plan": RecordKind.PLANNED,
    }[probe.kind]
    live = [r for r in pool if r.live(day) and r.kind is wanted]
    return max(live, key=lambda r: (r.source_turn, r.seq)).value if live else None


def score_answer(answer: Answer, expected: Answer) -> float:
    """1.0 for an exact match; sets get Jaccard credit; None only matches None."""
    if isinstance(expected, frozenset):
        got = answer if isinstance(answer, frozenset) else frozenset()
        union = got | expected
        return 1.0 if not union else len(got & expected) / len(union)
    return 1.0 if answer == expected else 0.0


def run_probes(records: Iterable[MemoryRecord], probes: Iterable[Probe], day: int) -> list[tuple[Probe, Answer, float]]:
    recs = list(records)
    out: list[tuple[Probe, Answer, float]] = []
    for probe in probes:
        ans = answer_probe(recs, probe, day)
        out.append((probe, ans, score_answer(ans, probe.expected)))
    return out
