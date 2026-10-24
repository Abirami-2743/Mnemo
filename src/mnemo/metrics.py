"""Memory-state quality: how well stored memory matches the ground-truth state.

The same scoring function drives the RL reward and the reported evaluation
metrics, so the policy is optimised for exactly what is later measured.

Matching key: (attribute, value, kind, status). A stored record is
  * matched    -- its key is in the ground-truth snapshot (counted once),
  * duplicate  -- additional copy of an already matched key,
  * stale      -- live but wrong *now* although the value was true at some point
                  (superseded / retracted / wrong kind, e.g. a temporary stay stored as home),
  * junk       -- the value was never true of the user (hypotheticals, third parties, noise).
Superseded-but-correct history that the ground truth does not list is neutral (it only costs
tokens). A historical record is never counted as stale: history is not an error.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .benchmark.generator import GTSnapshot
from .catalog import Catalog
from .config import RewardConfig
from .domain import MemoryRecord, RecordKind, RecordStatus

Key = tuple[str, str, RecordKind, RecordStatus]


def kind_weight(kind: RecordKind, status: RecordStatus, cfg: RewardConfig) -> float:
    if status is RecordStatus.HISTORICAL:
        return cfg.history_weight
    if kind is RecordKind.PERMANENT:
        return 1.0
    return cfg.temporary_weight


@dataclass(frozen=True, slots=True)
class StateScore:
    """Raw weighted components of a memory state."""

    gt_weight: float
    matched_weight: float
    stale_weight: float
    junk_weight: float
    duplicate_weight: float
    gt_items: int
    matched_items: int
    n_records: int
    n_live: int
    n_stale: int
    n_junk: int
    n_duplicate: int
    n_contradictions: int
    n_single_attrs: int
    tokens: int

    def value(self, cfg: RewardConfig, norm: float) -> float:
        raw = (
            self.matched_weight
            - cfg.stale_penalty * self.stale_weight
            - cfg.junk_penalty * self.junk_weight
            - cfg.duplicate_penalty * self.duplicate_weight
        )
        return raw / norm


@dataclass(frozen=True, slots=True)
class StateMetrics:
    """Reportable memory-state metrics, all in [0, 1] except tokens."""

    retention: float  # importance-weighted recall of ground-truth facts
    precision: float  # matched weight / (matched + stale + junk + duplicate weight)
    f1: float
    stale_rate: float  # stale live records / live records
    junk_rate: float
    redundancy_rate: float  # duplicate copies / all records
    contradiction_rate: float  # single-valued attrs with >1 distinct live permanent value
    tokens: int
    n_records: int
    n_gt: int

    def as_dict(self) -> dict[str, float]:
        return {
            "retention": self.retention,
            "precision": self.precision,
            "f1": self.f1,
            "stale_rate": self.stale_rate,
            "junk_rate": self.junk_rate,
            "redundancy_rate": self.redundancy_rate,
            "contradiction_rate": self.contradiction_rate,
            "tokens": float(self.tokens),
            "n_records": float(self.n_records),
        }


def _gt_index(snapshot: GTSnapshot, catalog: Catalog, cfg: RewardConfig) -> dict[Key, float]:
    return {
        (it.attribute, it.value, it.kind, it.status): catalog.importance(it.attribute)
        * kind_weight(it.kind, it.status, cfg)
        for it in snapshot.items
    }


def score_state(
    records: list[MemoryRecord],
    snapshot: GTSnapshot,
    day: int,
    catalog: Catalog,
    cfg: RewardConfig,
) -> StateScore:
    gt = _gt_index(snapshot, catalog, cfg)
    gt_weight = sum(gt.values())
    seen: Counter[Key] = Counter()
    matched_w = stale_w = junk_w = dup_w = 0.0
    n_stale = n_junk = n_dup = n_live = 0
    perm_values: dict[str, set[str]] = defaultdict(set)

    for r in records:
        live = r.live(day)
        if r.status is RecordStatus.CURRENT and not live:
            continue  # expired; will be purged at the start of the next turn
        key: Key = (r.attribute, r.value, r.kind, r.status)
        w = catalog.importance(r.attribute) * kind_weight(r.kind, r.status, cfg)
        if live:
            n_live += 1
            if r.kind is RecordKind.PERMANENT:
                perm_values[r.attribute].add(r.value)
        if key in gt:
            if seen[key] == 0:
                matched_w += gt[key]
            else:
                dup_w += w
                n_dup += 1
            seen[key] += 1
            continue
        ever = snapshot.ever_true.get(r.attribute, frozenset())
        if r.value in ever:
            if live:
                stale_w += w
                n_stale += 1
        else:
            junk_w += w
            n_junk += 1 if live else 0
            if not live:  # junk history still counts as junk weight but not a live record
                continue

    single = {a: v for a, v in perm_values.items() if not catalog.is_multi(a)}
    n_contra = sum(1 for v in single.values() if len(v) > 1)
    n_rec = sum(1 for r in records if r.status is RecordStatus.HISTORICAL or r.live(day))
    return StateScore(
        gt_weight=gt_weight,
        matched_weight=matched_w,
        stale_weight=stale_w,
        junk_weight=junk_w,
        duplicate_weight=dup_w,
        gt_items=len(gt),
        matched_items=len(seen),
        n_records=n_rec,
        n_live=n_live,
        n_stale=n_stale,
        n_junk=n_junk,
        n_duplicate=n_dup,
        n_contradictions=n_contra,
        n_single_attrs=len(single),
        tokens=sum(r.token_cost for r in records if r.status is RecordStatus.HISTORICAL or r.live(day)),
    )


def to_metrics(s: StateScore) -> StateMetrics:
    retention = s.matched_weight / s.gt_weight if s.gt_weight > 0 else 1.0
    denom = s.matched_weight + s.stale_weight + s.junk_weight + s.duplicate_weight
    precision = s.matched_weight / denom if denom > 0 else 1.0
    f1 = 0.0 if retention + precision == 0 else 2 * retention * precision / (retention + precision)
    return StateMetrics(
        retention=retention,
        precision=precision,
        f1=f1,
        stale_rate=s.n_stale / s.n_live if s.n_live else 0.0,
        junk_rate=s.n_junk / s.n_live if s.n_live else 0.0,
        redundancy_rate=s.n_duplicate / s.n_records if s.n_records else 0.0,
        contradiction_rate=s.n_contradictions / s.n_single_attrs if s.n_single_attrs else 0.0,
        tokens=s.tokens,
        n_records=s.n_records,
        n_gt=s.gt_items,
    )


def harmful_removals(removed: list[MemoryRecord], snapshot: GTSnapshot) -> int:
    """Records that were deleted/evicted although the ground truth still lists them."""
    keys = {(i.attribute, i.value, i.kind, i.status) for i in snapshot.items}
    return sum(1 for r in removed if (r.attribute, r.value, r.kind, r.status) in keys)
