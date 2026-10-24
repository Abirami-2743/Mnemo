"""Paired evaluation of memory-management policies, statistics and rule tuning."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .benchmark import BenchmarkGenerator, ExtractorNoise
from .catalog import Catalog
from .config import ExperimentConfig
from .engine import EpisodeEngine, EpisodeTrace
from .policies import OraclePolicy, Policy, RulePolicy, RuleThresholds

METRIC_KEYS = (
    "state_f1",
    "retention",
    "precision",
    "stale_rate",
    "junk_rate",
    "redundancy_rate",
    "contradiction_rate",
    "avg_tokens",
    "final_tokens",
    "qa_accuracy",
    "qa_final",
    "harmful_deletions",
    "rejected_frac",
    "return",
)
HIGHER_IS_BETTER = {
    "state_f1": True,
    "retention": True,
    "precision": True,
    "stale_rate": False,
    "junk_rate": False,
    "redundancy_rate": False,
    "contradiction_rate": False,
    "avg_tokens": False,
    "final_tokens": False,
    "qa_accuracy": True,
    "qa_final": True,
    "harmful_deletions": False,
    "rejected_frac": False,
    "return": True,
}


def summarize(trace: EpisodeTrace) -> dict[str, float]:
    """Per-episode metrics: state metrics are averaged over all turns once memory is non-trivial."""
    pt = [m for m in trace.per_turn if m.n_gt > 0] or trace.per_turn
    mean = lambda key: float(np.mean([getattr(m, key) for m in pt]))  # noqa: E731
    qa = [c.accuracy for c in trace.checkpoints]
    return {
        "state_f1": mean("f1"),
        "retention": mean("retention"),
        "precision": mean("precision"),
        "stale_rate": mean("stale_rate"),
        "junk_rate": mean("junk_rate"),
        "redundancy_rate": mean("redundancy_rate"),
        "contradiction_rate": mean("contradiction_rate"),
        "avg_tokens": mean("tokens"),
        "final_tokens": float(trace.per_turn[-1].tokens),
        "qa_accuracy": float(np.mean(qa)) if qa else float("nan"),
        "qa_final": qa[-1] if qa else float("nan"),
        "harmful_deletions": float(trace.harmful),
        "rejected_frac": trace.n_rejected / max(trace.n_decisions, 1),
        "return": trace.total_reward,
    }


@dataclass
class EvalResult:
    system: str
    episode_ids: list[int]
    metrics: dict[str, np.ndarray]  # key -> per-episode values
    by_profile: dict[str, list[int]]
    action_counts: dict[str, int]
    qa_by_kind: dict[str, float]


def evaluate_policy(
    policy: Policy,
    catalog: Catalog,
    cfg: ExperimentConfig,
    episode_ids: Sequence[int],
    *,
    budget: int | None = None,
    noise_scale: float = 1.0,
) -> EvalResult:
    gen = BenchmarkGenerator(catalog, cfg.benchmark)
    noise = ExtractorNoise(cfg.noise, catalog, noise_scale)
    rows: list[dict[str, float]] = []
    profiles: dict[str, list[int]] = {}
    actions = {"ADD": 0, "UPDATE": 0, "IGNORE": 0, "DELETE": 0}
    kind_acc: dict[str, list[tuple[float, int]]] = {}
    for pos, ep_id in enumerate(episode_ids):
        episode = gen.generate(ep_id)
        profiles.setdefault(episode.profile, []).append(pos)
        engine = EpisodeEngine(catalog, cfg, budget=budget, overflow=policy.overflow)
        policy.reset()
        if isinstance(policy, OraclePolicy):
            policy.bind(episode)
        dec = engine.start(episode, noise.extract_episode(episode))
        while dec is not None:
            engine.step(policy.act(dec))
            dec = engine.decision
        for a in engine.trace.actions:
            actions[a.name] += 1
        for ck in engine.trace.checkpoints:
            for k, (acc, n) in ck.by_kind.items():
                kind_acc.setdefault(k, []).append((acc, n))
        rows.append(summarize(engine.trace))
    metrics = {k: np.array([r[k] for r in rows], dtype=float) for k in METRIC_KEYS}
    qa_by_kind = {k: float(sum(a * n for a, n in v) / sum(n for _, n in v)) for k, v in kind_acc.items()}
    return EvalResult(policy.name, list(episode_ids), metrics, profiles, actions, qa_by_kind)


def bootstrap_ci(values: np.ndarray, *, n: int, level: float, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI over episodes (NaNs dropped)."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(v.mean()), float(lo), float(hi)


def paired_diff(a: np.ndarray, b: np.ndarray, *, n: int, level: float, seed: int = 0) -> dict[str, float]:
    """Paired difference a-b with bootstrap CI and a sign-flip permutation p-value."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    mean, lo, hi = bootstrap_ci(d, n=n, level=level, seed=seed)
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (n + 1))
    return {"mean": mean, "lo": lo, "hi": hi, "p": p}


def tune_rule_policy(
    catalog: Catalog, cfg: ExperimentConfig, episode_ids: Sequence[int]
) -> tuple[RulePolicy, dict[str, float]]:
    """Coordinate-descent over the confidence thresholds using validation state-F1 + QA."""
    grid = cfg.evaluation.rule_tuning.grid

    def objective(th: RuleThresholds) -> float:
        res = evaluate_policy(RulePolicy(catalog, th), catalog, cfg, episode_ids)
        return float(np.nanmean(res.metrics["return"]))

    best = RuleThresholds()
    best_score = objective(best)
    for _ in range(cfg.evaluation.rule_tuning.passes):
        for field_name in ("state", "change", "retract"):
            for g in grid:
                cand = RuleThresholds(**{**best.as_dict(), field_name: g})
                score = objective(cand)
                if score > best_score + 1e-9:
                    best, best_score = cand, score
    return RulePolicy(catalog, best, name="rule-based"), {
        **best.as_dict(),
        "val_return": best_score,
    }


def sweep(
    policies: Sequence[Policy],
    catalog: Catalog,
    cfg: ExperimentConfig,
    episode_ids: Sequence[int],
    axis: str,
    values: Sequence[float],
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[float, EvalResult]]:
    out: dict[str, dict[float, EvalResult]] = {p.name: {} for p in policies}
    for v, pol in itertools.product(values, policies):
        kwargs = {"budget": int(v)} if axis == "budget" else {"noise_scale": float(v)}
        out[pol.name][v] = evaluate_policy(pol, catalog, cfg, episode_ids, **kwargs)
        if progress:
            progress(f"{axis}={v} {pol.name}")
    return out
