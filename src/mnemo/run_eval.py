"""Full experiment: evaluate every system, sweeps, ablations; write results JSON/Markdown/plots."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from .benchmark import BenchmarkGenerator
from .catalog import load_catalog
from .config import PROJECT_ROOT, ExperimentConfig, load_experiment_config
from .evaluation import (
    METRIC_KEYS,
    EvalResult,
    bootstrap_ci,
    evaluate_policy,
    paired_diff,
    tune_rule_policy,
)
from .policies import OraclePolicy, RandomMaskedPolicy, StoreAllPolicy
from .rl.policy import PPOPolicy, make_untrained

log = logging.getLogger("mnemo.eval")
HEADLINE = (
    "state_f1",
    "retention",
    "precision",
    "stale_rate",
    "junk_rate",
    "contradiction_rate",
    "redundancy_rate",
    "avg_tokens",
    "qa_accuracy",
    "harmful_deletions",
)


def _stats(res: EvalResult, cfg: ExperimentConfig) -> dict[str, dict[str, float]]:
    ev = cfg.evaluation
    out = {}
    for k in METRIC_KEYS:
        m, lo, hi = bootstrap_ci(res.metrics[k], n=ev.bootstrap_samples, level=ev.confidence_level)
        out[k] = {"mean": m, "lo": lo, "hi": hi}
    return out


def _mean_result(name: str, results: list[EvalResult]) -> EvalResult:
    """Per-episode average over training seeds (paired by episode)."""
    metrics = {k: np.mean([r.metrics[k] for r in results], axis=0) for k in METRIC_KEYS}
    actions = {a: int(np.mean([r.action_counts[a] for r in results])) for a in results[0].action_counts}
    kinds = {k: float(np.mean([r.qa_by_kind.get(k, np.nan) for r in results])) for k in results[0].qa_by_kind}
    return EvalResult(name, results[0].episode_ids, metrics, results[0].by_profile, actions, kinds)


def _profile_table(res: EvalResult) -> dict[str, dict[str, float]]:
    return {
        prof: {k: float(np.nanmean(res.metrics[k][idx])) for k in ("state_f1", "qa_accuracy", "avg_tokens")}
        for prof, idx in res.by_profile.items()
    }


def run(
    cfg: ExperimentConfig,
    out_dir: Path,
    models_dir: Path,
    quick: bool = False,
    episodes: tuple[int | None, int | None, int | None] = (None, None, None),
) -> dict:
    t0 = time.time()
    catalog = load_catalog()
    gen = BenchmarkGenerator(catalog, cfg.benchmark)
    ev = cfg.evaluation
    n_test = episodes[0] or (40 if quick else ev.test_episodes)
    n_ood = episodes[1] or (30 if quick else ev.ood_episodes)
    n_sweep = episodes[2] or (20 if quick else ev.sweep_episodes)
    test_ids = gen.episode_ids("test", n_test)
    ood_ids = gen.episode_ids("ood", n_ood)
    sweep_ids = gen.episode_ids("test", n_sweep)

    # ---- systems
    log.info("tuning rule-based baseline on validation episodes")
    rule, rule_info = tune_rule_policy(catalog, cfg, gen.episode_ids("val", 30 if quick else ev.rule_tuning.episodes))
    seeds = sorted(p for p in models_dir.glob("ppo_seed*/best_model.zip"))
    if not seeds:
        raise SystemExit(f"no trained models found in {models_dir}; run scripts/train_all.sh first")
    ppo = [PPOPolicy.load(p, name=f"mnemo-ppo[{p.parent.name}]") for p in seeds]
    untrained = make_untrained(catalog, cfg, seed=0)
    fixed = [StoreAllPolicy(), RandomMaskedPolicy(0), untrained, rule, OraclePolicy(catalog, cfg)]

    def eval_all(ids, **kw) -> dict[str, EvalResult]:
        res = {p.name: evaluate_policy(p, catalog, cfg, ids, **kw) for p in fixed}
        per_seed = [evaluate_policy(p, catalog, cfg, ids, **kw) for p in ppo]
        res["mnemo-ppo"] = _mean_result("mnemo-ppo", per_seed)
        for r in per_seed:
            res[r.system] = r
        return res

    log.info("evaluating test (%d) and OOD (%d) episodes", n_test, n_ood)
    test = eval_all(test_ids)
    ood = eval_all(ood_ids)

    def table(res: dict[str, EvalResult]) -> dict:
        return {name: _stats(r, cfg) for name, r in res.items()}

    def diffs(res: dict[str, EvalResult]) -> dict:
        out = {}
        for ref in ("store-all", "rule-based", "untrained-policy"):
            out[f"mnemo-ppo vs {ref}"] = {
                k: paired_diff(
                    res["mnemo-ppo"].metrics[k], res[ref].metrics[k], n=ev.bootstrap_samples, level=ev.confidence_level
                )
                for k in HEADLINE
            }
        return out

    # ---- sweeps (noise robustness, budget pressure)
    sweep_systems = {"store-all": fixed[0], "rule-based": rule, "oracle": fixed[4]}
    sweeps: dict[str, dict] = {"noise": {}, "budget": {}}
    for axis, values, kw in (("noise", ev.noise_scales, "noise_scale"), ("budget", ev.budgets, "budget")):
        for v in values:
            log.info("sweep %s=%s", axis, v)
            arg = {kw: float(v) if axis == "noise" else int(v)}
            row = {n: evaluate_policy(p, catalog, cfg, sweep_ids, **arg) for n, p in sweep_systems.items()}
            row["mnemo-ppo"] = _mean_result(
                "mnemo-ppo", [evaluate_policy(p, catalog, cfg, sweep_ids, **arg) for p in ppo]
            )
            sweeps[axis][str(v)] = {
                n: {
                    k: float(np.nanmean(r.metrics[k]))
                    for k in ("state_f1", "qa_accuracy", "avg_tokens", "stale_rate", "retention")
                }
                for n, r in row.items()
            }

    # ---- ablations (single seed, shorter training)
    ablations = {}
    for name, flag in (("no_support", ("support",)), ("no_confidence", ("confidence",)), ("no_token_cost", ())):
        path = models_dir / f"abl_{name}" / "best_model.zip"
        if path.exists():
            r = evaluate_policy(PPOPolicy.load(path, name=name, ablate=flag), catalog, cfg, test_ids)
            ablations[name] = _stats(r, cfg)
    log.info("writing results")

    results = {
        "meta": {
            "config_fingerprint": cfg.fingerprint(),
            "n_test": n_test,
            "n_ood": n_ood,
            "n_sweep": n_sweep,
            "ppo_models": [str(p.parent.name) for p in seeds],
            "rule_thresholds": rule_info,
            "budget_tokens": cfg.memory.budget_tokens,
            "quick": quick,
            "eval_seconds": round(time.time() - t0, 1),
        },
        "test": table(test),
        "ood": table(ood),
        "paired_diffs_test": diffs(test),
        "paired_diffs_ood": diffs(ood),
        "by_profile_test": {
            n: _profile_table(r) for n, r in test.items() if n in ("store-all", "rule-based", "mnemo-ppo", "oracle")
        },
        "action_counts_test": {n: r.action_counts for n, r in test.items()},
        "qa_by_kind_test": {
            n: r.qa_by_kind for n, r in test.items() if n in ("store-all", "rule-based", "mnemo-ppo", "oracle")
        },
        "sweeps": sweeps,
        "ablations_test": ablations,
        "training": {
            p.parent.name: json.loads((p.parent / "train_meta.json").read_text())
            for p in seeds
            if (p.parent / "train_meta.json").exists()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    best = max(seeds, key=lambda p: json.loads((p.parent / "train_meta.json").read_text())["best_val_return"])
    dest = models_dir / "mnemo_ppo"
    dest.mkdir(exist_ok=True)
    shutil.copy(best, dest / "best_model.zip")
    from .report import write_report

    write_report(results, out_dir)
    log.info("done in %.0fs; serving model = %s", time.time() - t0, best.parent.name)
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate MNEMO against baselines")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--quick", action="store_true", help="small smoke evaluation")
    ap.add_argument("--test-episodes", type=int, default=None, help="override evaluation.test_episodes")
    ap.add_argument("--ood-episodes", type=int, default=None, help="override evaluation.ood_episodes")
    ap.add_argument("--sweep-episodes", type=int, default=None, help="override evaluation.sweep_episodes")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    run(
        load_experiment_config(a.config),
        Path(a.out),
        Path(a.models),
        a.quick,
        (a.test_episodes, a.ood_episodes, a.sweep_episodes),
    )


if __name__ == "__main__":
    main()
