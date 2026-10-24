from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from mnemo.benchmark import BenchmarkGenerator
from mnemo.evaluation import bootstrap_ci, evaluate_policy, paired_diff
from mnemo.features import FEATURE_NAMES, OBS_DIM, feature_mask
from mnemo.policies import OraclePolicy, RandomMaskedPolicy, RulePolicy, StoreAllPolicy
from mnemo.rl.env import MemoryEnv


def test_env_passes_gymnasium_checker(catalog, cfg):
    env = MemoryEnv(catalog, cfg, seed=3)
    check_env(env, skip_render_check=True)


def test_env_masks_and_observations(catalog, cfg):
    env = MemoryEnv(catalog, cfg, seed=4)
    obs, _ = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(300):
        mask = env.action_masks()
        assert mask[2], "IGNORE must always be valid"
        obs, r, done, _, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all() and np.isfinite(r)
        if done:
            obs, _ = env.reset()


def test_feature_ablation_masks(catalog, cfg):
    m = feature_mask(("confidence",))
    assert m[FEATURE_NAMES.index("confidence")] == 0 and m[FEATURE_NAMES.index("prior_seen")] == 1


def test_policy_ordering_on_validation(catalog, cfg):
    g = BenchmarkGenerator(catalog, cfg.benchmark)
    ids = g.episode_ids("val", 40)
    f1 = {
        p.name: np.mean(evaluate_policy(p, catalog, cfg, ids).metrics["state_f1"])
        for p in (StoreAllPolicy(), RandomMaskedPolicy(0), RulePolicy(catalog), OraclePolicy(catalog, cfg))
    }
    assert f1["oracle"] > f1["rule-based"] > f1["store-all"]
    assert f1["oracle"] > f1["random"]


def test_evaluation_is_reproducible(catalog, cfg):
    ids = BenchmarkGenerator(catalog, cfg.benchmark).episode_ids("val", 10)
    a = evaluate_policy(RulePolicy(catalog), catalog, cfg, ids)
    b = evaluate_policy(RulePolicy(catalog), catalog, cfg, ids)
    for k in a.metrics:
        np.testing.assert_array_equal(a.metrics[k], b.metrics[k])


def test_bootstrap_and_paired_diff():
    x = np.linspace(0, 1, 50)
    m, lo, hi = bootstrap_ci(x, n=500, level=0.95)
    assert lo < m < hi
    d = paired_diff(x + 0.5, x, n=500, level=0.95)
    assert d["lo"] > 0 and d["p"] < 0.05
    assert paired_diff(x, x, n=500, level=0.95)["p"] > 0.9
