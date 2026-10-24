from __future__ import annotations

import pytest

from mnemo.benchmark import BenchmarkGenerator, ExtractorNoise
from mnemo.config import PROJECT_ROOT
from mnemo.engine import EpisodeEngine

MODEL = PROJECT_ROOT / "models" / "mnemo_ppo" / "best_model.zip"
NPZ = PROJECT_ROOT / "models" / "mnemo_ppo" / "policy.npz"


@pytest.mark.skipif(not (MODEL.exists() and NPZ.exists()), reason="trained model not present")
def test_numpy_policy_matches_sb3_on_every_decision(catalog, cfg):
    from mnemo.rl.numpy_policy import NumpyPolicy
    from mnemo.rl.policy import PPOPolicy

    ref, fast = PPOPolicy.load(MODEL), NumpyPolicy(NPZ)
    gen = BenchmarkGenerator(catalog, cfg.benchmark)
    noise = ExtractorNoise(cfg.noise, catalog)
    n = 0
    for eid in gen.episode_ids("val", 15):
        ep = gen.generate(eid)
        eng = EpisodeEngine(catalog, cfg)
        dec = eng.start(ep, noise.extract_episode(ep))
        while dec is not None:
            a = ref.act(dec)
            assert fast.act(dec) == a
            n += 1
            eng.step(a)
            dec = eng.decision
    assert n > 300
