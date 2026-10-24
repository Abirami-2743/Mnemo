"""Gymnasium environment: one episode = one synthetic user conversation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..benchmark import BenchmarkGenerator, ExtractorNoise
from ..catalog import Catalog
from ..config import ExperimentConfig
from ..domain import N_ACTIONS, Action
from ..engine import EpisodeEngine
from ..features import OBS_DIM, feature_mask


class MemoryEnv(gym.Env):
    """Observation = features of one extracted claim; action in {ADD, UPDATE, IGNORE, DELETE}.

    Invalid actions are masked (`action_masks`, for MaskablePPO); if one is sent anyway the
    engine rejects it and applies IGNORE. Budget and extractor-noise level are randomised per
    episode from the config so the policy does not overfit one operating point.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        catalog: Catalog,
        cfg: ExperimentConfig,
        *,
        split: str = "train",
        seed: int = 0,
        randomize: bool = True,
        ablate: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.catalog, self.cfg = catalog, cfg
        self.gen = BenchmarkGenerator(catalog, cfg.benchmark)
        self.split = split
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.fmask = feature_mask(ablate)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._engine: EpisodeEngine | None = None
        self._mask = np.ones(N_ACTIONS, dtype=bool)
        self._lo, self._hi = cfg.benchmark.splits[split]

    def _sample_setup(self) -> tuple[int, int, float]:
        ep_id = int(self.rng.integers(self._lo, self._hi))
        budget, scale = self.cfg.memory.budget_tokens, 1.0
        if self.randomize:
            budget = int(self.rng.choice(self.cfg.ppo.train_budgets))
            lo, hi = self.cfg.ppo.noise_scale_range
            scale = float(self.rng.uniform(lo, hi))
        return ep_id, budget, scale

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(20):  # an episode with zero claims is vanishingly rare; just resample
            ep_id, budget, scale = self._sample_setup()
            if options and "episode_id" in options:
                ep_id = int(options["episode_id"])
            episode = self.gen.generate(ep_id)
            noise = ExtractorNoise(self.cfg.noise, self.catalog, scale)
            self._engine = EpisodeEngine(self.catalog, self.cfg, budget=budget)
            dec = self._engine.start(episode, noise.extract_episode(episode))
            if dec is not None:
                self._mask = dec.mask
                return dec.obs * self.fmask, {}
        raise RuntimeError("could not sample an episode with at least one claim")

    def step(self, action: int):
        assert self._engine is not None
        res = self._engine.step(Action(int(action)))
        dec = self._engine.decision
        if dec is None:
            return (
                np.zeros(OBS_DIM, dtype=np.float32),
                res.reward,
                True,
                False,
                {"parts": res.reward_parts},
            )
        self._mask = dec.mask
        return dec.obs * self.fmask, res.reward, False, False, {"parts": res.reward_parts}

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()
