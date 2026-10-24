"""Wrap an sb3-contrib MaskablePPO model as a memory policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain import Action
from ..engine import Decision
from ..features import feature_mask
from ..operations import Overflow


class PPOPolicy:
    overflow: Overflow = "reject"

    def __init__(
        self,
        model,
        *,
        name: str = "mnemo-ppo",
        deterministic: bool = True,
        ablate: tuple[str, ...] = (),
        seed: int = 0,
    ) -> None:
        self.model = model
        self.name = name
        self.deterministic = deterministic
        self.fmask = feature_mask(ablate)
        self.seed = seed
        self._rng_seeded = False

    def reset(self) -> None:
        return None

    def act(self, d: Decision) -> Action:
        if not self._rng_seeded:
            self.model.policy.set_training_mode(False)
            self._rng_seeded = True
        obs = (d.obs * self.fmask)[None, :]
        action, _ = self.model.predict(obs, action_masks=d.mask[None, :], deterministic=self.deterministic)
        return Action(int(action[0]))

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> PPOPolicy:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(str(path), device="cpu")
        return cls(model, **kwargs)


def make_untrained(catalog, cfg, *, seed: int, ablate: tuple[str, ...] = ()) -> PPOPolicy:
    """Same architecture and initialisation scheme as the trained model, zero training."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from .env import MemoryEnv

    env = DummyVecEnv([lambda: MemoryEnv(catalog, cfg, seed=seed)])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        policy_kwargs={"net_arch": list(cfg.ppo.net_arch)},
    )
    np.random.seed(seed)
    return PPOPolicy(model, name="untrained-policy", deterministic=False, ablate=ablate, seed=seed)
