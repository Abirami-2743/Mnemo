"""PPO training (sb3-contrib MaskablePPO) with validation-based checkpoint selection."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ..benchmark import BenchmarkGenerator
from ..catalog import load_catalog
from ..config import PROJECT_ROOT, ExperimentConfig, load_experiment_config
from ..evaluation import evaluate_policy
from .env import MemoryEnv
from .policy import PPOPolicy

log = logging.getLogger("mnemo.train")


class ValidationCallback(BaseCallback):
    """Every `every` timesteps evaluate on the validation split; keep the best checkpoint."""

    def __init__(self, catalog, cfg: ExperimentConfig, out_dir: Path, ablate: tuple[str, ...]):
        super().__init__()
        self.catalog, self.cfg, self.out_dir, self.ablate = catalog, cfg, out_dir, ablate
        gen = BenchmarkGenerator(catalog, cfg.benchmark)
        self.val_ids = gen.episode_ids("val", cfg.ppo.validation_episodes)
        self.best = -np.inf
        self.history: list[dict[str, float]] = []
        self._next = cfg.ppo.validation_every

    def _validate(self) -> None:
        pol = PPOPolicy(self.model, ablate=self.ablate)
        res = evaluate_policy(pol, self.catalog, self.cfg, self.val_ids)
        score = float(np.nanmean(res.metrics["return"]))
        row = {
            "timesteps": float(self.num_timesteps),
            "val_return": score,
            "val_state_f1": float(np.nanmean(res.metrics["state_f1"])),
            "val_qa": float(np.nanmean(res.metrics["qa_accuracy"])),
        }
        self.history.append(row)
        log.info("t=%d val_return=%.3f f1=%.3f qa=%.3f", self.num_timesteps, score, row["val_state_f1"], row["val_qa"])
        if score > self.best:
            self.best = score
            self.model.save(str(self.out_dir / "best_model.zip"))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._validate()
            self._next += self.cfg.ppo.validation_every
        return True

    def _on_training_end(self) -> None:
        self._validate()


def _lr_schedule(initial: float, final_fraction: float):
    return lambda progress_remaining: initial * (final_fraction + (1 - final_fraction) * progress_remaining)


def train(
    cfg: ExperimentConfig,
    *,
    seed: int,
    out_dir: Path,
    ablate: tuple[str, ...] = (),
    token_cost: float | None = None,
    total_timesteps: int | None = None,
) -> dict:
    catalog = load_catalog()
    if token_cost is not None:
        cfg = cfg.model_copy(deep=True)
        cfg.reward.token_cost = token_cost
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    p = cfg.ppo
    env = VecMonitor(
        DummyVecEnv(
            [
                (lambda i=i: MemoryEnv(catalog, cfg, split="train", seed=seed * 1000 + i, ablate=ablate))
                for i in range(p.n_envs)
            ]
        )
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=_lr_schedule(p.learning_rate, p.lr_final_fraction),
        n_steps=p.n_steps,
        batch_size=p.batch_size,
        n_epochs=p.n_epochs,
        gamma=p.gamma,
        gae_lambda=p.gae_lambda,
        clip_range=p.clip_range,
        ent_coef=p.ent_coef,
        vf_coef=p.vf_coef,
        max_grad_norm=p.max_grad_norm,
        policy_kwargs={"net_arch": list(p.net_arch)},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    cb = ValidationCallback(catalog, cfg, out_dir, ablate)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps or p.total_timesteps, callback=cb)
    meta = {
        "seed": seed,
        "ablate": list(ablate),
        "token_cost": cfg.reward.token_cost,
        "timesteps": total_timesteps or p.total_timesteps,
        "train_seconds": round(time.time() - t0, 1),
        "best_val_return": cb.best,
        "config_fingerprint": cfg.fingerprint(),
        "history": cb.history,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a MNEMO PPO memory policy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--ablate", nargs="*", default=[], choices=["confidence", "support", "budget"])
    ap.add_argument("--token-cost", type=float, default=None, help="override reward.token_cost")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_experiment_config(args.config)
    out = Path(args.out) if args.out else PROJECT_ROOT / "models" / f"ppo_seed{args.seed}"
    meta = train(
        cfg,
        seed=args.seed,
        out_dir=out,
        ablate=tuple(args.ablate),
        token_cost=args.token_cost,
        total_timesteps=args.timesteps,
    )
    log.info("done: best val return %.3f in %.0fs", meta["best_val_return"], meta["train_seconds"])


if __name__ == "__main__":
    main()
