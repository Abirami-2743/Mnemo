"""Typed configuration: experiment parameters (YAML) and service settings (env)."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "configs" / "catalog.yaml"

TURN_KINDS = (
    "state_new",
    "state_repeat",
    "change",
    "temporary",
    "plan",
    "retract",
    "hypothetical",
    "third_party",
    "chitchat",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryConfig(_Strict):
    budget_tokens: int = Field(140, gt=0)
    default_temp_ttl: int = Field(7, gt=0)
    default_plan_start: int = Field(7, ge=0)
    default_plan_duration: int = Field(7, gt=0)
    max_records_norm: int = Field(40, gt=0)
    history_depth: int = Field(1, ge=0)


class BenchmarkConfig(_Strict):
    min_turns: int = Field(36, gt=0)
    max_turns: int = Field(56, gt=0)
    gap_days: tuple[int, int] = (1, 6)
    temp_duration: tuple[int, int] = (3, 14)
    plan_start: tuple[int, int] = (3, 20)
    plan_duration: tuple[int, int] = (3, 10)
    attributes_per_episode: tuple[int, int] = (6, 10)
    checkpoint_every: int = Field(12, gt=0)
    probes_per_checkpoint: int = Field(8, gt=0)
    confuser_value_prob: float = Field(0.15, ge=0, le=1)
    attribute_arrival_floor: float = Field(0.5, ge=0)
    profiles: dict[str, dict[str, float]]
    profile_weights: dict[str, float]
    ood_profile_weights: dict[str, float]
    splits: dict[str, tuple[int, int]]

    @model_validator(mode="after")
    def _check(self) -> BenchmarkConfig:
        if self.min_turns > self.max_turns:
            raise ValueError("min_turns must be <= max_turns")
        for name, mix in self.profiles.items():
            unknown = set(mix) - set(TURN_KINDS)
            if unknown:
                raise ValueError(f"profile {name!r} has unknown turn kinds: {sorted(unknown)}")
            if sum(mix.values()) <= 0:
                raise ValueError(f"profile {name!r} has no probability mass")
        for weights in (self.profile_weights, self.ood_profile_weights):
            missing = set(weights) - set(self.profiles)
            if missing:
                raise ValueError(f"unknown profiles referenced: {sorted(missing)}")
        for split in ("train", "val", "test", "ood"):
            if split not in self.splits:
                raise ValueError(f"splits must define {split!r}")
        ranges = sorted(self.splits.values())
        for (_, hi), (lo, _) in pairwise(ranges):
            if hi > lo:
                raise ValueError("split seed ranges must be disjoint")
        return self


class NoiseConfig(_Strict):
    miss_rate: float = Field(0.03, ge=0, le=1)
    spurious_rate: float = Field(0.03, ge=0, le=1)
    value_error_rate: float = Field(0.02, ge=0, le=1)
    confusion: dict[str, dict[str, float]]
    spurious_cues: dict[str, float]
    conf_correct: tuple[float, float] = (8.0, 2.0)
    conf_wrong: tuple[float, float] = (3.0, 3.0)


class RewardConfig(_Strict):
    history_weight: float = Field(0.25, ge=0)
    temporary_weight: float = Field(0.6, ge=0)
    stale_penalty: float = Field(1.0, ge=0)
    junk_penalty: float = Field(1.0, ge=0)
    duplicate_penalty: float = Field(0.5, ge=0)
    token_cost: float = Field(0.05, ge=0)
    qa_weight: float = Field(1.0, ge=0)


class PPOConfig(_Strict):
    total_timesteps: int = Field(800_000, gt=0)
    n_envs: int = Field(8, gt=0)
    n_steps: int = Field(256, gt=0)
    batch_size: int = Field(256, gt=0)
    n_epochs: int = Field(10, gt=0)
    learning_rate: float = Field(3e-4, gt=0)
    lr_final_fraction: float = Field(0.1, gt=0, le=1)
    gamma: float = Field(0.99, gt=0, le=1)
    gae_lambda: float = Field(0.95, gt=0, le=1)
    clip_range: float = Field(0.2, gt=0)
    ent_coef: float = Field(0.01, ge=0)
    vf_coef: float = Field(0.5, ge=0)
    max_grad_norm: float = Field(0.5, gt=0)
    net_arch: list[int] = [128, 128]
    validation_every: int = Field(40_000, gt=0)
    validation_episodes: int = Field(100, gt=0)
    seeds: list[int] = [0, 1, 2]
    train_budgets: list[int] = [80, 110, 140, 180, 240]
    noise_scale_range: tuple[float, float] = (0.5, 1.5)


class RuleTuningConfig(_Strict):
    episodes: int = Field(120, gt=0)
    grid: list[float] = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    passes: int = Field(2, gt=0)


class EvaluationConfig(_Strict):
    test_episodes: int = Field(500, gt=0)
    ood_episodes: int = Field(300, gt=0)
    sweep_episodes: int = Field(200, gt=0)
    noise_scales: list[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
    budgets: list[int] = [60, 100, 140, 200, 400]
    bootstrap_samples: int = Field(2000, gt=0)
    confidence_level: float = Field(0.95, gt=0, lt=1)
    rule_tuning: RuleTuningConfig = RuleTuningConfig()

    @field_validator("noise_scales")
    @classmethod
    def _non_negative(cls, v: list[float]) -> list[float]:
        if any(x < 0 for x in v):
            raise ValueError("noise scales must be >= 0")
        return v


class ExperimentConfig(_Strict):
    memory: MemoryConfig = MemoryConfig()
    benchmark: BenchmarkConfig
    noise: NoiseConfig
    reward: RewardConfig = RewardConfig()
    ppo: PPOConfig = PPOConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    def fingerprint(self) -> str:
        """Stable short hash of the full configuration (recorded next to every result)."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load and validate the experiment configuration from YAML."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)


class Settings(BaseSettings):
    """Runtime settings for the API service (environment variables / .env)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    store_backend: Literal["auto", "mongo", "memory"] = "auto"
    mongodb_uri: str | None = None
    mongodb_db: str = "mnemo"
    mongodb_timeout_ms: int = 3000

    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2

    model_path: Path = PROJECT_ROOT / "models" / "mnemo_ppo" / "best_model.zip"
    policy_path: Path = PROJECT_ROOT / "models" / "mnemo_ppo" / "policy.npz"  # torch-free inference weights
    config_path: Path = DEFAULT_CONFIG_PATH
    catalog_path: Path = DEFAULT_CATALOG_PATH
    results_dir: Path = PROJECT_ROOT / "results"
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_sessions: int = 200
    session_ttl_s: int = 3600
    max_text_chars: int = 500
    log_level: str = "INFO"

    @property
    def backend(self) -> Literal["mongo", "memory"]:
        if self.store_backend == "auto":
            return "mongo" if self.mongodb_uri else "memory"
        return self.store_backend

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        return self.groq_api_key is not None and bool(self.groq_api_key.get_secret_value())

    @model_validator(mode="after")
    def _validate_backend(self) -> Settings:
        if self.store_backend == "mongo" and not self.mongodb_uri:
            raise ValueError("STORE_BACKEND=mongo requires MONGODB_URI")
        return self
