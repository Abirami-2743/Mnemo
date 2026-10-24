"""Simulated extractor noise.

Real extractors (LLM or rule-based) misread cues, hallucinate facts, miss facts,
and report confidences that are informative but imperfect. The RL policy is
trained against a controllable model of exactly that, and evaluated across noise
levels (`scale`) so conclusions are not tied to one error rate.
"""

from __future__ import annotations

import numpy as np

from ..catalog import Catalog
from ..config import NoiseConfig
from ..domain import KIND_TO_CUE, Cue, Extraction
from .generator import Episode, Truth

_NOISE_STREAM = 7919


class ExtractorNoise:
    def __init__(self, cfg: NoiseConfig, catalog: Catalog, scale: float = 1.0) -> None:
        if scale < 0:
            raise ValueError("noise scale must be >= 0")
        self.cfg = cfg
        self.catalog = catalog
        self.scale = scale

    def _p(self, p: float) -> float:
        return min(1.0, p * self.scale)

    def _confusion_row(self, true_cue: Cue) -> list[tuple[Cue, float]]:
        row = [(Cue(k), self._p(p)) for k, p in self.cfg.confusion.get(true_cue.value, {}).items()]
        total = sum(p for _, p in row)
        if total > 1.0:
            row = [(c, p / total) for c, p in row]
        return row

    def _confidence(self, rng: np.random.Generator, wrong: bool) -> float:
        a, b = self.cfg.conf_wrong if wrong else self.cfg.conf_correct
        return float(np.clip(rng.beta(a, b), 0.01, 0.99))

    def extract(self, truth: Truth, rng: np.random.Generator) -> Extraction | None:
        true_cue = KIND_TO_CUE[truth.kind]
        if true_cue is None:
            if rng.random() >= self._p(self.cfg.spurious_rate):
                return None
            cues = [Cue(k) for k in self.cfg.spurious_cues]
            probs = np.array([self.cfg.spurious_cues[c.value] for c in cues], dtype=float)
            cue = cues[int(rng.choice(len(cues), p=probs / probs.sum()))]
            attr = self.catalog.names[int(rng.integers(len(self.catalog.names)))]
            values = self.catalog.spec(attr).values
            value = values[int(rng.integers(len(values)))]
            return Extraction(
                attribute=attr,
                value=value,
                cue=cue,
                confidence=self._confidence(rng, wrong=True),
                source="sim",
            )

        assert truth.attribute is not None
        if rng.random() < self._p(self.cfg.miss_rate):
            return None

        cue = true_cue
        wrong = False
        draw = rng.random()
        acc = 0.0
        for candidate, p in self._confusion_row(true_cue):
            acc += p
            if draw < acc:
                cue, wrong = candidate, True
                break

        value = truth.value
        if rng.random() < self._p(self.cfg.value_error_rate):
            options = [v for v in self.catalog.spec(truth.attribute).values if v != value]
            value = options[int(rng.integers(len(options)))]
            wrong = True

        duration = truth.duration_days if cue in (Cue.TEMPORARY, Cue.PLAN) else None
        starts = truth.starts_in_days if cue is Cue.PLAN else None
        return Extraction(
            attribute=truth.attribute,
            value=value,
            cue=cue,
            confidence=self._confidence(rng, wrong),
            duration_days=duration,
            starts_in_days=starts,
            source="sim",
        )

    def extract_episode(self, episode: Episode) -> list[Extraction | None]:
        """Deterministic noise realisation for an episode (identical for every system compared)."""
        rng = np.random.default_rng([episode.episode_id, _NOISE_STREAM])
        return [self.extract(t.truth, rng) for t in episode.turns]
