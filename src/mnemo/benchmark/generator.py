"""Procedural benchmark: synthetic user timelines with exact ground-truth memory state.

Every episode simulates a *latent user* (what is actually true about them) and a
stream of utterances about that user. After every turn the generator records the
memory an ideal agent would hold (`GTSnapshot`); at checkpoints it asks probe
questions whose expected answers follow from the latent state. Nothing here is
hard-coded to a particular attribute: phrasing, values and importance come from
the catalog and the turn-kind mixtures from the experiment config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..catalog import Catalog
from ..config import BenchmarkConfig
from ..domain import RecordKind, RecordStatus, TurnKind

PROBE_KINDS = ("current", "previous", "temporary", "plan", "set")


@dataclass(frozen=True, slots=True)
class GTItem:
    """One fact an ideal memory holds at some point in time."""

    attribute: str
    value: str
    kind: RecordKind
    status: RecordStatus


@dataclass(frozen=True, slots=True)
class GTSnapshot:
    day: int
    items: tuple[GTItem, ...]
    ever_true: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class Truth:
    """What a turn really was. Hidden from the policy; used for reward and analysis."""

    kind: TurnKind
    attribute: str | None = None
    value: str | None = None
    duration_days: int | None = None
    starts_in_days: int | None = None


@dataclass(frozen=True, slots=True)
class Probe:
    kind: str
    attribute: str
    expected: str | frozenset[str] | None
    question: str


@dataclass(frozen=True, slots=True)
class Turn:
    index: int
    day: int
    text: str
    truth: Truth
    snapshot: GTSnapshot


@dataclass(frozen=True, slots=True)
class Checkpoint:
    turn_index: int
    day: int
    probes: tuple[Probe, ...]


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: int
    profile: str
    turns: tuple[Turn, ...]
    checkpoints: tuple[Checkpoint, ...]

    @property
    def final_checkpoint(self) -> Checkpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    @property
    def horizon(self) -> int:
        return len(self.turns)


class _Latent:
    """The user's true state, evolving as the conversation proceeds."""

    def __init__(self) -> None:
        self.current: dict[str, str] = {}
        self.multi: dict[str, list[str]] = {}
        self.history: dict[str, str] = {}
        self.temps: dict[str, tuple[str, int]] = {}  # attr -> (value, end_day)
        self.plans: dict[str, tuple[str, int]] = {}  # attr -> (value, end_day)
        self.ever_true: dict[str, set[str]] = {}
        self.mentioned: set[str] = set()

    def expire(self, day: int) -> None:
        self.temps = {a: v for a, v in self.temps.items() if v[1] > day}
        self.plans = {a: v for a, v in self.plans.items() if v[1] > day}

    def mark_true(self, attribute: str, value: str) -> None:
        self.ever_true.setdefault(attribute, set()).add(value)

    def is_set(self, attribute: str, multi: bool) -> bool:
        return bool(self.multi.get(attribute)) if multi else attribute in self.current

    def snapshot(self, day: int) -> GTSnapshot:
        items: list[GTItem] = []
        for attr, value in self.current.items():
            items.append(GTItem(attr, value, RecordKind.PERMANENT, RecordStatus.CURRENT))
        for attr, values in self.multi.items():
            for value in values:
                items.append(GTItem(attr, value, RecordKind.PERMANENT, RecordStatus.CURRENT))
        for attr, (value, _) in self.temps.items():
            items.append(GTItem(attr, value, RecordKind.TEMPORARY, RecordStatus.CURRENT))
        for attr, (value, _) in self.plans.items():
            items.append(GTItem(attr, value, RecordKind.PLANNED, RecordStatus.CURRENT))
        for attr, value in self.history.items():
            items.append(GTItem(attr, value, RecordKind.PERMANENT, RecordStatus.HISTORICAL))
        return GTSnapshot(
            day=day,
            items=tuple(items),
            ever_true={a: frozenset(v) for a, v in self.ever_true.items()},
        )


class BenchmarkGenerator:
    def __init__(self, catalog: Catalog, cfg: BenchmarkConfig) -> None:
        self.catalog = catalog
        self.cfg = cfg

    # ------------------------------------------------------------------ splits
    def split_of(self, episode_id: int) -> str:
        for name, (lo, hi) in self.cfg.splits.items():
            if lo <= episode_id < hi:
                return name
        raise ValueError(f"episode id {episode_id} is outside every split range")

    def episode_ids(self, split: str, n: int) -> list[int]:
        lo, hi = self.cfg.splits[split]
        if n > hi - lo:
            raise ValueError(f"split {split!r} holds only {hi - lo} episodes (asked for {n})")
        return list(range(lo, lo + n))

    def _profile_weights(self, split: str) -> dict[str, float]:
        return self.cfg.ood_profile_weights if split == "ood" else self.cfg.profile_weights

    # ---------------------------------------------------------------- generate
    def generate(self, episode_id: int, profile: str | None = None) -> Episode:
        rng = np.random.default_rng([episode_id, 1_000_003])
        if profile is None:
            weights = self._profile_weights(self.split_of(episode_id))
            names = sorted(weights)
            p = np.array([weights[n] for n in names], dtype=float)
            profile = names[int(rng.choice(len(names), p=p / p.sum()))]
        mixture = self.cfg.profiles[profile]

        active = self._sample_active_attributes(rng)
        n_turns = int(rng.integers(self.cfg.min_turns, self.cfg.max_turns + 1))
        latent = _Latent()
        turns: list[Turn] = []
        checkpoints: list[Checkpoint] = []
        day = 0
        for i in range(n_turns):
            day += int(rng.integers(self.cfg.gap_days[0], self.cfg.gap_days[1] + 1))
            latent.expire(day)
            kind = self._sample_kind(rng, latent, active, mixture)
            truth, text = self._realise(rng, kind, latent, active, day)
            turns.append(Turn(i, day, text, truth, latent.snapshot(day)))
            is_last = i == n_turns - 1
            if is_last or (i + 1) % self.cfg.checkpoint_every == 0:
                probes = self._make_probes(rng, latent)
                if probes:
                    checkpoints.append(Checkpoint(i, day, probes))
        return Episode(episode_id, profile, tuple(turns), tuple(checkpoints))

    # ------------------------------------------------------------- sampling
    def _sample_active_attributes(self, rng: np.random.Generator) -> list[str]:
        names = list(self.catalog.names)
        k = int(
            rng.integers(
                self.cfg.attributes_per_episode[0],
                min(self.cfg.attributes_per_episode[1], len(names)) + 1,
            )
        )
        w = np.array([self.cfg.attribute_arrival_floor + self.catalog.importance(n) for n in names])
        picked = rng.choice(len(names), size=min(k, len(names)), replace=False, p=w / w.sum())
        return [names[int(j)] for j in picked]

    def _feasible(self, kind: TurnKind, latent: _Latent, active: Sequence[str]) -> bool:
        cat = self.catalog
        if kind is TurnKind.CHITCHAT:
            return True
        if kind is TurnKind.STATE_NEW:
            return any(self._unset_values(latent, a) for a in active)
        if kind in (TurnKind.HYPOTHETICAL, TurnKind.THIRD_PARTY):
            return any(cat.spec(a).supports(kind) for a in active)
        set_attrs = [a for a in active if latent.is_set(a, cat.is_multi(a))]
        if kind is TurnKind.STATE_REPEAT or kind is TurnKind.RETRACT:
            return any(cat.spec(a).supports(kind) for a in set_attrs)
        # change / temporary / plan only make sense for single-valued attributes
        return any(not cat.is_multi(a) and cat.spec(a).supports(kind) for a in set_attrs)

    def _unset_values(self, latent: _Latent, attribute: str) -> list[str]:
        spec = self.catalog.spec(attribute)
        if spec.multi:
            have = set(latent.multi.get(attribute, []))
            return [v for v in spec.values if v not in have]
        return [] if attribute in latent.current else list(spec.values)

    def _sample_kind(
        self,
        rng: np.random.Generator,
        latent: _Latent,
        active: Sequence[str],
        mixture: dict[str, float],
    ) -> TurnKind:
        kinds = [TurnKind(k) for k in mixture]
        weights = np.array(
            [mixture[k.value] if self._feasible(k, latent, active) else 0.0 for k in kinds],
            dtype=float,
        )
        return kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]

    def _pick_attribute(
        self,
        rng: np.random.Generator,
        candidates: Sequence[str],
        *,
        weighted: bool = False,
    ) -> str:
        if weighted:
            w = np.array([self.cfg.attribute_arrival_floor + self.catalog.importance(a) for a in candidates])
            return candidates[int(rng.choice(len(candidates), p=w / w.sum()))]
        return candidates[int(rng.integers(len(candidates)))]

    def _other_value(self, rng: np.random.Generator, attribute: str, avoid: set[str]) -> str:
        values = self.catalog.spec(attribute).values
        options = [v for v in values if v not in avoid] or list(values)
        return options[int(rng.integers(len(options)))]

    def _render(
        self,
        rng: np.random.Generator,
        attribute: str,
        template_key: str,
        value: str,
        duration: int | None = None,
        starts: int | None = None,
    ) -> str:
        templates = self.catalog.spec(attribute).templates[template_key]
        tpl = templates[int(rng.integers(len(templates)))]
        return tpl.format(v=value, d=duration, s=starts)

    # ------------------------------------------------------------ realisation
    def _realise(
        self,
        rng: np.random.Generator,
        kind: TurnKind,
        latent: _Latent,
        active: Sequence[str],
        day: int,
    ) -> tuple[Truth, str]:
        cat = self.catalog
        cfg = self.cfg

        if kind is TurnKind.CHITCHAT:
            text = cat.chitchat[int(rng.integers(len(cat.chitchat)))]
            return Truth(kind), text

        if kind is TurnKind.STATE_NEW:
            options = [a for a in active if self._unset_values(latent, a)]
            attr = self._pick_attribute(rng, options, weighted=True)
            pool = self._unset_values(latent, attr)
            value = pool[int(rng.integers(len(pool)))]
            if cat.is_multi(attr):
                latent.multi.setdefault(attr, []).append(value)
            else:
                latent.current[attr] = value
            latent.mark_true(attr, value)
            latent.mentioned.add(attr)
            return Truth(kind, attr, value), self._render(rng, attr, "state", value)

        set_attrs = [a for a in active if latent.is_set(a, cat.is_multi(a))]

        if kind is TurnKind.STATE_REPEAT:
            options = [a for a in set_attrs if cat.spec(a).supports(kind)]
            attr = self._pick_attribute(rng, options)
            if cat.is_multi(attr):
                pool = latent.multi[attr]
                value = pool[int(rng.integers(len(pool)))]
            else:
                value = latent.current[attr]
            return Truth(kind, attr, value), self._render(rng, attr, "state", value)

        if kind is TurnKind.CHANGE:
            options = [a for a in set_attrs if not cat.is_multi(a) and cat.spec(a).supports(kind)]
            attr = self._pick_attribute(rng, options, weighted=True)
            old = latent.current[attr]
            value = self._other_value(rng, attr, {old})
            latent.history[attr] = old
            latent.current[attr] = value
            latent.mark_true(attr, value)
            return Truth(kind, attr, value), self._render(rng, attr, "change", value)

        if kind is TurnKind.TEMPORARY or kind is TurnKind.PLAN:
            options = [a for a in set_attrs if not cat.is_multi(a) and cat.spec(a).supports(kind)]
            attr = self._pick_attribute(rng, options, weighted=True)
            value = self._other_value(rng, attr, {latent.current[attr]})
            duration = int(rng.integers(cfg.temp_duration[0], cfg.temp_duration[1] + 1))
            if kind is TurnKind.TEMPORARY:
                latent.temps[attr] = (value, day + duration)
                latent.mark_true(attr, value)
                text = self._render(rng, attr, "temporary", value, duration=duration)
                return Truth(kind, attr, value, duration_days=duration), text
            duration = int(rng.integers(cfg.plan_duration[0], cfg.plan_duration[1] + 1))
            starts = int(rng.integers(cfg.plan_start[0], cfg.plan_start[1] + 1))
            latent.plans[attr] = (value, day + starts + duration)
            latent.mark_true(attr, value)
            text = self._render(rng, attr, "plan", value, duration=duration, starts=starts)
            return (
                Truth(kind, attr, value, duration_days=duration, starts_in_days=starts),
                text,
            )

        if kind is TurnKind.RETRACT:
            options = [a for a in set_attrs if cat.spec(a).supports(kind)]
            attr = self._pick_attribute(rng, options)
            if cat.is_multi(attr):
                pool = latent.multi[attr]
                value = pool[int(rng.integers(len(pool)))]
                pool.remove(value)
            else:
                value = latent.current.pop(attr)
            return Truth(kind, attr, value), self._render(rng, attr, "retract", value)

        # hypothetical / third-party claims never change the user's true state
        options = [a for a in active if cat.spec(a).supports(kind)]
        attr = self._pick_attribute(rng, options)
        spec = cat.spec(attr)
        mine = set(latent.multi.get(attr, [])) if spec.multi else {latent.current.get(attr, "")} - {""}
        if mine and rng.random() < cfg.confuser_value_prob:
            pool = sorted(mine)
            value = pool[int(rng.integers(len(pool)))]
        else:
            value = self._other_value(rng, attr, mine)
        latent.mentioned.add(attr)
        return Truth(kind, attr, value), self._render(rng, attr, kind.value, value)

    # ----------------------------------------------------------------- probes
    def _make_probes(self, rng: np.random.Generator, latent: _Latent) -> tuple[Probe, ...]:
        cat = self.catalog
        candidates: list[tuple[Probe, float]] = []
        for attr in sorted(latent.mentioned | set(latent.current) | set(latent.multi)):
            spec = cat.spec(attr)
            w = spec.importance
            if spec.multi:
                expected = frozenset(latent.multi.get(attr, []))
                candidates.append((Probe("set", attr, expected, spec.question("set")), w))
                continue
            candidates.append((Probe("current", attr, latent.current.get(attr), spec.question("current")), w))
            if attr in latent.history:
                candidates.append((Probe("previous", attr, latent.history[attr], spec.question("previous")), w))
            if attr in latent.temps:
                candidates.append((Probe("temporary", attr, latent.temps[attr][0], spec.question("temporary")), w))
            if attr in latent.plans:
                candidates.append((Probe("plan", attr, latent.plans[attr][0], spec.question("plan")), w))
        if not candidates:
            return ()
        n = min(self.cfg.probes_per_checkpoint, len(candidates))
        w = np.array([c[1] for c in candidates])
        idx = rng.choice(len(candidates), size=n, replace=False, p=w / w.sum())
        return tuple(candidates[int(j)][0] for j in sorted(idx))
