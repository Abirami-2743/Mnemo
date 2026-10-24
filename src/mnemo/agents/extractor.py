"""Claim extraction: free text -> structured `Extraction` (LLM with offline fallback)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..catalog import Catalog
from ..config import PROJECT_ROOT
from ..domain import Cue, Extraction
from .llm import GroqClient, LLMError

log = logging.getLogger(__name__)
CUES_PATH = PROJECT_ROOT / "configs" / "cues.yaml"


class HeuristicExtractor:
    """Deterministic, offline extractor driven by the catalog and `configs/cues.yaml`."""

    source = "heuristic"

    def __init__(self, catalog: Catalog, lexicon_path: Path | None = None) -> None:
        self.catalog = catalog
        raw = yaml.safe_load((lexicon_path or CUES_PATH).read_text())
        self._order = [Cue(c) for c in raw["order"]]
        self._patterns = {Cue(k): [re.compile(p, re.I) for p in v] for k, v in raw["patterns"].items()}
        self._dur = [re.compile(p, re.I) for p in raw["duration"]]
        self._starts = re.compile(raw["starts_in"], re.I)
        self._conf = raw["confidence"]
        values = sorted((v for s in catalog.attributes.values() for v in s.values), key=len, reverse=True)
        self._value_re = re.compile(r"(?<![\w-])(" + "|".join(re.escape(v) for v in values) + r")(?![\w-])", re.I)

    def extract(self, text: str) -> Extraction | None:
        m = self._value_re.search(text)
        if not m:
            return None
        hit = self.catalog.lookup_value(m.group(1))
        if hit is None:
            return None
        attribute, value = hit
        cue, explicit = Cue.STATE, False
        for candidate in self._order:
            if any(p.search(text) for p in self._patterns.get(candidate, [])):
                cue, explicit = candidate, True
                break
        duration = starts = None
        if cue in (Cue.TEMPORARY, Cue.PLAN):
            duration = next((int(m.group("d")) for p in self._dur if (m := p.search(text))), None)
        if cue is Cue.PLAN:
            sm = self._starts.search(text)
            starts = int(sm.group("s")) if sm else None
        return Extraction(
            attribute=attribute,
            value=value,
            cue=cue,
            confidence=self._conf["explicit"] if explicit else self._conf["default_state"],
            duration_days=duration,
            starts_in_days=starts,
            source=self.source,
        )


_SYSTEM = """You extract ONE memory-worthy claim about the USER from a chat message.
Return JSON: {"attribute": <one of ATTRS or null>, "value": <string or null>,
"cue": <state|change|temporary|plan|retract|hypothetical|third_party>,
"confidence": <0..1>, "duration_days": <int|null>, "starts_in_days": <int|null>}.
Cues: state = plain fact; change = permanent update of an earlier fact; temporary = holds only for
a duration; plan = will hold in the future; retract = user withdraws a fact; hypothetical =
imagined / conditional; third_party = about someone else. If the message carries no fact about the
user's attributes, return {"attribute": null}. ATTRS (name: allowed values): %s"""


class LLMExtractor:
    """Groq-backed extractor that falls back to the heuristic one on any failure."""

    source = "llm"

    def __init__(self, client: GroqClient, catalog: Catalog, fallback: HeuristicExtractor) -> None:
        self.client, self.catalog, self.fallback = client, catalog, fallback
        spec = "; ".join(
            f"{n}: {', '.join(s.values)}{' (multi)' if s.multi else ''}" for n, s in catalog.attributes.items()
        )
        self._system = _SYSTEM % spec

    async def extract(self, text: str) -> Extraction | None:
        try:
            data = await self.client.complete_json(self._system, text)
            return self._parse(data)
        except (LLMError, ValueError, KeyError) as exc:
            log.warning("LLM extraction failed (%s); using heuristic fallback", exc)
            return self.fallback.extract(text)

    def _parse(self, data: dict[str, Any]) -> Extraction | None:
        attr = data.get("attribute")
        if not attr:
            return None
        if not self.catalog.has(attr):
            raise ValueError(f"unknown attribute {attr!r}")
        value = data.get("value")
        cue = Cue(data["cue"])
        if value is not None:
            value = self.catalog.canonical_value(attr, str(value))
        return Extraction(
            attribute=attr,
            value=value,
            cue=cue,
            confidence=float(min(max(data.get("confidence", 0.7), 0.0), 1.0)),
            duration_days=data.get("duration_days") or None,
            starts_in_days=data.get("starts_in_days"),
            source=self.source,
        )


def dumps(extraction: Extraction | None) -> str:
    return json.dumps(extraction.model_dump(mode="json") if extraction else None)
