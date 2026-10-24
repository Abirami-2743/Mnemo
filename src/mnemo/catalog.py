"""Data-driven attribute catalog (loaded from configs/catalog.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import DEFAULT_CATALOG_PATH
from .domain import TurnKind

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_ALLOWED_PLACEHOLDERS: dict[str, set[str]] = {
    "state": {"v"},
    "change": {"v"},
    "retract": {"v"},
    "hypothetical": {"v"},
    "third_party": {"v"},
    "temporary": {"v", "d"},
    "plan": {"v", "d", "s"},
}
_TEMPLATE_KINDS = tuple(_ALLOWED_PLACEHOLDERS)


class CatalogError(ValueError):
    """Raised when the catalog file is inconsistent."""


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    label: str
    multi: bool
    importance: float
    values: tuple[str, ...]
    keywords: tuple[str, ...]
    templates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def supports(self, kind: TurnKind) -> bool:
        """Whether the benchmark can generate a turn of this kind for the attribute."""
        if kind is TurnKind.CHITCHAT:
            return True
        if kind is TurnKind.STATE_NEW or kind is TurnKind.STATE_REPEAT:
            return "state" in self.templates
        return kind.value in self.templates

    def question(self, probe_kind: str) -> str:
        """Natural-language probe question for this attribute."""
        if probe_kind == "current":
            return f"What is the user's {self.label}?"
        if probe_kind == "set":
            return f"What are the user's {self.label}?"
        if probe_kind == "previous":
            return f"What was the user's previous {self.label}, before the latest change?"
        if probe_kind == "temporary":
            return f"Does the user currently have a temporary {self.label}? If so, what is it?"
        if probe_kind == "plan":
            return f"Does the user have an upcoming planned {self.label}? If so, what is it?"
        raise ValueError(f"unknown probe kind: {probe_kind}")


class Catalog:
    def __init__(self, attributes: dict[str, AttributeSpec], chitchat: tuple[str, ...]) -> None:
        if not attributes:
            raise CatalogError("catalog defines no attributes")
        self.attributes = attributes
        self.chitchat = chitchat
        self.names: tuple[str, ...] = tuple(attributes)
        self._value_index: dict[str, tuple[str, str]] = {}
        for spec in attributes.values():
            for value in spec.values:
                key = value.lower()
                if key in self._value_index:
                    raise CatalogError(f"value {value!r} is used by more than one attribute")
                self._value_index[key] = (spec.name, value)

    def spec(self, attribute: str) -> AttributeSpec:
        try:
            return self.attributes[attribute]
        except KeyError as exc:
            raise KeyError(f"unknown attribute: {attribute!r}") from exc

    def has(self, attribute: str) -> bool:
        return attribute in self.attributes

    def is_multi(self, attribute: str) -> bool:
        return self.spec(attribute).multi

    def importance(self, attribute: str) -> float:
        return self.spec(attribute).importance

    def lookup_value(self, raw: str) -> tuple[str, str] | None:
        """Map a surface value to (attribute, canonical value), case-insensitively."""
        return self._value_index.get(raw.strip().lower())

    def canonical_value(self, attribute: str, raw: str) -> str:
        """Canonical spelling for catalog values; free-form values pass through trimmed."""
        hit = self.lookup_value(raw)
        if hit is not None and hit[0] == attribute:
            return hit[1]
        return raw.strip()

    @property
    def max_importance(self) -> float:
        return max(s.importance for s in self.attributes.values())


def load_catalog(path: str | Path | None = None) -> Catalog:
    p = Path(path) if path else DEFAULT_CATALOG_PATH
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "attributes" not in raw:
        raise CatalogError(f"{p} must contain an 'attributes' mapping")

    attributes: dict[str, AttributeSpec] = {}
    for name, body in raw["attributes"].items():
        cardinality = body.get("cardinality")
        if cardinality not in ("single", "multi"):
            raise CatalogError(f"{name}: cardinality must be 'single' or 'multi'")
        values = tuple(str(v) for v in body.get("values", ()))
        if len(values) < 2:
            raise CatalogError(f"{name}: needs at least two values")
        importance = float(body.get("importance", 0.0))
        if importance <= 0:
            raise CatalogError(f"{name}: importance must be > 0")
        templates: dict[str, tuple[str, ...]] = {}
        for kind, items in (body.get("templates") or {}).items():
            if kind not in _TEMPLATE_KINDS:
                raise CatalogError(f"{name}: unknown template kind {kind!r}")
            for tpl in items:
                found = set(_PLACEHOLDER.findall(tpl))
                if "v" not in found:
                    raise CatalogError(f"{name}.{kind}: template lacks {{v}}: {tpl!r}")
                extra = found - _ALLOWED_PLACEHOLDERS[kind]
                if extra:
                    raise CatalogError(f"{name}.{kind}: unexpected placeholders {sorted(extra)}")
                needed = _ALLOWED_PLACEHOLDERS[kind] - {"v"}
                if needed - found and kind == "plan":
                    raise CatalogError(f"{name}.{kind}: template must use {{s}} and {{d}}: {tpl!r}")
            templates[kind] = tuple(items)
        if "state" not in templates:
            raise CatalogError(f"{name}: a 'state' template is required")
        if cardinality == "multi" and "change" in templates:
            raise CatalogError(f"{name}: multi-valued attributes cannot define 'change'")
        if cardinality == "single" and "change" not in templates:
            raise CatalogError(f"{name}: single-valued attributes need a 'change' template")
        attributes[name] = AttributeSpec(
            name=name,
            label=str(body.get("label", name.replace("_", " "))),
            multi=cardinality == "multi",
            importance=importance,
            values=values,
            keywords=tuple(str(k).lower() for k in body.get("keywords", ())),
            templates=templates,
        )
    chitchat = tuple(str(x) for x in raw.get("chitchat", ()))
    if not chitchat:
        raise CatalogError("chitchat list must not be empty")
    return Catalog(attributes, chitchat)
