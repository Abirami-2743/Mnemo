"""Deterministic token accounting for memory records.

A real tokenizer would tie results to one model family and add a heavyweight
dependency, so MNEMO uses the standard ~4 characters-per-token estimate for
English text on the exact line that would be injected into an LLM prompt.
The estimate is applied identically to every system under comparison.
"""

from __future__ import annotations

import math

from .domain import RecordKind, RecordStatus

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def render_memory_line(
    attribute: str,
    value: str,
    kind: RecordKind,
    status: RecordStatus,
    valid_from: int,
    valid_to: int | None,
) -> str:
    """The prompt-ready line for one memory record."""
    if status is RecordStatus.HISTORICAL:
        tag = "[previous]"
    elif kind is RecordKind.TEMPORARY:
        tag = f"[temporary until day {valid_to}]"
    elif kind is RecordKind.PLANNED:
        tag = f"[planned day {valid_from}-{valid_to}]"
    else:
        tag = "[current]"
    return f"{tag} {attribute}: {value}"


def record_token_cost(
    attribute: str,
    value: str,
    kind: RecordKind,
    status: RecordStatus,
    valid_from: int,
    valid_to: int | None,
) -> int:
    return estimate_tokens(render_memory_line(attribute, value, kind, status, valid_from, valid_to))
