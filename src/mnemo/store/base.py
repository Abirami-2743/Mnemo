"""Storage protocol. The temporal/budget semantics live in `mnemo.operations`, not here.

Backends are deliberately thin: they persist records and an append-only event
log. Records handed out by `get_records` must be treated as immutable values;
all changes go through `insert` / `update` / `delete_many`. Keeping the
semantics in one place is what lets the in-memory backend (used for fast RL
training) and the MongoDB backend (used by the service) behave identically.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..domain import MemoryRecord

# Cues that count as the user *asserting* a fact (used for corroboration statistics).
ASSERTION_CUES = frozenset({"state", "change"})


@runtime_checkable
class MemoryStore(Protocol):
    backend: str

    def reset_user(self, user_id: str) -> None:
        """Remove every record and event belonging to the user."""

    def get_records(self, user_id: str) -> list[MemoryRecord]:
        """All stored records (any status), ordered by creation sequence."""

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a new record; the store assigns `seq` and `id`. Returns the stored record."""

    def update(self, record: MemoryRecord) -> None:
        """Replace the stored record with the same id."""

    def delete_many(self, user_id: str, ids: list[str]) -> int:
        """Delete records by id; returns how many were removed."""

    def append_event(self, user_id: str, event: dict[str, Any]) -> None:
        """Append one observation/decision event to the audit log."""

    def claim_stats(self, user_id: str, attribute: str, value: str) -> tuple[int, int]:
        """(times the user asserted this attribute/value, how many of those were applied)."""

    def events(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Most recent events, oldest first."""

    def ping(self) -> bool:
        """True if the backend is reachable."""
