"""In-process store. Used for RL training/evaluation (speed) and as a demo fallback."""

from __future__ import annotations

import threading
import time
from typing import Any

from ..domain import MemoryRecord
from .base import ASSERTION_CUES


class InMemoryStore:
    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, MemoryRecord]] = {}
        self._seq: dict[str, int] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._claims: dict[str, dict[tuple[str, str], list[int]]] = {}

    def reset_user(self, user_id: str) -> None:
        with self._lock:
            self._records.pop(user_id, None)
            self._seq.pop(user_id, None)
            self._events.pop(user_id, None)
            self._claims.pop(user_id, None)

    def get_records(self, user_id: str) -> list[MemoryRecord]:
        with self._lock:
            return list(self._records.get(user_id, {}).values())

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            seq = self._seq.get(record.user_id, 0) + 1
            self._seq[record.user_id] = seq
            stored = record.clone(id=f"{record.user_id}:{seq}", seq=seq)
            self._records.setdefault(record.user_id, {})[stored.id] = stored
            return stored

    def update(self, record: MemoryRecord) -> None:
        with self._lock:
            bucket = self._records.get(record.user_id, {})
            if record.id not in bucket:
                raise KeyError(f"unknown record id: {record.id}")
            bucket[record.id] = record

    def delete_many(self, user_id: str, ids: list[str]) -> int:
        with self._lock:
            bucket = self._records.get(user_id, {})
            removed = 0
            for rid in ids:
                if bucket.pop(rid, None) is not None:
                    removed += 1
            return removed

    def append_event(self, user_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            doc = {**event, "user_id": user_id, "ts": time.time()}
            self._events.setdefault(user_id, []).append(doc)
            attribute, value = doc.get("attribute"), doc.get("value")
            if attribute and value is not None and doc.get("cue") in ASSERTION_CUES:
                counters = self._claims.setdefault(user_id, {}).setdefault((attribute, value), [0, 0])
                counters[0] += 1
                if doc.get("applied"):
                    counters[1] += 1

    def claim_stats(self, user_id: str, attribute: str, value: str) -> tuple[int, int]:
        with self._lock:
            counters = self._claims.get(user_id, {}).get((attribute, value))
            return (counters[0], counters[1]) if counters else (0, 0)

    def events(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(user_id, []))[-limit:]

    def ping(self) -> bool:
        return True
