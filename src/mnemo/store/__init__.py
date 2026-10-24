"""Persistence backends for memory records."""

from __future__ import annotations

from ..config import Settings
from .base import ASSERTION_CUES, MemoryStore
from .memory import InMemoryStore
from .mongo import MongoStore

__all__ = ["ASSERTION_CUES", "InMemoryStore", "MemoryStore", "MongoStore", "create_store"]


def create_store(settings: Settings) -> MemoryStore:
    """Build the configured store; the MongoDB backend fails fast if it is unreachable."""
    if settings.backend == "mongo":
        assert settings.mongodb_uri  # guaranteed by Settings validation
        store = MongoStore.from_uri(settings.mongodb_uri, settings.mongodb_db, settings.mongodb_timeout_ms)
        if not store.ping():
            raise RuntimeError("MongoDB is not reachable. Check MONGODB_URI or set STORE_BACKEND=memory.")
        store.ensure_indexes()
        return store
    return InMemoryStore()
