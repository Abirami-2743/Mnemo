"""MongoDB-backed store (pymongo).

Collections
-----------
memories  one document per memory record; `_id` == `id` == "<user_id>:<seq>".
events    append-only log of every observed claim and the decision taken for it.
counters  per-user monotonically increasing sequence (atomic `$inc`).

The budgeted design bounds the number of documents per user, so every read is a
single indexed `find({user_id})` and the temporal filtering is done in one shared
Python code path (`mnemo.operations`) instead of being re-expressed as queries.
Multi-document writes are not wrapped in a transaction: a step is applied by one
session worker at a time (see the per-session lock in the service layer).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, MongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from ..domain import MemoryRecord
from .base import ASSERTION_CUES

log = logging.getLogger(__name__)


class MongoStore:
    backend = "mongo"

    def __init__(
        self,
        client: Any,
        db_name: str = "mnemo",
        *,
        memories: str = "memories",
        events: str = "events",
        counters: str = "counters",
    ) -> None:
        self._client = client
        self._db = client[db_name]
        self._memories = self._db[memories]
        self._events = self._db[events]
        self._counters = self._db[counters]

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "mnemo", timeout_ms: int = 3000) -> MongoStore:
        client: MongoClient[Any] = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, tz_aware=True)
        return cls(client, db_name)

    def ensure_indexes(self) -> None:
        self._memories.create_index([("user_id", ASCENDING), ("seq", ASCENDING)], unique=True, name="user_seq")
        self._memories.create_index(
            [("user_id", ASCENDING), ("attribute", ASCENDING), ("status", ASCENDING)],
            name="user_attr_status",
        )
        self._events.create_index(
            [
                ("user_id", ASCENDING),
                ("attribute", ASCENDING),
                ("value", ASCENDING),
                ("cue", ASCENDING),
            ],
            name="user_claim",
        )
        self._events.create_index([("user_id", ASCENDING), ("turn", ASCENDING)], name="user_turn")

    def reset_user(self, user_id: str) -> None:
        self._memories.delete_many({"user_id": user_id})
        self._events.delete_many({"user_id": user_id})
        self._counters.delete_many({"_id": user_id})

    def get_records(self, user_id: str) -> list[MemoryRecord]:
        cursor = self._memories.find({"user_id": user_id}).sort("seq", ASCENDING)
        return [MemoryRecord.from_doc(doc) for doc in cursor]

    def _next_seq(self, user_id: str) -> int:
        doc = self._counters.find_one_and_update(
            {"_id": user_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    def insert(self, record: MemoryRecord) -> MemoryRecord:
        seq = self._next_seq(record.user_id)
        stored = record.clone(id=f"{record.user_id}:{seq}", seq=seq)
        doc = stored.to_doc()
        doc["_id"] = stored.id
        self._memories.insert_one(doc)
        return stored

    def update(self, record: MemoryRecord) -> None:
        doc = record.to_doc()
        doc["_id"] = record.id
        result = self._memories.replace_one({"_id": record.id}, doc)
        if result.matched_count != 1:
            raise KeyError(f"unknown record id: {record.id}")

    def delete_many(self, user_id: str, ids: list[str]) -> int:
        if not ids:
            return 0
        result = self._memories.delete_many({"user_id": user_id, "_id": {"$in": ids}})
        return int(result.deleted_count)

    def append_event(self, user_id: str, event: dict[str, Any]) -> None:
        self._events.insert_one({**event, "user_id": user_id, "ts": datetime.now(UTC)})

    def claim_stats(self, user_id: str, attribute: str, value: str) -> tuple[int, int]:
        base = {
            "user_id": user_id,
            "attribute": attribute,
            "value": value,
            "cue": {"$in": sorted(ASSERTION_CUES)},
        }
        seen = self._events.count_documents(base)
        applied = self._events.count_documents({**base, "applied": True})
        return int(seen), int(applied)

    def events(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        cursor = self._events.find({"user_id": user_id}, {"_id": 0}).sort("turn", -1).limit(limit)
        return list(reversed(list(cursor)))

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
        except PyMongoError:
            log.warning("MongoDB ping failed", exc_info=True)
            return False
        return True

    def close(self) -> None:
        self._client.close()
