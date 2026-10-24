"""Core domain types shared by the store, the RL environment, the agents and the API."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Action(IntEnum):
    """The four memory-management actions the policy chooses between."""

    ADD = 0
    UPDATE = 1
    IGNORE = 2
    DELETE = 3


ACTION_NAMES: tuple[str, ...] = tuple(a.name for a in Action)
N_ACTIONS = len(Action)


class Cue(str, Enum):
    """Temporal / epistemic cue attached to an extracted claim."""

    STATE = "state"
    CHANGE = "change"
    TEMPORARY = "temporary"
    PLAN = "plan"
    RETRACT = "retract"
    HYPOTHETICAL = "hypothetical"
    THIRD_PARTY = "third_party"


CUES: tuple[Cue, ...] = tuple(Cue)


class TurnKind(str, Enum):
    """Ground-truth kind of a benchmark turn (never visible to the policy)."""

    STATE_NEW = "state_new"
    STATE_REPEAT = "state_repeat"
    CHANGE = "change"
    TEMPORARY = "temporary"
    PLAN = "plan"
    RETRACT = "retract"
    HYPOTHETICAL = "hypothetical"
    THIRD_PARTY = "third_party"
    CHITCHAT = "chitchat"


KIND_TO_CUE: dict[TurnKind, Cue | None] = {
    TurnKind.STATE_NEW: Cue.STATE,
    TurnKind.STATE_REPEAT: Cue.STATE,
    TurnKind.CHANGE: Cue.CHANGE,
    TurnKind.TEMPORARY: Cue.TEMPORARY,
    TurnKind.PLAN: Cue.PLAN,
    TurnKind.RETRACT: Cue.RETRACT,
    TurnKind.HYPOTHETICAL: Cue.HYPOTHETICAL,
    TurnKind.THIRD_PARTY: Cue.THIRD_PARTY,
    TurnKind.CHITCHAT: None,
}


class RecordKind(str, Enum):
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    PLANNED = "planned"


class RecordStatus(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"


# The kind of record a claim creates is decided only by the cue the extractor reported.
CUE_TO_RECORD_KIND: dict[Cue, RecordKind] = {
    Cue.STATE: RecordKind.PERMANENT,
    Cue.CHANGE: RecordKind.PERMANENT,
    Cue.TEMPORARY: RecordKind.TEMPORARY,
    Cue.PLAN: RecordKind.PLANNED,
    Cue.RETRACT: RecordKind.PERMANENT,
    Cue.HYPOTHETICAL: RecordKind.PERMANENT,
    Cue.THIRD_PARTY: RecordKind.PERMANENT,
}


class Extraction(BaseModel):
    """A structured claim extracted from one user utterance."""

    model_config = ConfigDict(frozen=True)

    attribute: str
    value: str | None = None
    cue: Cue
    confidence: float = Field(ge=0.0, le=1.0)
    duration_days: int | None = Field(default=None, ge=1, le=3650)
    starts_in_days: int | None = Field(default=None, ge=0, le=3650)
    source: str = "sim"

    @model_validator(mode="after")
    def _value_required(self) -> Extraction:
        if self.value is None and self.cue is not Cue.RETRACT:
            raise ValueError("value is required unless the cue is 'retract'")
        return self


@dataclass(slots=True)
class MemoryRecord:
    """One persisted memory. `valid_to` is exclusive; None means open-ended."""

    id: str
    user_id: str
    seq: int
    attribute: str
    value: str
    kind: RecordKind
    status: RecordStatus
    valid_from: int
    valid_to: int | None
    created_day: int
    updated_day: int
    source_turn: int
    confidence: float
    token_cost: int

    def expired(self, day: int) -> bool:
        return self.status is RecordStatus.CURRENT and self.valid_to is not None and self.valid_to <= day

    def live(self, day: int) -> bool:
        return self.status is RecordStatus.CURRENT and not self.expired(day)

    def clone(self, **changes: Any) -> MemoryRecord:
        return dataclasses.replace(self, **changes)

    def to_doc(self) -> dict[str, Any]:
        doc = dataclasses.asdict(self)
        doc["kind"] = self.kind.value
        doc["status"] = self.status.value
        return doc

    @classmethod
    def from_doc(cls, doc: dict[str, Any]) -> MemoryRecord:
        data = {f.name: doc[f.name] for f in dataclasses.fields(cls)}
        data["kind"] = RecordKind(data["kind"])
        data["status"] = RecordStatus(data["status"])
        return cls(**data)

    def public(self) -> dict[str, Any]:
        """JSON-safe view for API responses."""
        return {
            "id": self.id,
            "attribute": self.attribute,
            "value": self.value,
            "kind": self.kind.value,
            "status": self.status.value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "created_day": self.created_day,
            "source_turn": self.source_turn,
            "confidence": round(self.confidence, 3),
            "token_cost": self.token_cost,
        }
