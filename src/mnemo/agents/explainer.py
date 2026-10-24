"""Post-hoc narration of a memory decision (never used as a training or evaluation signal)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from ..domain import Extraction
from ..operations import Outcome
from .llm import GroqClient, LLMError

log = logging.getLogger(__name__)

_SYSTEM = (
    "You narrate decisions of a memory manager for an AI assistant in one or two short, plain "
    "sentences. Mention the user's attribute and value and why the action fits. Do not invent facts."
)

_REASONS = {
    "ok": "{action} was applied.",
    "ignored": "nothing needed to change.",
    "over_budget": "the token budget had no room, so nothing was stored.",
    "no_target": "there was no matching record to {requested_lower}.",
    "no_change": "the stored value already matches.",
    "update_not_applicable": "this attribute holds several values, so a new value is added instead.",
    "retraction_cannot_be_added": "a retraction removes memory rather than adding it.",
    "retraction_cannot_update": "a retraction removes memory rather than updating it.",
    "no_extraction": "no claim about the user was found.",
}


def template_explanation(extraction: Extraction | None, outcome: Outcome) -> str:
    if extraction is None:
        return "No claim about the user was found, so memory is unchanged."
    subject = (
        f"{extraction.attribute.replace('_', ' ')} = {extraction.value}" if extraction.value else extraction.attribute
    )
    head = (
        f"Heard a {extraction.cue.value.replace('_', ' ')} claim ({subject}, confidence {extraction.confidence:.2f}). "
    )
    tail = _REASONS.get(outcome.reason, "{action} was applied.").format(
        action=outcome.effective.name, requested_lower=outcome.requested.name.lower()
    )
    if outcome.rejected:
        return head + f"{outcome.requested.name} was not possible: {tail}"
    return head + f"{outcome.effective.name}: " + tail


async def stream_explanation(
    client: GroqClient | None, extraction: Extraction | None, outcome: Outcome, system_name: str
) -> AsyncIterator[str]:
    fallback = template_explanation(extraction, outcome)
    if client is None or extraction is None:
        yield fallback
        return
    prompt = (
        f"System: {system_name}\nClaim: {extraction.model_dump(mode='json')}\n"
        f"Requested action: {outcome.requested.name}; effective action: {outcome.effective.name} "
        f"(reason: {outcome.reason}).\nChanges: {outcome.changes()}"
    )
    emitted = False
    try:
        async for tok in client.stream_text(_SYSTEM, prompt):
            emitted = True
            yield tok
    except LLMError as exc:
        log.warning("explainer stream failed: %s", exc)
        if not emitted:
            yield fallback
