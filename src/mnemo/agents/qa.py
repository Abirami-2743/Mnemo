"""QA agents: answer questions about the user from stored memory only."""

from __future__ import annotations

import logging

from ..benchmark.generator import Probe
from ..catalog import Catalog
from ..domain import MemoryRecord
from ..qa import Answer, answer_probe
from ..tokens import render_memory_line
from .llm import GroqClient, LLMError

log = logging.getLogger(__name__)

_SYSTEM = (
    "Answer the question about the user using ONLY the memory lines provided. Lines tagged "
    "[current] are permanent facts, [temporary ...] and [planned ...] hold only in their window, "
    "[previous] is superseded history. If two lines conflict, prefer the most recent one. "
    'Return JSON {"answer": <string or list of strings or null>}; null if memory does not say.'
)


def memory_prompt(records: list[MemoryRecord]) -> str:
    return (
        "\n".join(
            render_memory_line(r.attribute, r.value, r.kind, r.status, r.valid_from, r.valid_to)
            for r in sorted(records, key=lambda r: r.seq)
        )
        or "(memory is empty)"
    )


class QAAgent:
    def __init__(self, catalog: Catalog, client: GroqClient | None = None) -> None:
        self.catalog = catalog
        self.client = client

    async def answer_free(self, question: str, records: list[MemoryRecord], day: int) -> tuple[str, str]:
        """Returns (answer text, source) for a free-form question."""
        live = [r for r in records if r.live(day) or r.status.value == "historical"]
        if self.client is not None:
            try:
                out = await self.client.complete_json(
                    _SYSTEM, f"Memory:\n{memory_prompt(live)}\n\nQuestion: {question}"
                )
                ans = out.get("answer")
                return (", ".join(ans) if isinstance(ans, list) else str(ans) if ans else "unknown"), "llm"
            except LLMError as exc:
                log.warning("QA LLM failed: %s", exc)
        return self._keyword_answer(question, live, day), "reader"

    def _keyword_answer(self, question: str, records: list[MemoryRecord], day: int) -> str:
        q = question.lower()
        best, score = None, 0
        for name, spec in self.catalog.attributes.items():
            s = sum(1 for k in (*spec.keywords, spec.label.lower(), name.replace("_", " ")) if k in q)
            if s > score:
                best, score = name, s
        if best is None:
            return "unknown"
        spec = self.catalog.spec(best)
        kind = "set" if spec.multi else "current"
        if "previous" in q or "before" in q or "used to" in q:
            kind = "previous"
        elif "temporar" in q:
            kind = "temporary"
        elif "upcoming" in q or "plan" in q or "next" in q:
            kind = "plan"
        ans: Answer = answer_probe(records, Probe(kind, best, None, question), day)
        if isinstance(ans, frozenset):
            return ", ".join(sorted(ans)) or "unknown"
        return ans or "unknown"
