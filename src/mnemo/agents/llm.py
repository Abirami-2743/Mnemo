"""Minimal async client for Groq's OpenAI-compatible chat API (streaming + JSON mode)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class GroqClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 20.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        delay = 0.5
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        raise LLMError(f"Groq request failed: {last!r}")

    async def complete_json(self, system: str, user: str, *, max_tokens: int = 300) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = await self._post(payload)
        try:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"unparseable LLM response: {exc!r}") from exc

    async def stream_text(self, system: str, user: str, *, max_tokens: int = 120) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, ValueError):
                        continue
                    if delta:
                        yield delta
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise LLMError(f"Groq stream failed: {exc!r}") from exc
