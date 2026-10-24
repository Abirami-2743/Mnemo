"""FastAPI application."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings
from .core import MemoryService, SessionLimit, SessionNotFound, sse

log = logging.getLogger("mnemo.api")


class CreateSession(BaseModel):
    scenario_seed: int | None = Field(None, ge=0, description="Start a scripted benchmark scenario")


class TurnRequest(BaseModel):
    text: str | None = Field(None, min_length=1, description="User message; omit to play the scripted turn")
    advance_days: int = Field(1, ge=0, le=365)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)


def create_app(settings: Settings | None = None, service: MemoryService | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = service or MemoryService(settings)
        log.info("MNEMO ready: %s", app.state.svc.health())
        yield
        await app.state.svc.aclose()

    app = FastAPI(
        title="MNEMO",
        version=__version__,
        lifespan=lifespan,
        description="Reinforcement-learned memory management for LLM agents.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"], allow_headers=["*"]
    )

    def svc(request: Request) -> MemoryService:
        return request.app.state.svc

    @app.exception_handler(SessionNotFound)
    async def _nf(_: Request, exc: SessionNotFound):
        return JSONResponse({"detail": f"unknown session {exc}"}, status_code=404)

    @app.exception_handler(SessionLimit)
    async def _limit(_: Request, exc: SessionLimit):
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        return svc(request).health()

    @app.post("/api/sessions", status_code=201)
    async def create_session(body: CreateSession, request: Request) -> dict[str, Any]:
        s = svc(request)
        return s.scenario_info(s.create_session(body.scenario_seed))

    @app.get("/api/sessions/{sid}/memory")
    async def memory(sid: str, request: Request) -> dict[str, Any]:
        s = svc(request)
        sess = s.get(sid)
        return {"day": sess.day, "turn": sess.turn, "memory": s.memory_view(sess)}

    @app.post("/api/sessions/{sid}/turn")
    async def turn(sid: str, body: TurnRequest, request: Request) -> StreamingResponse:
        s = svc(request)
        sess = s.get(sid)
        if body.text is not None and len(body.text) > settings.max_text_chars:
            raise HTTPException(422, f"text longer than {settings.max_text_chars} characters")
        if body.text is None and sess.episode is None:
            raise HTTPException(422, "no scripted scenario in this session; provide text")

        async def stream() -> AsyncIterator[str]:
            try:
                async for frame in s.run_turn(sess, body.text, body.advance_days):
                    yield frame
            except Exception as exc:
                log.exception("turn failed")
                yield sse("error", {"detail": str(exc)})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/sessions/{sid}/ask")
    async def ask(sid: str, body: AskRequest, request: Request) -> dict[str, Any]:
        s = svc(request)
        return await s.ask(s.get(sid), body.question)

    @app.delete("/api/sessions/{sid}", status_code=204)
    async def delete(sid: str, request: Request) -> None:
        s = svc(request)
        s.get(sid)
        s.delete_session(sid)

    @app.get("/api/results")
    async def results() -> Any:
        path = settings.results_dir / "results.json"
        if not path.exists():
            raise HTTPException(404, "results.json not found; run `mnemo evaluate`")
        return json.loads(path.read_text())

    dist = settings.frontend_dist
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str) -> FileResponse:
            target = (dist / full_path).resolve()
            if full_path and dist.resolve() in target.parents and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


def app_factory() -> FastAPI:
    return create_app()
