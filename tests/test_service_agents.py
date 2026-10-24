from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mnemo.agents.extractor import HeuristicExtractor, LLMExtractor
from mnemo.agents.llm import GroqClient, LLMError
from mnemo.config import Settings
from mnemo.domain import Cue
from mnemo.service.app import create_app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        store_backend="memory",
        model_path=tmp_path / "missing.zip",
        results_dir=tmp_path,
        frontend_dist=tmp_path / "nodist",
        _env_file=None,
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_heuristic_extractor_cues(catalog):
    h = HeuristicExtractor(catalog)
    cases = {
        "I moved to Bangalore permanently.": (Cue.CHANGE, "location", "Bangalore"),
        "I'm visiting Pune for 5 days.": (Cue.TEMPORARY, "location", "Pune"),
        "I'm travelling to Delhi in 9 days for 4 days.": (Cue.PLAN, "location", "Delhi"),
        "Please forget that I live in Chennai.": (Cue.RETRACT, "location", "Chennai"),
        "My cousin lives in Mumbai.": (Cue.THIRD_PARTY, "location", "Mumbai"),
        "If I lived in Berlin I'd cycle more.": (Cue.HYPOTHETICAL, "location", "Berlin"),
        "I work at Zoho.": (Cue.STATE, "employer", "Zoho"),
    }
    for text, (cue, attr, val) in cases.items():
        e = h.extract(text)
        assert (e.cue, e.attribute, e.value) == (cue, attr, val), text
    plan = h.extract("I'm travelling to Delhi in 9 days for 4 days.")
    assert (plan.starts_in_days, plan.duration_days) == (9, 4)
    assert h.extract("Thanks, that helps!") is None


def _llm(handler):
    return GroqClient("k", model="m", transport=httpx.MockTransport(handler), max_retries=1)


@pytest.mark.asyncio
async def test_llm_extractor_parses_and_falls_back(catalog):
    payload = {"attribute": "location", "value": "bangalore", "cue": "change", "confidence": 0.8}

    def ok(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"} and request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})

    heur = HeuristicExtractor(catalog)
    e = await LLMExtractor(_llm(ok), catalog, heur).extract("moved to blr")
    assert (e.attribute, e.value, e.cue, e.source) == ("location", "Bangalore", Cue.CHANGE, "llm")

    bad = LLMExtractor(_llm(lambda r: httpx.Response(500)), catalog, heur)
    e = await bad.extract("I work at Zoho.")
    assert e.source == "heuristic" and e.value == "Zoho"

    junk = LLMExtractor(
        _llm(
            lambda r: httpx.Response(
                200, json={"choices": [{"message": {"content": '{"attribute": "salary", "cue": "state"}'}}]}
            )
        ),
        catalog,
        heur,
    )
    assert (await junk.extract("I work at Stripe.")).value == "Stripe"


@pytest.mark.asyncio
async def test_llm_stream_and_errors():
    chunks = [{"choices": [{"delta": {"content": t}}]} for t in ("Hel", "lo")]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    client = _llm(lambda r: httpx.Response(200, text=body))
    assert "".join([t async for t in client.stream_text("s", "u")]) == "Hello"
    with pytest.raises(LLMError):
        await _llm(lambda r: httpx.Response(401)).complete_json("s", "u")


def test_health_and_session_lifecycle(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["store_backend"] == "memory" and "store-all" in h["systems"]
    sid = client.post("/api/sessions", json={"scenario_seed": 3}).json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/turn", json={})
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "turn" and "extraction" in names and "action" in names and names[-1] == "done"
    assert any(n == "scores" for n in names)
    mem = client.get(f"/api/sessions/{sid}/memory").json()
    assert set(mem["memory"]) == set(h["systems"])
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}/memory").status_code == 404


def test_free_text_turns_update_memory_and_answer_questions(client):
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    for text in ("I live in Chennai.", "I moved to Bangalore permanently."):
        ev = parse_sse(client.post(f"/api/sessions/{sid}/turn", json={"text": text, "advance_days": 2}).text)
        assert ev[-1][0] == "done"
    mem = client.get(f"/api/sessions/{sid}/memory").json()["memory"]
    rule = {(r["value"], r["status"]) for r in mem["rule-based"]["records"]}
    assert rule == {("Bangalore", "current"), ("Chennai", "historical")}
    assert len(mem["store-all"]["records"]) == 2
    ans = client.post(f"/api/sessions/{sid}/ask", json={"question": "What is the user's city of residence?"}).json()
    assert ans["answers"]["rule-based"]["answer"] == "Bangalore"


def test_validation_and_errors(client):
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/turn", json={}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/turn", json={"text": "x" * 600}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/turn", json={"text": "hi", "advance_days": -1}).status_code == 422
    assert client.post("/api/sessions/nope/turn", json={"text": "hi"}).status_code == 404
    assert client.get("/api/results").status_code == 404


def test_scenario_plays_to_the_end_with_qa(client):
    sid = client.post("/api/sessions", json={"scenario_seed": 5}).json()["session_id"]
    qa_events, finished = 0, False
    for _ in range(80):
        ev = parse_sse(client.post(f"/api/sessions/{sid}/turn", json={}).text)
        qa_events += sum(1 for n, _ in ev if n == "qa_result")
        if ev[-1][1].get("finished"):
            finished = True
            break
    assert finished and qa_events >= 2
