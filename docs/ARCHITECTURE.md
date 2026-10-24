# Architecture

```
                 ┌──────────────────────── configs/ (catalog.yaml, default.yaml, cues.yaml) ────────────────────────┐
                 │                                                                                                  │
 user text ──► Extractor ──► Extraction(attr, value, cue, conf, dur) ──► Memory policy ──► Action ──► operations.py ──► MemoryStore
              (Groq | heuristic)                                    (store-all | rule | PPO)                           (MongoDB | in-memory)
                                                                                                      │
                                                    ┌─────────────────────────────────────────────────┤
                                                    ▼                                                 ▼
                                            Explainer (Groq, streamed)                       metrics.py / qa.py  ──► reward, evaluation
                                                    │
                                              SSE ──► React dashboard
```

## Modules

| Module | Responsibility |
|---|---|
| `domain.py` | `Action`, `Cue`, `Extraction`, `MemoryRecord` |
| `operations.py` | **single definition** of ADD/UPDATE/IGNORE/DELETE, budget, expiry, history; plans actions without side effects (validity mask) then applies them |
| `store/` | `MemoryStore` protocol; `InMemoryStore` (training, fallback) and `MongoStore` (service). Semantics are *not* in the backends, so both behave identically (parity-tested with `mongomock` and a live-server CI job) |
| `benchmark/` | procedural episodes with exact ground truth; extractor-noise model |
| `metrics.py`, `qa.py` | memory-state quality and the deterministic reader used for downstream accuracy |
| `engine.py` | stepwise episode engine shared by the Gym env, the evaluator and baselines |
| `rl/` | Gymnasium env, `MaskablePPO` training with validation checkpointing, policy wrapper |
| `evaluation.py`, `run_eval.py`, `report.py` | paired evaluation, bootstrap statistics, sweeps, ablations, report/plots |
| `agents/` | Groq client, LLM + offline extractor, explainer, QA agent (all degrade gracefully offline) |
| `service/` | session orchestration and FastAPI app (SSE streaming) |
| `frontend/` | React + TypeScript + Recharts dashboard |

## MongoDB data model

`memories` — one document per record
`{_id: "<user>:<seq>", user_id, seq, attribute, value, kind, status, valid_from, valid_to, created_day, updated_day, source_turn, confidence, token_cost}`

`events` — append-only audit log of every claim and decision
`{user_id, turn, day, attribute, value, cue, confidence, requested, effective, applied, reason, changes, system, ts}`

`counters` — atomic per-user sequence (`$inc`).

Indexes: `(user_id, seq)` unique; `(user_id, attribute, status)`; `(user_id, attribute, value, cue)` (corroboration
statistics used as policy features); `(user_id, turn)`.

Each live session writes one `user_id` per compared system (`<session>:<system>`), so the three managers stay isolated while
processing the same conversation. Sessions are reaped after `SESSION_TTL_S`, which also deletes their documents.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health` | store reachability, extractor/LLM status, loaded systems |
| POST | `/api/sessions` | `{scenario_seed?}`; scripted benchmark scenario or free-text |
| POST | `/api/sessions/{id}/turn` | `{text?, advance_days?}` → **SSE** frames: `turn`, `extraction`, `action`×systems, `explain_token`…, `scores`, `qa_result`, `done` |
| GET | `/api/sessions/{id}/memory` | current records per system |
| POST | `/api/sessions/{id}/ask` | free-form question answered from each system's memory |
| DELETE | `/api/sessions/{id}` | removes the session's documents |
| GET | `/api/results` | `results/results.json` |

Turns within a session are serialised by a per-session `asyncio.Lock`; the PPO forward pass and store I/O are short, so no
worker pool is needed at this scale.

## Failure behaviour

* No `GROQ_API_KEY`: heuristic extractor + template explanations + keyword QA reader. Fully functional offline.
* Groq error/timeout/malformed JSON: retried with backoff, then falls back per call; the turn never fails because of the LLM.
* MongoDB configured but unreachable: the service **fails fast at startup** with an explicit message (no silent downgrade).
* Missing PPO model: the `mnemo-ppo` column is disabled and the UI says so.
