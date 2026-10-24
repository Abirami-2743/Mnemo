import type { Health, SessionInfo } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* body was not JSON */ }
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => fetch("/api/health").then(json<Health>);
export const getResults = () => fetch("/api/results").then(json<any>);
export const createSession = (scenario_seed: number | null) =>
  fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario_seed }) }).then(json<SessionInfo>);
export const deleteSession = (id: string) => fetch(`/api/sessions/${id}`, { method: "DELETE" });
export const ask = (id: string, question: string) =>
  fetch(`/api/sessions/${id}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) })
    .then(json<{ answers: Record<string, { answer: string; source: string }> }>);

export interface SSEEvent { event: string; data: any }

/** POST a turn and yield server-sent events as they arrive (EventSource cannot POST). */
export async function* streamTurn(id: string, body: { text?: string; advance_days?: number }, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  const res = await fetch(`/api/sessions/${id}/turn`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!res.ok || !res.body) await json(res);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = /^event: (.*)$/m.exec(block)?.[1];
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (ev && data) yield { event: ev, data: JSON.parse(data) };
    }
  }
}
