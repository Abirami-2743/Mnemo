import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ask, createSession, deleteSession, streamTurn } from "./api";
import type { ActionName, ChatLine, MemRecord, QAResult, SessionInfo, SystemState } from "./types";

const COLORS: Record<string, string> = { "store-all": "#8a94a6", "rule-based": "#d98a12", "mnemo-ppo": "#2a6fdb" };
const fresh = (): SystemState => ({ records: [], tokens: 0, touched: {} });

function describe(r: MemRecord) {
  if (r.status === "historical") return "previous";
  if (r.kind === "temporary") return `until day ${r.valid_to}`;
  if (r.kind === "planned") return `day ${r.valid_from}–${r.valid_to}`;
  return "current";
}

function SystemCard({ name, s, budget }: { name: string; s: SystemState; budget: number }) {
  const fill = Math.min(100, (s.tokens / budget) * 100);
  return (
    <div className="card" style={{ borderTop: `3px solid ${COLORS[name] ?? "#999"}` }}>
      <div className="sys-head">
        <b>{name}</b>
        {s.action && <span className={`act ${s.action}`} title={s.reason}>{s.action}{s.requested && s.requested !== s.action ? ` (wanted ${s.requested})` : ""}</span>}
      </div>
      <div className={`meter ${fill > 95 ? "full" : fill > 75 ? "warn" : ""}`}><div style={{ width: `${fill}%` }} /></div>
      <div className="chips">
        <span className="chip">{s.tokens}/{budget} tokens</span>
        <span className="chip">{s.records.length} records</span>
        {s.metrics && <span className="chip">F1 {s.metrics.f1.toFixed(2)}</span>}
        {s.metrics && <span className="chip">stale {(s.metrics.stale_rate * 100).toFixed(0)}%</span>}
      </div>
      {s.records.length === 0 && <div className="empty">memory is empty</div>}
      {s.records.map((r) => (
        <div key={r.id} className={`rec ${r.status} ${s.touched[r.id] ?? ""}`}>
          <span className="k">{r.attribute.replace(/_/g, " ")}</span>
          <span className="tag">{describe(r)}</span>
          <span className="v">{r.value}</span>
          <span className="tag">{r.token_cost} tok</span>
        </div>
      ))}
    </div>
  );
}

export default function Demo({ systemsAvailable }: { systemsAvailable: string[] }) {
  const [seed, setSeed] = useState(7);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [systems, setSystems] = useState<Record<string, SystemState>>({});
  const [chat, setChat] = useState<ChatLine[]>([]);
  const [explain, setExplain] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [qa, setQa] = useState<QAResult[]>([]);
  const [series, setSeries] = useState<Record<string, number | string>[]>([]);
  const [text, setText] = useState("");
  const [question, setQuestion] = useState("What is the user's city of residence?");
  const [answers, setAnswers] = useState<Record<string, { answer: string; source: string }> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [auto, setAuto] = useState(false);
  const [finished, setFinished] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const sessionRef = useRef<SessionInfo | null>(null);

  const names = useMemo(() => session?.systems ?? systemsAvailable, [session, systemsAvailable]);
  const budget = session?.budget_tokens ?? 140;

  const reset = useCallback(async (scripted: boolean) => {
    abort.current?.abort();
    setAuto(false); setError(null); setChat([]); setExplain(""); setQa([]); setSeries([]); setAnswers(null); setFinished(false);
    if (sessionRef.current) deleteSession(sessionRef.current.session_id).catch(() => undefined);
    try {
      const s = await createSession(scripted ? seed : null);
      sessionRef.current = s;
      setSession(s);
      setSystems(Object.fromEntries(s.systems.map((n) => [n, fresh()])));
    } catch (e) { setError(String(e)); }
  }, [seed]);

  useEffect(() => { void reset(true); return () => { abort.current?.abort(); if (sessionRef.current) void deleteSession(sessionRef.current.session_id); }; /* eslint-disable-next-line */ }, []);
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [chat.length]);

  const play = useCallback(async (body: { text?: string; advance_days?: number }) => {
    const s = sessionRef.current;
    if (!s || busy) return false;
    setBusy(true); setError(null); setExplain(""); setStreaming(false);
    const ctrl = new AbortController(); abort.current = ctrl;
    try {
      for await (const { event, data } of streamTurn(s.session_id, body, ctrl.signal)) {
        if (event === "turn") setChat((c) => [...c, { turn: data.turn, day: data.day, text: data.text }]);
        else if (event === "extraction") setChat((c) => c.map((l, i) => (i === c.length - 1 ? { ...l, extraction: data.extraction } : l)));
        else if (event === "action") {
          const touched: SystemState["touched"] = {};
          const ch = data.changes;
          ch.inserted.forEach((r: { id: string }) => (touched[r.id] = "inserted"));
          ch.superseded.forEach((r: { id: string }) => (touched[r.id] = "superseded"));
          setSystems((p) => ({ ...p, [data.system]: { records: data.records, tokens: data.tokens, action: data.effective as ActionName, requested: data.requested as ActionName, reason: data.reason, touched, metrics: data.metrics } }));
        } else if (event === "explain_token") { setStreaming(true); setExplain((t) => t + data.text); }
        else if (event === "explain_end") setStreaming(false);
        else if (event === "scores") setSeries((p) => [...p, { turn: data.turn, ...Object.fromEntries(Object.entries(data.scores as Record<string, { f1: number }>).map(([k, v]) => [k, +v.f1.toFixed(3)])) }]);
        else if (event === "qa_result") setQa((p) => [...p.filter((x) => x.system !== data.system), data as QAResult]);
        else if (event === "error") throw new Error(data.detail);
        else if (event === "done") setFinished(Boolean(data.finished));
      }
      return true;
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String(e));
      return false;
    } finally { setBusy(false); setStreaming(false); }
  }, [busy]);

  useEffect(() => {
    if (!auto || busy || finished || !session?.scenario) { if (finished) setAuto(false); return; }
    const t = setTimeout(() => { void play({}).then((ok) => { if (!ok) setAuto(false); }); }, 350);
    return () => clearTimeout(t);
  }, [auto, busy, finished, session, play]);

  const lastQa = useMemo(() => {
    const byTurn = new Map<string, QAResult>();
    qa.forEach((q) => byTurn.set(q.system, q));
    return [...byTurn.values()];
  }, [qa]);

  return (
    <>
      {error && <div className="err">{error}</div>}
      <div className="controls">
        <label>Scenario seed <input type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, +e.target.value))} style={{ width: 80 }} /></label>
        <button className="btn ghost" onClick={() => void reset(true)}>New scenario</button>
        <button className="btn" disabled={busy || finished || !session?.scenario} onClick={() => void play({})}>Step ▸</button>
        <button className="btn ghost" disabled={finished || !session?.scenario} onClick={() => setAuto((a) => !a)}>{auto ? "Pause" : "Autoplay"}</button>
        <button className="btn ghost" onClick={() => void reset(false)}>Free-text mode</button>
        {session?.scenario && <span className="note">scenario #{session.scenario.episode_id} · {session.scenario.profile} · {chat.length}/{session.scenario.turns} turns · extractor: {session.extractor}</span>}
        {!session?.scenario && session && <span className="note">free-text mode · extractor: {session.extractor}</span>}
      </div>
      <form className="controls" onSubmit={(e) => { e.preventDefault(); if (text.trim()) { void play({ text: text.trim(), advance_days: 1 }); setText(""); } }}>
        <input className="grow" type="text" maxLength={500} value={text} placeholder='Say something about the user, e.g. "I moved to Pune permanently."' onChange={(e) => setText(e.target.value)} />
        <button className="btn" disabled={busy || !text.trim()}>Send</button>
      </form>

      <div className="grid g3">{names.map((n) => <SystemCard key={n} name={n} s={systems[n] ?? fresh()} budget={budget} />)}</div>

      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="grid">
          <div className="card">
            <h3>Decision narration <span className="note">post-hoc explanation · not used in training or scoring</span></h3>
            <div className={`explain ${streaming ? "cursor" : ""}`}>{explain || <span className="empty">Step through the conversation to see why MNEMO acts.</span>}</div>
          </div>
          <div className="card">
            <h3>Conversation</h3>
            <div className="chat">
              {chat.length === 0 && <div className="empty">no turns yet</div>}
              {chat.map((l) => (
                <div className="msg" key={l.turn}>
                  <small>turn {l.turn} · day {l.day}{l.extraction ? ` · ${l.extraction.cue} · ${l.extraction.attribute}=${l.extraction.value} · conf ${l.extraction.confidence.toFixed(2)}` : " · no claim"}</small>
                  {l.text}
                </div>
              ))}
              <div ref={chatEnd} />
            </div>
          </div>
        </div>
        <div className="grid">
          <div className="card">
            <h3>Memory-state F1 vs ground truth</h3>
            {series.length === 0 ? <div className="empty">scripted scenarios only</div> : (
              <ResponsiveContainer width="100%" height={210}>
                <LineChart data={series}><CartesianGrid strokeDasharray="3 3" stroke="var(--line)" /><XAxis dataKey="turn" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
                  {names.map((n) => <Line key={n} type="monotone" dataKey={n} stroke={COLORS[n]} dot={false} strokeWidth={2} isAnimationActive={false} />)}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <div className="card">
            <h3>Probe questions <span className="note">answered from stored memory only</span></h3>
            {lastQa.length === 0 ? <div className="empty">Checkpoints fire every few turns in a scripted scenario.</div> : (
              <table>
                <thead><tr><th>System</th><th>Accuracy</th></tr></thead>
                <tbody>{lastQa.map((q) => <tr key={q.system}><td>{q.system}</td><td className="num"><b>{(q.accuracy * 100).toFixed(0)}%</b> ({q.probes.filter((p) => p.score === 1).length}/{q.probes.length})</td></tr>)}</tbody>
              </table>
            )}
            {lastQa.find((q) => q.system === "mnemo-ppo") && (
              <table style={{ marginTop: 8 }}>
                <thead><tr><th>MNEMO question</th><th>Expected</th><th>Answer</th></tr></thead>
                <tbody>{lastQa.find((q) => q.system === "mnemo-ppo")!.probes.map((p, i) => (
                  <tr key={i}><td>{p.question}</td><td>{String(p.expected ?? "—")}</td><td className={p.score === 1 ? "good" : "badc"}>{String(p.answer ?? "—")}</td></tr>
                ))}</tbody>
              </table>
            )}
          </div>
          <div className="card">
            <h3>Ask anything <span className="note">LLM if configured, else keyword reader</span></h3>
            <form className="controls" onSubmit={(e) => { e.preventDefault(); const s = sessionRef.current; if (s) ask(s.session_id, question).then((r) => setAnswers(r.answers)).catch((x) => setError(String(x))); }}>
              <input className="grow" type="text" value={question} onChange={(e) => setQuestion(e.target.value)} />
              <button className="btn ghost">Ask</button>
            </form>
            {answers && <table><tbody>{Object.entries(answers).map(([k, v]) => <tr key={k}><td>{k}</td><td><b>{v.answer}</b> <span className="note">({v.source})</span></td></tr>)}</tbody></table>}
          </div>
        </div>
      </div>
    </>
  );
}
