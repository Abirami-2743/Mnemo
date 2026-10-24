import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getResults } from "./api";

const SYSTEMS = ["store-all", "rule-based", "untrained-policy", "mnemo-ppo", "oracle"];
const COLORS: Record<string, string> = { "store-all": "#8a94a6", "rule-based": "#d98a12", "untrained-policy": "#b08ad9", "mnemo-ppo": "#2a6fdb", oracle: "#1f9d62" };
const COLS: [string, string, number][] = [["state_f1", "State F1 ↑", 3], ["retention", "Retention ↑", 3], ["precision", "Precision ↑", 3], ["stale_rate", "Stale ↓", 3], ["junk_rate", "Junk ↓", 3], ["avg_tokens", "Tokens ↓", 1], ["qa_accuracy", "QA acc ↑", 3]];

type Stat = { mean: number; lo: number; hi: number };
function Table({ data }: { data: Record<string, Record<string, Stat>> }) {
  return (
    <table>
      <thead><tr><th>System</th>{COLS.map((c) => <th key={c[0]}>{c[1]}</th>)}</tr></thead>
      <tbody>
        {SYSTEMS.filter((s) => data[s]).map((s) => (
          <tr key={s} style={s === "mnemo-ppo" ? { fontWeight: 700 } : undefined}>
            <td>{s}</td>
            {COLS.map(([k, , nd]) => <td className="num" key={k}>{data[s][k].mean.toFixed(nd)} <span className="note">±{((data[s][k].hi - data[s][k].lo) / 2).toFixed(nd)}</span></td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Results() {
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getResults().then(setRes).catch((e) => setErr(String(e))); }, []);
  if (err) return <div className="err">{err} — run <code>mnemo evaluate</code> to generate results.</div>;
  if (!res) return <div className="empty">loading…</div>;

  const bars = ["state_f1", "qa_accuracy"].map((k) => ({ metric: k === "state_f1" ? "State F1" : "QA accuracy", ...Object.fromEntries(SYSTEMS.filter((s) => res.test[s]).map((s) => [s, +res.test[s][k].mean.toFixed(3)])) }));
  const sweep = (axis: string, key: string) => Object.entries(res.sweeps[axis]).map(([v, row]: [string, any]) => ({ x: v, ...Object.fromEntries(Object.entries(row).map(([s, m]: [string, any]) => [s, +m[key].toFixed(3)])) }));
  const diff = res.paired_diffs_test["mnemo-ppo vs rule-based"];
  return (
    <div className="grid">
      <div className="card">
        <h3>Held-out test split <span className="note">{res.meta.n_test} episodes · mean ± 95% CI half-width · config {res.meta.config_fingerprint}</span></h3>
        <Table data={res.test} />
      </div>
      <div className="card">
        <h3>MNEMO-PPO − tuned rule-based (paired, 95% CI)</h3>
        <table><thead><tr><th>Metric</th><th>Difference</th><th>95% CI</th><th>Significant</th></tr></thead>
          <tbody>{Object.entries(diff).map(([k, d]: [string, any]) => (
            <tr key={k}><td>{k}</td><td className="num">{d.mean.toFixed(4)}</td><td className="num">[{d.lo.toFixed(4)}, {d.hi.toFixed(4)}]</td><td>{d.lo > 0 || d.hi < 0 ? "yes" : "no"}</td></tr>))}</tbody></table>
      </div>
      <div className="grid g2">
        <div className="card"><h3>Headline</h3>
          <ResponsiveContainer width="100%" height={240}><BarChart data={bars}><CartesianGrid stroke="var(--line)" strokeDasharray="3 3" /><XAxis dataKey="metric" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
            {SYSTEMS.filter((s) => res.test[s]).map((s) => <Bar key={s} dataKey={s} fill={COLORS[s]} />)}</BarChart></ResponsiveContainer></div>
        <div className="card"><h3>OOD split (unseen profile mix)</h3><Table data={res.ood} /></div>
      </div>
      <div className="grid g2">
        {[["noise", "State F1 vs extractor-noise multiplier"], ["budget", "State F1 vs token budget"]].map(([axis, title]) => (
          <div className="card" key={axis}><h3>{title}</h3>
            <ResponsiveContainer width="100%" height={240}><LineChart data={sweep(axis, "state_f1")}><CartesianGrid stroke="var(--line)" strokeDasharray="3 3" /><XAxis dataKey="x" type="category" allowDuplicatedCategory={false} /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
              {["store-all", "rule-based", "mnemo-ppo", "oracle"].map((s) => <Line key={s} dataKey={s} stroke={COLORS[s]} strokeWidth={2} />)}</LineChart></ResponsiveContainer></div>
        ))}
      </div>
      <p className="note">Every number on this page is read live from <code>results/results.json</code>, produced by <code>mnemo evaluate</code>.</p>
    </div>
  );
}
