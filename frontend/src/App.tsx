import { useEffect, useState } from "react";
import { getHealth } from "./api";
import Demo from "./Demo";
import Results from "./Results";
import type { Health } from "./types";

export default function App() {
  const [tab, setTab] = useState<"demo" | "results">("demo");
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => { getHealth().then(setHealth).catch(() => setDown(true)); }, []);
  return (
    <>
      <header className="top">
        <div className="logo"><i /> MNEMO</div>
        <nav className="tabs">
          <button className={`tab ${tab === "demo" ? "on" : ""}`} onClick={() => setTab("demo")}>Live demo</button>
          <button className={`tab ${tab === "results" ? "on" : ""}`} onClick={() => setTab("results")}>Results</button>
        </nav>
        <div className="status">
          {down && <span className="pill bad">API unreachable</span>}
          {health && <>
            <span className={`pill ${health.store_reachable ? "ok" : "bad"}`}>store: {health.store_backend}</span>
            <span className="pill">extractor: {health.extractor}</span>
            <span className="pill">{health.llm ? `LLM: ${health.llm}` : "LLM: off (offline mode)"}</span>
            {!health.systems.includes("mnemo-ppo") && <span className="pill bad">PPO model missing</span>}
          </>}
        </div>
      </header>
      <main>{tab === "demo" ? <Demo systemsAvailable={health?.systems ?? ["store-all", "rule-based", "mnemo-ppo"]} /> : <Results />}</main>
    </>
  );
}
