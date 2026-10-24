export interface MemRecord {
  id: string;
  attribute: string;
  value: string;
  kind: "permanent" | "temporary" | "planned";
  status: "current" | "historical";
  valid_from: number;
  valid_to: number | null;
  token_cost: number;
  confidence: number;
}

export interface Extraction {
  attribute: string;
  value: string | null;
  cue: string;
  confidence: number;
  duration_days: number | null;
  starts_in_days: number | null;
  source: string;
}

export type ActionName = "ADD" | "UPDATE" | "IGNORE" | "DELETE";
export interface Metrics { f1: number; retention: number; precision: number; stale_rate: number; tokens: number }

export interface SystemState {
  records: MemRecord[];
  tokens: number;
  action?: ActionName;
  requested?: ActionName;
  reason?: string;
  touched: Record<string, "inserted" | "superseded" | "deleted" | "expired" | "evicted">;
  metrics?: Metrics;
}

export interface Probe { question: string; kind: string; expected: string | string[] | null; answer: string | string[] | null; score: number }
export interface QAResult { system: string; accuracy: number; probes: Probe[] }
export interface ChatLine { turn: number; day: number; text: string; extraction?: Extraction | null }
export interface SessionInfo {
  session_id: string;
  scenario: { episode_id: number; profile: string; turns: number } | null;
  budget_tokens: number;
  systems: string[];
  extractor: string;
}
export interface Health { status: string; store_backend: string; store_reachable: boolean; llm: string | null; extractor: string; systems: string[] }
