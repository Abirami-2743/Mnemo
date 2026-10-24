# Methodology

This document states exactly what MNEMO measures, so every number in `results/RESULTS.md` can be audited.

## 1. Problem formulation

An LLM agent receives a stream of user utterances. A (noisy) **extractor** turns each utterance into at most
one structured claim `(attribute, value, cue, confidence, duration, starts_in)`. A **memory policy** observes the
claim and the current memory and chooses one action:

| Action | Effect (defined once, in `mnemo/operations.py`) |
|---|---|
| `ADD` | append a new record (kind decided by the cue: permanent / temporary / planned) |
| `UPDATE` | supersede the live record of the same attribute and kind; a superseded permanent value is kept as bounded *history*, a temporary/planned one is replaced |
| `IGNORE` | do nothing |
| `DELETE` | remove the live record(s) matching the claim |

Temporary and planned records expire automatically at `valid_to`. The memory has a **hard token budget**; history is
evicted first, and an action that cannot fit is *infeasible*. Infeasible actions are masked out of the policy's action
distribution (`MaskablePPO`) and, if still requested, rejected and executed as `IGNORE`.

All systems — store-all, rule-based, untrained, PPO, oracle — are bound by the same operations module, the same budget,
and (in evaluation) the same pre-computed extractor noise per episode, so comparisons are paired.

## 2. Benchmark (procedural, exact ground truth)

`mnemo/benchmark/generator.py` simulates a *latent user* and emits utterances about them. Nine turn kinds:
new fact, repeated fact, permanent change, temporary stay, future plan, retraction, hypothetical, third-party claim,
chit-chat. Phrasing, values, attribute importance and cardinality come from `configs/catalog.yaml`; turn-kind mixtures
("profiles": balanced, mover, noisy, sparse, adversarial), turn counts, durations and split seed ranges come from
`configs/default.yaml`. After each turn the generator stores the ground-truth memory an ideal agent would hold
(`GTSnapshot`), and at checkpoints it generates probe questions (current value, previous value, temporary value,
planned value, set of values) with exact expected answers.

Splits are disjoint seed ranges: **train** (RL), **val** (checkpoint selection, rule-baseline tuning), **test**
(headline numbers), **ood** (adversarial/noisy profile mixture never used in training).

Hypothetical and third-party turns ("If I lived in Pune…", "My cousin lives in Pune") sometimes reuse the user's *real*
value (`confuser_value_prob`), so "the value appears in the text" is not a usable shortcut.

### Extractor noise
Real extractors misread. `mnemo/benchmark/noise.py` perturbs the true claim with: missed facts, spurious facts from
chit-chat, cue confusions (e.g. temporary read as permanent), wrong values, and a confidence score drawn from
overlapping Beta distributions (informative but imperfect). Rates are in `configs/default.yaml → noise`. Training
randomises the noise multiplier in `[0.5, 1.5]`; evaluation sweeps `0…2` to test robustness beyond the training range.

## 3. Memory-state quality (the central measurement)

A stored record is matched to ground truth by `(attribute, value, kind, status)`.

* **matched** – key present in ground truth (first copy).
* **duplicate** – further copies of a matched key.
* **stale** – a *live* record that is wrong now although the value was true at some point (superseded, retracted, or
  stored with the wrong kind, e.g. a temporary stay stored as home).
* **junk** – value never true of the user (hypothetical, third party, spurious).
* Correct superseded history that the ground truth does not list is **neutral** (it costs tokens only). History is never
  counted as stale.

Reported metrics (time-averaged over turns once ground truth is non-empty): importance-weighted **retention**
(recall), **precision**, **state F1**, **stale rate**, **junk rate**, **redundancy rate**, **contradiction rate**
(single-valued attributes holding >1 live permanent value), **tokens**, **harmful deletions** (removed records that the
ground truth still lists), plus **QA accuracy**.

**Downstream QA** uses a deterministic reader that sees only the stored records ("latest evidence wins"), so the score
reflects memory quality, not reader intelligence. Probes are scored exact-match (sets: Jaccard).

## 4. Reward (`mnemo/engine.py`)

For a decision at turn *t* with ground-truth snapshot *G_t*:

```
r_state  = ( V(M_after, G_t) - V(M_before, G_t) ) / max_importance
V(M, G)  = matched_w - c_stale*stale_w - c_junk*junk_w - c_dup*dup_w
r_tokens = - token_cost * (tokens_after - tokens_before) / budget
r_qa     = qa_weight * probe_accuracy          (only at checkpoints)
r        = r_state + r_tokens + r_qa
```

Both `V` terms are evaluated against the *same* snapshot, so the reward isolates the effect of the action (exogenous
expiry cannot be credited to the policy). Weights are importance × kind weight (permanent 1, temporary/planned
`temporary_weight`, history `history_weight`). Because `V` penalises deleting matched records, a "delete everything"
policy is not rewarded; `harmful_deletions` is reported separately as a check. All constants live in
`configs/default.yaml → reward`.

## 5. Policy and training

* Observation: 31 features (`mnemo/features.py`): cue one-hot, extractor confidence, attribute importance, whether a
  same-kind target / same value / history exists, budget fill, headroom after ADD, corroboration statistics from the
  event log (`prior_seen`, `prior_applied_frac` — read from the MongoDB `events` collection in the service), and the
  action-validity mask.
* Algorithm: PPO via `sb3-contrib` `MaskablePPO`, MLP 128×128, linear learning-rate decay, 8 parallel envs.
* Checkpoint selection on the validation split (never the test split).
* Training randomises the token budget (`ppo.train_budgets`) and noise multiplier per episode.
* Three independent seeds; headline numbers average the seeds per episode, and each seed is also reported.

This is PPO over a *structured-claim* policy, not PPO fine-tuning of a language model. The language model (Groq) is used
only at inference time as the extractor, explainer and QA agent; it never participates in training or in the reported
metrics.

## 6. Baselines

| System | Description |
|---|---|
| store-all | ADD every claim; FIFO eviction when the budget is full |
| rule-based | cue/confidence rules; **confidence thresholds are tuned by coordinate descent on the validation split** (a deliberately strong baseline) |
| untrained policy | identical architecture and initialisation scheme as PPO, zero training, stochastic actions |
| random | uniform over valid actions (sanity floor) |
| oracle | one-step greedy using hidden ground truth; an upper reference, not an achievable system |

## 7. Statistics

Per-episode metrics → mean with 95 % percentile bootstrap CI (2 000 resamples); system differences are **paired by
episode** (identical episodes and noise draws), with bootstrap CI and a sign-flip permutation p-value.

## 8. Limitations (read before citing)

* The benchmark is synthetic; templates come from a catalog. Absolute numbers do not transfer to real conversations.
  The claim supported is about *relative* behaviour under controlled temporal phenomena.
* The offline heuristic extractor reads the same templates the generator writes, so its 100 % accuracy on benchmark text
  is in-distribution and says nothing about free text; use the Groq extractor for real input.
* Token counts use a ~4 characters/token estimate on the exact prompt line, applied identically to all systems.
* PPO's reward uses ground-truth state, which a deployed system would not have; MNEMO is trained in simulation.
* No prompted-LLM memory baseline is included in the reported results (it needs network access to an LLM API).
* Related work (Memory-R1, STALE, Mem-α, MEM1) already studies learned memory management and stale memories. The
  contribution claimed here is the *combination* — temporal versioning, direct memory-state evaluation, a hard budget and
  noisy-extractor robustness — not the idea of RL for memory.
