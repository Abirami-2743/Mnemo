# MNEMO: Reinforcement Learning for Temporally Consistent, Budget-Constrained Memory Management in LLM Agents

**Abstract.** Persistent memory lets LLM agents personalise across sessions, but naive accumulation fills the store with
outdated, redundant and contradictory records that waste context tokens and cause stale-personalisation errors. MNEMO
formulates memory maintenance as a sequential decision problem: after each conversational turn a policy chooses ADD, UPDATE,
IGNORE or DELETE over a MongoDB-backed store of temporally versioned facts (permanent, temporary and planned records with
validity windows; superseded values are kept as bounded history rather than overwritten) under a hard token budget.
A noisy extractor converts utterances into structured claims with a temporal/epistemic cue and a confidence; a policy
trained with Proximal Policy Optimisation (masked-action PPO, three seeds) decides what to do with each claim. The reward is
computed from the ground-truth memory state and penalises stale, junk and duplicate records, rewards retention, charges for
tokens and adds a downstream question-answering bonus.

We evaluate on procedurally generated user timelines (nine turn types: new fact, repeat, permanent change, temporary stay,
future plan, retraction, hypothetical, third-party claim, chit-chat) with exact ground truth, a held-out test split and a
distribution-shifted split, comparing against store-all, a validation-tuned rule-based manager, an untrained policy and a
one-step oracle. Memory-state quality is measured directly (state F1, retention, precision, stale/junk/redundancy/contradiction
rates, harmful deletions, tokens) alongside downstream QA accuracy from a reader that sees only stored memory.

On 300 held-out episodes MNEMO-PPO reaches state F1 0.836 and QA accuracy 0.828, against 0.454 and 0.430 for store-all, while
using 48.6 instead of 100.0 memory tokens (-51%) and cutting the stale-record rate from 28.0% to 7.7%. Against a rule-based
manager whose confidence thresholds were tuned on validation data, MNEMO-PPO is statistically indistinguishable on state F1
(-0.001, 95% CI [-0.007, +0.005]) and QA accuracy (-0.004, [-0.010, +0.002]), but holds fewer junk records (5.6% vs 7.4%),
makes 24% fewer harmful deletions (0.48 vs 0.63 per episode) and uses 4.4% fewer tokens (all paired 95% CIs exclude zero). The
learned policy reaches roughly 90% of a ground-truth oracle's state F1. Single-seed, shorter-training ablations suggest the extractor-confidence and
corroboration features are the main drivers of the learned policy's quality. We report the parity with the tuned heuristic
as it is: the contribution is a reproducible, auditable benchmark and system, not a claim that RL dominates well-tuned rules.
