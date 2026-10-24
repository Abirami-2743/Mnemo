#!/usr/bin/env bash
# Train the headline PPO seeds and the ablation runs (2 processes at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
run() { "$PY" -m mnemo.rl.train "$@" ; }
mkdir -p logs models
run --seed 0 --out models/ppo_seed0 > logs/train_seed0.log 2>&1 &
run --seed 1 --out models/ppo_seed1 > logs/train_seed1.log 2>&1 &
wait
run --seed 2 --out models/ppo_seed2 > logs/train_seed2.log 2>&1 &
run --seed 0 --timesteps 400000 --ablate support --out models/abl_no_support > logs/abl_no_support.log 2>&1 &
wait
run --seed 0 --timesteps 400000 --ablate confidence --out models/abl_no_confidence > logs/abl_no_confidence.log 2>&1 &
run --seed 0 --timesteps 400000 --token-cost 0 --out models/abl_no_token_cost > logs/abl_no_token_cost.log 2>&1 &
wait
echo ALL_TRAINING_DONE > logs/train_done.flag
