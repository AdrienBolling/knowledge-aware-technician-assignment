#!/usr/bin/env bash
# Lifecycle deep-dive on dgy GPU 2 (user request 2026-09-02): the four
# deep-dive agents (HTT-RL, PO-HTT-RL, Emp-Topsis, Emp-Spt) on the
# lifecycle scenario with the extended per-decision records
# (mttr_rolling_std, machine_rate_mean/std, fleet_knowledge_std,
# per-type disruption counts/times).  Runs INSIDE zellij; separate
# parts root (never merged into published trees).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
Q=reports/deepdive_lc_queue.log
say() { echo "$(date -u +%FT%TZ) [ddlc] $*" | tee -a "$Q"; }
mkdir -p reports

say "DEEPDIVE LC ARMED (pid $$, GPU 2, 4 agents @ lifecycle)"
for A in hc_v6 po_v6 empirical_topsis empirical_spt; do
  if [ -s "reports/deepdive_lc_parts/$A/lifecycle/episodes.csv" ]; then
    say "$A cached"; continue
  fi
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario lifecycle --record-every 200 \
    --agents "$A" --eval-seed 20260722 \
    --out-root "reports/deepdive_lc_parts/$A" \
    > "reports/deepdive_lc_${A}.log" 2>&1 &
done
wait
for A in hc_v6 po_v6 empirical_topsis empirical_spt; do
  [ -s "reports/deepdive_lc_parts/$A/lifecycle/episodes.csv" ] \
    && say "$A done" || say "$A MISSING (check reports/deepdive_lc_${A}.log)"
done
say "DEEPDIVE LC DONE"
