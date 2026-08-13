#!/usr/bin/env bash
# Lifecycle-scenario benchmark (v6w corrected world) on dgy — run inside
# a zellij session.
#
# Companion to dgy_v6w_benchmarks.sh: same seed, same parts tree
# (reports/hvp_v6w_parts/<agent>/lifecycle), merged into
# reports/hvp_eval_v6w/lifecycle.  The scenario profile (n_eps=1,
# sim=5M, steps=1.5M) lives in eval_human_vs_performance.py; the
# 10-event fleet/park mutation schedule in
# run_configs/benchmark_suite/lifecycle.json.  Parts are idempotent —
# the serval-only checkpoints (v3/v4/gaefix/anchors) join the same tree
# when serval-paris returns.
#
# uv --no-sync everywhere (dgy torch downgrade); GPU 0; ≤6 concurrent
# parts.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2
Q=reports/v6w_lifecycle_queue.log
SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
AGENTS="hc_v6 hc_v6_last topsis empirical_topsis empirical_spt shortest_processing optimal_assignment batch_milp greedy_reward shortest_queue least_busy least_fatigued round_robin random train_weakest reserve_specialist"
say() { echo "$(date -u +%FT%TZ) [v6w-lc] $*" | tee -a "$Q"; }
mkdir -p reports

say "V6W LIFECYCLE ARMED (pid $$, $(echo $AGENTS | wc -w) agents)"

part() {  # $1 agent
  local A=$1
  if [ -s "$PARTS/$A/lifecycle/episodes.csv" ]; then
    say "part $A/lifecycle cached"
    return
  fi
  say "part $A/lifecycle start"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario lifecycle --record-every 200 --agents "$A" \
    --eval-seed "$SEED" --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_lifecycle.log" 2>&1
  say "part $A/lifecycle rc=$?"
}

NJOBS=0
for A in $AGENTS; do
  part "$A" &
  NJOBS=$((NJOBS + 1))
  if [ $((NJOBS % 6)) = 0 ]; then wait; fi
done
wait
say "all parts done"

nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
  --parts "$PARTS" --dest "$OUTROOT/lifecycle" --scenario lifecycle \
  >> reports/v6w_lc_merge.log 2>&1
say "merge lifecycle rc=$?"

nice -n 10 uv run --no-sync python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v6w_lc_merge.log 2>&1
say "analysis rc=$?"
say "V6W LIFECYCLE DONE"
