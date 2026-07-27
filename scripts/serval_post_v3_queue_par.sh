#!/usr/bin/env bash
# Parallel successor to serval_post_v3_queue.sh — assumes v3 training and
# the plateau gate are DONE (checkpoints/hc_v3_final populated) and the
# machine has spare capacity (28 cores, GPU mostly free).
#
# Wave A (all concurrent):
#   - standard-ladder scenarios not yet done (skip if episodes.csv exists)
#   - both anchor trainings (rainbow=CPU, ppo_transformer=GPU)
#   - all 5M very_long runs, one out-root PER AGENT so no CSV races:
#     14 heuristics through an 8-way pool, 4 RL checkpoints unpooled
# Wave B (after everything above):
#   - anchor eval merged into the baseline scenario
#   - merge 5M parts into reports/hvp_eval_v3/very_long + analyzer
#
# Idempotent: re-running skips any part whose episodes.csv already
# exists, so a crash costs one agent's run, not the ladder.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export Q=reports/post_v3_queue.log
export SEED=20260722
say() { echo "$(date -u +%FT%TZ) [par] $*" >> "$Q"; }
export -f say

say "PARALLEL QUEUE ARMED (pid $$)"

(
  while true; do
    sleep 600
    echo "$(date -u +%FT%TZ) [par][hb] $(ls reports/hvp_vl_parts 2>/dev/null | wc -l) vl parts present" >> "$Q"
  done
) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

std() {
  local S=$1; shift
  if [ -f "reports/hvp_eval_v3/$S/episodes.csv" ]; then
    say "std $S already done — skip"; return 0
  fi
  say "std $S start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" "$@" --agents all --eval-seed "$SEED" \
    --out-root reports/hvp_eval_v3 >> "reports/eval_std_${S}.log" 2>&1
  say "std $S rc=$?"
}

vl() {
  local A=$1
  if [ -f "reports/hvp_vl_parts/$A/very_long/episodes.csv" ]; then
    say "vl $A already done — skip"; return 0
  fi
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "reports/hvp_vl_parts/$A" \
    >> "reports/eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
export -f vl

anchor() {
  local A=$1
  if [ -f "checkpoints/anchors/${A}_best.pt" ]; then
    say "anchor $A already trained — skip"; return 0
  fi
  say "anchor $A train start"
  nice -n 10 uv run python -m experiment.cli \
    --env run_configs/benchmark_suite/baseline.json \
    --agent "run_configs/agents/${A}.json" \
    --experiment run_configs/experiments/anchor_train.json \
    --exp-id "anchor_${A}" >> "reports/train_anchor_${A}.log" 2>&1
  say "anchor $A train rc=$?"
}

# ---------------- Wave A ----------------
std small_scale &
std massive_scale --steps 25000 --n-eps 3 &
anchor rainbow_dqn &
anchor ppo_transformer &

HEUR="random round_robin least_busy least_fatigued shortest_queue
shortest_processing optimal_assignment batch_milp topsis
reserve_specialist greedy_reward train_weakest empirical_spt
empirical_topsis"
printf '%s\n' $HEUR | nice -n 10 xargs -P 8 -I{} bash -c 'vl "$@"' _ {} &
for A in human performance hc_v3 hc_v3_last; do
  vl "$A" &
done

wait
say "wave A complete"

# ---------------- Wave B ----------------
say "anchor eval start"
nice -n 10 uv run python scripts/eval_human_vs_performance.py \
  --scenario baseline --agents ppo_transformer,rainbow_dqn \
  --eval-seed "$SEED" --merge --out-root reports/hvp_eval_v3 \
  >> reports/eval_std_anchors.log 2>&1
say "anchor eval rc=$?"

nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts reports/hvp_vl_parts --dest reports/hvp_eval_v3/very_long \
  >> reports/eval_vl_merge.log 2>&1
say "vl merge rc=$?"

nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root reports/hvp_eval_v3 >> reports/eval_vl_merge.log 2>&1
say "analysis rc=$?"
say "PARALLEL QUEUE DONE"
