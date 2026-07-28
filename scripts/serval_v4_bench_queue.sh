#!/usr/bin/env bash
# Chain 1 of the v4 generation: full benchmark ladder in the
# mechanisms-ON world (knowledge decay + Kijima alpha=0.25), fresh
# out-root reports/hvp_eval_v4 / parts root reports/hvp_v4_vl_parts.
# Existing (old-world-trained) checkpoints are evaluated as the
# "before" agents; hc_v4 merges in later via chain 2.
#
# Wave A (concurrent): 3 standard scenarios; anchor retrainings in the
#   new world (old anchors moved to checkpoints/anchors_oldworld);
#   5M runs — 14 heuristics (pool 6) + 8 RL checkpoints (pool 4).
# Wave B: anchor eval --merge, 5M parts merge, analyzer.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export Q=reports/v4_bench_queue.log
export SEED=20260722
export OUTROOT=reports/hvp_eval_v4
export PARTS=reports/hvp_v4_vl_parts
say() { echo "$(date -u +%FT%TZ) [v4b] $*" >> "$Q"; }
export -f say

say "V4 BENCH QUEUE ARMED (pid $$)"

# Old-world anchors aside so the fixed eval paths resolve to new-world ones.
if [ -f checkpoints/anchors/ppo_transformer_best.pt ] \
   && [ ! -d checkpoints/anchors_oldworld ]; then
  mv checkpoints/anchors checkpoints/anchors_oldworld
  mkdir -p checkpoints/anchors
  say "old-world anchors moved to checkpoints/anchors_oldworld"
fi

std() {
  local S=$1; shift
  if [ -f "$OUTROOT/$S/episodes.csv" ]; then say "std $S done — skip"; return 0; fi
  say "std $S start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" "$@" --agents all --eval-seed "$SEED" \
    --out-root "$OUTROOT" >> "reports/v4_eval_std_${S}.log" 2>&1
  say "std $S rc=$?"
}

vl() {
  local A=$1
  if [ -f "$PARTS/$A/very_long/episodes.csv" ]; then say "vl $A done — skip"; return 0; fi
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "$PARTS/$A" \
    >> "reports/v4_eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
export -f vl

anchor() {
  local A=$1
  if [ -f "checkpoints/anchors/${A}_best.pt" ]; then say "anchor $A done — skip"; return 0; fi
  say "anchor $A train start"
  nice -n 10 uv run python -m experiment.cli \
    --env run_configs/benchmark_suite/baseline.json \
    --agent "run_configs/agents/${A}.json" \
    --experiment run_configs/experiments/anchor_train.json \
    --exp-id "v4_anchor_${A}" >> "reports/v4_train_anchor_${A}.log" 2>&1
  say "anchor $A train rc=$?"
}

# ---------------- Wave A ----------------
std baseline &
std small_scale &
std massive_scale --steps 25000 --n-eps 3 &
anchor rainbow_dqn &
anchor ppo_transformer &

HEUR="random round_robin least_busy least_fatigued shortest_queue
shortest_processing optimal_assignment batch_milp topsis
reserve_specialist greedy_reward train_weakest empirical_spt
empirical_topsis"
printf '%s\n' $HEUR | nice -n 10 xargs -P 6 -I{} bash -c 'vl "$@"' _ {} &

RL="human performance hc_v2 gaefix hc_v3 hc_v3_last hc_v3_ft hc_v3_ft_mid"
printf '%s\n' $RL | nice -n 10 xargs -P 4 -I{} bash -c 'vl "$@"' _ {} &

wait
say "wave A complete"

# ---------------- Wave B ----------------
say "anchor eval start"
nice -n 10 uv run python scripts/eval_human_vs_performance.py \
  --scenario baseline --agents ppo_transformer,rainbow_dqn \
  --eval-seed "$SEED" --merge --out-root "$OUTROOT" \
  >> reports/v4_eval_std_anchors.log 2>&1
say "anchor eval rc=$?"

nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts "$PARTS" --dest "$OUTROOT/very_long" \
  >> reports/v4_eval_merge.log 2>&1
say "vl merge rc=$?"
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v4_eval_merge.log 2>&1
say "analysis rc=$?"
say "V4 BENCH QUEUE DONE"
