#!/usr/bin/env bash
# Disruption-instrumented benchmark generation on dgy — run inside zellij.
#
# Purpose (user request 2026-09-02): re-run the summary-table roster
# (14 deployable agents x 5 scenarios) with the per-type disruption
# metrics (disruptions_{injury,exhaustion,vacation} counts + cumulative
# held time, episode totals AND step trajectories).
#
# OUT-ROOT IS SEPARATE from the published generation: eval
# nondeterminism at the 30-technician scenarios means these rows will
# NOT byte-match reports/hvp_eval_v6w — never merge them there.
# Parts: reports/hvp_disr_parts/<agent>/<scenario>; merged tree:
# reports/hvp_eval_disr.
#
# GPUs 0-2 (GPU 3 dead), lanes assigned round-robin; uv --no-sync
# everywhere (deliberate torch downgrade on dgy).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/disr_bench_queue.log
EVAL_SEED=20260722
PARTS=reports/hvp_disr_parts
OUTROOT=reports/hvp_eval_disr
KEYS="hc_v6 ft_quality empirical_topsis empirical_spt batch_milp shortest_queue least_fatigued round_robin least_busy train_weakest random a2c_mlp grpo_mlp dql_mlp"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
say() { echo "$(date -u +%FT%TZ) [disr] $*" | tee -a "$Q"; }
mkdir -p reports "$PARTS"

say "DISR BENCH QUEUE ARMED (pid $$, 14 agents x 5 scenarios, GPUs 0-2)"

# ---------- smoke gate: instrumentation present end-to-end ----------
SMOKE="reports/disr_smoke_part"
if [ ! -s "$SMOKE/small_scale/episodes.csv" ]; then
  say "smoke start (random @ small_scale)"
  CUDA_VISIBLE_DEVICES=0 nice -n 10 uv run --no-sync python \
    scripts/eval_human_vs_performance.py \
    --scenario small_scale --agents random --eval-seed "$EVAL_SEED" \
    --out-root "$SMOKE" > reports/disr_smoke.log 2>&1
  say "smoke rc=$?"
fi
if ! head -1 "$SMOKE/small_scale/episodes.csv" 2>/dev/null \
     | grep -q "disruption_time_vacation"; then
  say "ABORT: smoke episodes.csv lacks per-type disruption columns"
  exit 1
fi
say "smoke gate passed (per-type columns present)"

part() {  # $1 agent, $2 scenario, $3 gpu
  local A=$1 S=$2 G=$3 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "part $A/$S start (GPU $G)"
  CUDA_VISIBLE_DEVICES=$G nice -n 10 uv run --no-sync python \
    scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/disr_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}

NJOBS=0
for S in $SCENARIOS; do
  for A in $KEYS; do
    part "$A" "$S" "$((NJOBS % 3))" &
    NJOBS=$((NJOBS + 1))
    if [ $((NJOBS % 6)) = 0 ]; then wait; fi
  done
  wait
  say "scenario $S parts done"
done
wait
say "all disr parts done"

for S in $SCENARIOS; do
  nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/disr_merge.log 2>&1
  say "merge $S rc=$?"
done
say "DISR BENCH QUEUE DONE"
