#!/usr/bin/env bash
# Corrected-world benchmark generation (v6w) on dgy — run inside a
# zellij session.
#
# First generation under KATA-1 (travel_time 15 + failure-wise
# knowledge live).  NEW out-root — do NOT merge into hvp_eval_v4
# (different world).  Roster = hc_v6 best/last + every checkpoint-free
# baseline.  The v3/v4/gaefix/anchor checkpoints live only on
# serval-paris (down) — re-evaluate them into the same parts tree when
# it returns; parts are idempotent.
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
Q=reports/v6w_bench_queue.log
SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
AGENTS="hc_v6 hc_v6_last topsis empirical_topsis empirical_spt shortest_processing optimal_assignment batch_milp greedy_reward shortest_queue least_busy least_fatigued round_robin random train_weakest reserve_specialist"
SCENARIOS="baseline small_scale massive_scale very_long"
say() { echo "$(date -u +%FT%TZ) [v6w] $*" | tee -a "$Q"; }
mkdir -p reports

say "V6W BENCH ARMED (pid $$, $(echo $AGENTS | wc -w) agents x 4 scenarios)"

part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  say "part $A/$S start"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}

NJOBS=0
for S in $SCENARIOS; do
  for A in $AGENTS; do
    part "$A" "$S" &
    NJOBS=$((NJOBS + 1))
    if [ $((NJOBS % 6)) = 0 ]; then wait; fi
  done
done
wait
say "all parts done"

for S in $SCENARIOS; do
  nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/v6w_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run --no-sync python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v6w_merge.log 2>&1
say "analysis rc=$?"
say "V6W BENCH DONE"
