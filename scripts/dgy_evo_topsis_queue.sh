#!/usr/bin/env bash
# Evo-TOPSIS (GA-tuned multi-criteria rule, the benchmark's metaheuristic
# baseline) on dgy: CPU tuning on the multiscale training worlds, then the
# 5-scenario benchmark parts for the honest rule and its informed twin.
#
# Stage 1 (CPU, WORKERS processes): scripts/tune_evo_topsis.py ->
#   run_configs/agents/evo_topsis_weights.json (cached: skipped if present)
#   log reports/evo_topsis_tune.log, history reports/evo_topsis_tune_history.csv
# Stage 2: parts for evo_topsis + evo_topsis_inf x 5 scenarios into the
#   idempotent parts tree reports/hvp_v6w_parts/<agent>/<scenario>
#   (the harness loads every checkpoint before filtering, so a GPU is
#   exposed for that; the rules themselves run on CPU).  NO merge here:
#   pull the parts + the weights JSON to the desktop and merge into the
#   canonical local reports/hvp_eval_v6w.
# Markers: DONE_TUNE_EVO_TOPSIS rc= -> EVO TOPSIS TUNE DONE -> EVO TOPSIS DGY QUEUE DONE
# uv --no-sync everywhere (deliberate torch downgrade on dgy).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
Q=reports/evo_topsis_dgy_queue.log
EVAL_SEED=20260722
PARTS=reports/hvp_v6w_parts
WEIGHTS=run_configs/agents/evo_topsis_weights.json
KEYS="evo_topsis evo_topsis_inf"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
WORKERS="${WORKERS:-24}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
say() { echo "$(date -u +%FT%TZ) [evo-topsis-dgy] $*" | tee -a "$Q"; }
mkdir -p reports "$PARTS" run_configs/agents
say "EVO TOPSIS DGY QUEUE ARMED (pid $$, workers $WORKERS, GPU $GPU for parts)"

if [ -s "$WEIGHTS" ]; then
  say "tune cached ($WEIGHTS)"
else
  say "tune start: pop 24, gens 20, worlds 6, val 12, workers $WORKERS"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 nice -n 10 uv run --no-sync python \
    scripts/tune_evo_topsis.py --pop 24 --gens 20 --worlds 6 --val-worlds 12 \
    --workers "$WORKERS" --seed 7 --out "$WEIGHTS" \
    --history reports/evo_topsis_tune_history.csv \
    >> reports/evo_topsis_tune.log 2>&1
  rc=$?
  say "DONE_TUNE_EVO_TOPSIS rc=$rc"
  if [ "$rc" != 0 ] || [ ! -s "$WEIGHTS" ]; then say "ABORT: tuning failed (see reports/evo_topsis_tune.log)"; exit 1; fi
fi
say "EVO TOPSIS TUNE DONE: weights = $(uv run --no-sync python -c "import json;print(json.load(open('$WEIGHTS'))['weights'])" 2>/dev/null)"

part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "part $A/$S start"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 nice -n 10 uv run --no-sync python \
    scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/v6w_part_${A}_${S}.log" 2>&1
  say "part $A/$S rc=$?"
}

NJOBS=0
for S in $SCENARIOS; do
  for A in $KEYS; do
    part "$A" "$S" &
    NJOBS=$((NJOBS + 1))
    if [ $((NJOBS % 5)) = 0 ]; then wait; fi
  done
done
wait
for S in $SCENARIOS; do
  for A in $KEYS; do
    [ -s "$PARTS/$A/$S/episodes.csv" ] || say "MISSING part $A/$S (see reports/v6w_part_${A}_${S}.log)"
  done
done
say "EVO TOPSIS DGY QUEUE DONE"
