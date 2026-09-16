#!/usr/bin/env bash
# Neutral-turnover variants of S5 (lifecycle), dgy — run inside zellij.
#
# Reviewer point: S5 retires the most knowledgeable technicians, a targeted
# key-person stress test.  Two evaluation-only variants
# (scripts/make_turnover_configs.py) supplement it:
#   lifecycle_random_retire  retire_technician events use select=random
#   lifecycle_random_timing  random selection + retire/hire pairs at times
#                            drawn once, uniformly over the horizon
# Everything else is identical to lifecycle.json.  One harness part per
# (agent, variant), S5 protocol: eval seed 20260722, record every 200.
#
# Same-code reference: the published S5 rows (reports/hvp_eval_v6w) come
# from older runs on other hardware, without PYTHONHASHSEED=0.  The queue
# also reruns the targeted `lifecycle` scenario with this code, so each
# variant has a like-for-like targeted reference.
#
# CPU lane (CUDA_VISIBLE_DEVICES=""), at most MAXJ parts at a time:
#   10 heuristics on both variants and on targeted lifecycle, then the
#   optional MLP anchors (a2c_mlp, grpo_mlp, dql_mlp) on both variants.
# GPU lane (LEARNED_GPU, default 1), one learned part at a time, started
# after WAIT_MARKER appears in WAIT_LOG (the live-S5 benchmarks on GPU 1).
# Order (key comparisons first): ft_quality (HTT-RL qua.) and po_v6
# (production-only twin) on random_retire, then on random_timing, then
# hc_v6 (HTT-RL ref.) on both, then the targeted lifecycle reruns of po_v6
# and hc_v6 (ft_quality already has one: ~/kata_live live-S5 queue, same
# src).  Only the best checkpoints run (the paper's main comparison);
# ft_quality and po_v6 best == last.
#
# Booking: no part starts unless it can finish before DEADLINE_UTC (each part
# has a duration guard in hours).  Parts are cached: a rerun skips finished
# parts.
#
# Layout expected on dgy: this tree at ~/kata_turnover, with .venv and the
# needed checkpoints/ subdirectories linked to the main checkout.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export PYTHONHASHSEED=0
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/turnover_queue.log
OUT=reports/hvp_turnover_parts
VARIANTS="lifecycle_random_retire lifecycle_random_timing"
HEUR_SCENARIOS="${HEUR_SCENARIOS:-$VARIANTS lifecycle}"
HEURISTICS="${HEURISTICS:-topsis shortest_processing optimal_assignment reserve_specialist shortest_queue least_fatigued round_robin least_busy train_weakest random}"
MLP_ANCHORS="${MLP_ANCHORS:-a2c_mlp grpo_mlp dql_mlp}"
# key/variant pairs, run in this order
LEARNED="${LEARNED:-ft_quality/lifecycle_random_retire po_v6/lifecycle_random_retire ft_quality/lifecycle_random_timing po_v6/lifecycle_random_timing hc_v6/lifecycle_random_retire hc_v6/lifecycle_random_timing po_v6/lifecycle hc_v6/lifecycle}"
MAXJ="${MAXJ:-8}"
LEARNED_GPU="${LEARNED_GPU:-1}"
WAIT_LOG="${WAIT_LOG:-$HOME/kata_live/reports/live_s5_queue.log}"
WAIT_MARKER="${WAIT_MARKER:-LIVE LANE DONE}"
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-20 11:00:00}"
H_HEUR=2 H_MLP=6 H_LEARNED=10   # duration guards (hours)
say() { echo "$(date -u +%FT%TZ) [turnover] $*" | tee -a "$Q"; }
fits() {  # $1 hours
  [ $(( $(date -u +%s) + $1 * 3600 )) -le "$(date -u -d "$DEADLINE_UTC" +%s)" ]
}
mkdir -p reports "$OUT"

say "TURNOVER QUEUE ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown))"

part() {  # $1 harness key  $2 variant  $3 device (cpu|gpu index)  $4 guard hours
  local K=$1 V=$2 D=$3 G=$4 CVD
  if [ -s "$OUT/$K/$V/episodes.csv" ]; then say "$K/$V cached"; return; fi
  fits "$G" || { say "$K/$V skipped: cannot finish before $DEADLINE_UTC"; return; }
  if [ "$D" = cpu ]; then CVD=""; else CVD=$D; fi
  say "$K/$V start (device $D)"
  CUDA_VISIBLE_DEVICES=$CVD nice -n 5 "$PY" scripts/eval_human_vs_performance.py \
    --scenario "$V" --agents "$K" --eval-seed 20260722 --record-every 200 \
    --out-root "$OUT/$K" > "reports/turnover_${K}_${V}.log" 2>&1
  say "$K/$V rc=$?"
}

(
  for K in $HEURISTICS; do
    for V in $HEUR_SCENARIOS; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
      part "$K" "$V" cpu "$H_HEUR" &
      sleep 5
    done
  done
  wait
  say "TURNOVER HEURISTICS DONE"
  for K in $MLP_ANCHORS; do
    for V in $VARIANTS; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
      part "$K" "$V" cpu "$H_MLP" &
      sleep 5
    done
  done
  wait
  say "TURNOVER CPU LANE DONE"
) &
(
  if [ -n "$WAIT_MARKER" ]; then
    say "GPU lane waiting for '$WAIT_MARKER' in $WAIT_LOG"
    until grep -q "$WAIT_MARKER" "$WAIT_LOG" 2>/dev/null; do sleep 300; done
    say "GPU lane released"
  fi
  for KV in $LEARNED; do
    part "${KV%%/*}" "${KV#*/}" "$LEARNED_GPU" "$H_LEARNED"
  done
  say "TURNOVER GPU LANE DONE"
) &
wait
say "TURNOVER QUEUE DONE"
