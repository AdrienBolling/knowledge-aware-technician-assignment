#!/usr/bin/env bash
# Simulator-fix measurement (branch sim-event-fixes), dgy — run inside zellij.
#
# Measures the effect of the two simulator fixes on the five benchmark
# scenarios.  Three simulator configs (switches of kata.core.legacy):
#   legacy  KATA_LEGACY_BUFFER_INTERRUPT=1 KATA_LEGACY_MACHINE_TRACKING=1
#   fix1    buffer fix only (decision-boundary machine tracking kept)
#   fixed   both fixes
# Protocol = the published v6w generation: eval seed 20260722; small_scale
# 3 eps, baseline 5 eps, massive_scale 3 eps @ 25k steps, very_long and
# lifecycle 1 ep @ record-every 200.  PYTHONHASHSEED=0 in every part
# (collision-free machine ids for the benchmark layouts).
#
# CPU lane (CUDA_VISIBLE_DEVICES="", at most MAXC parts):
#   topsis, shortest_processing, random under legacy and fixed on all five
#   scenarios, plus fix1 on lifecycle only.  The heuristics read no machine
#   tracking state, so fix1 == fixed for them on S1-S4; on S5 the lifecycle
#   `replace_machine select=most_breakdowns` events read the tracked
#   breakdown counts, so fix1 differs there.
# GPU lane (at most MAXG parts; each part takes the GPU of GPUS with the most
#   free memory at its start):
#   ft_quality (HTT-RL qua.) and hc_v6 (HTT-RL ref.), best checkpoints:
#   lifecycle fixed/legacy/fix1, very_long fixed, massive_scale
#   fixed/legacy/fix1, baseline fixed, small_scale fixed (S5 first).
#
# Parts are cached: a part with an episodes.csv is skipped, so a rerun
# resumes.  No part starts unless its duration guard ends before
# DEADLINE_UTC.  One log line per part with rc=; marker SIMFIX MEASURE DONE.
#
# Layout on dgy: this tree at ~/kata_simfix with .venv linked to the main
# checkout's venv (deliberate torch 2.7.1+cu126) and checkpoints/
# {hc_v6_final,ft_quality}/set_transformer_best.pt linked.
#
# Smoke: SIM=3000 NEPS=1 OUT=reports/simfix_smoke Q=reports/simfix_smoke.log
#        LOGDIR=reports/simfix_smoke_logs scripts/dgy_simfix_measure.sh
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export PYTHONHASHSEED=0
export WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OUT="${OUT:-reports/simfix}"
Q="${Q:-reports/simfix_queue.log}"
LOGDIR="${LOGDIR:-reports/simfix_logs}"
SEED=20260722
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-20 11:00:00}"
MAXC="${MAXC:-8}"
MAXG="${MAXG:-8}"
GPUS="${GPUS:-0 1 2}"          # GPU 3 is unusable on dgy
GPU_STAGGER="${GPU_STAGGER:-45}"  # s between GPU part starts (memory-based pick)
SIM="${SIM:-}"                 # smoke only: override every horizon
NEPS="${NEPS:-}"               # smoke only: override every episode count
HEURISTICS="${HEURISTICS-topsis shortest_processing random}"
LEARNED="${LEARNED-ft_quality hc_v6}"

say() { echo "$(date -u +%FT%TZ) [simfix] $*" | tee -a "$Q"; }

# Duration guards in minutes (conservative; contended V100s run a learned
# decision in 30-120 ms, a heuristic decision in ~3 ms).
guard_min() {  # $1 lane (cpu|gpu)  $2 scenario
  if [ -n "$SIM" ]; then echo 30; return; fi
  case "$1:$2" in
    gpu:lifecycle) echo 1800 ;;   # 560k decisions
    gpu:very_long) echo 3000 ;;   # 1.0M decisions
    gpu:massive_scale) echo 360 ;;
    gpu:baseline) echo 120 ;;
    gpu:small_scale) echo 60 ;;
    cpu:lifecycle|cpu:very_long) echo 120 ;;
    *) echo 60 ;;
  esac
}
fits() {  # $1 minutes
  [ $(( $(date -u +%s) + $1 * 60 )) -le "$(date -u -d "$DEADLINE_UTC" +%s)" ]
}

pick_gpu() {  # the GPU of $GPUS with the most free memory now
  local best
  best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v ok=" $GPUS " '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && index(ok, " " $1 " ") {print $2, $1}' \
    | sort -rn | head -1 | awk '{print $2}')
  echo "${best:-${GPUS%% *}}"
}

part() {  # $1 config  $2 agent  $3 scenario  $4 lane (cpu|gpu)
  local C=$1 A=$2 S=$3 L=$4 LB LT EXTRA="" DEV="" G
  local dir="$OUT/$C/$A"
  if [ -s "$dir/$S/episodes.csv" ]; then say "$C/$A/$S cached"; return 0; fi
  G=$(guard_min "$L" "$S")
  if ! fits "$G"; then
    say "$C/$A/$S skipped: guard ${G} min ends after $DEADLINE_UTC"; return 0
  fi
  case "$C" in
    legacy) LB=1; LT=1 ;;
    fix1) LB=0; LT=1 ;;
    fixed) LB=0; LT=0 ;;
    *) say "$C/$A/$S unknown config"; return 1 ;;
  esac
  case "$S" in
    massive_scale) EXTRA="--steps 25000 --n-eps 3" ;;
    very_long|lifecycle) EXTRA="--record-every 200" ;;
  esac
  [ -n "$SIM" ] && EXTRA="$EXTRA --sim $SIM"
  [ -n "$NEPS" ] && EXTRA="$EXTRA --n-eps $NEPS"
  [ "$L" = gpu ] && DEV=$(pick_gpu)
  mkdir -p "$dir" "$LOGDIR"
  say "$C/$A/$S start (lane $L${DEV:+, GPU $DEV}, guard ${G} min)"
  KATA_LEGACY_BUFFER_INTERRUPT=$LB KATA_LEGACY_MACHINE_TRACKING=$LT \
  CUDA_VISIBLE_DEVICES="$DEV" nice -n 10 "$PY" scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$SEED" --out-root "$dir" \
    > "$LOGDIR/${C}_${A}_${S}.log" 2>&1
  local rc=$?
  if [ "$rc" = 0 ] && [ ! -s "$dir/$S/episodes.csv" ]; then rc=97; fi
  say "$C/$A/$S rc=$rc"
}

mkdir -p reports "$OUT" "$LOGDIR"
say "SIMFIX MEASURE ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown), out $OUT, MAXC $MAXC, MAXG $MAXG${SIM:+, SMOKE sim $SIM})"

for C in $LEARNED; do
  f=checkpoints/$([ "$C" = hc_v6 ] && echo hc_v6_final || echo "$C")/set_transformer_best.pt
  [ -f "$f" ] || say "WARNING: checkpoint $f missing ($C parts will fail)"
done

(
  for S in lifecycle very_long massive_scale baseline small_scale; do
    CONFS="legacy fixed"
    [ "$S" = lifecycle ] && CONFS="legacy fix1 fixed"
    for C in $CONFS; do
      for A in $HEURISTICS; do
        while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do wait -n; done
        part "$C" "$A" "$S" cpu &
        sleep 2
      done
    done
  done
  wait
  say "SIMFIX CPU LANE DONE"
) &
(
  JOBS=""
  for A in $LEARNED; do JOBS="$JOBS fixed/$A/lifecycle"; done
  for A in $LEARNED; do JOBS="$JOBS legacy/$A/lifecycle"; done
  for A in $LEARNED; do JOBS="$JOBS fix1/$A/lifecycle"; done
  for A in $LEARNED; do JOBS="$JOBS fixed/$A/very_long"; done
  for C in fixed legacy fix1; do
    for A in $LEARNED; do JOBS="$JOBS $C/$A/massive_scale"; done
  done
  for S in baseline small_scale; do
    for A in $LEARNED; do JOBS="$JOBS fixed/$A/$S"; done
  done
  for J in $JOBS; do
    C=${J%%/*}; R=${J#*/}; A=${R%%/*}; S=${R#*/}
    if [ -s "$OUT/$C/$A/$S/episodes.csv" ]; then say "$C/$A/$S cached"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAXG" ]; do wait -n; done
    part "$C" "$A" "$S" gpu &
    sleep "$GPU_STAGGER"
  done
  wait
  say "SIMFIX GPU LANE DONE"
) &
wait

if [ -f scripts/simfix_compare.py ]; then
  PUB=reports/hvp_eval_v6w
  [ -d "$PUB" ] || PUB="$HOME/repositories/knowledge-aware-technician-assignment/reports/hvp_eval_v6w"
  nice -n 10 "$PY" scripts/simfix_compare.py --root "$OUT" --published "$PUB" \
    --out "$OUT/SIMFIX_COMPARE.md" >> "$LOGDIR/compare.log" 2>&1
  say "compare rc=$?"
fi
say "SIMFIX MEASURE DONE"
