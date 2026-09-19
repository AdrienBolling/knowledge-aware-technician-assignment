#!/usr/bin/env bash
# Sequential non-RL baselines (rolling-horizon planner, greedy on the v5
# training reward), dgy CPU lane — run inside zellij.
#
# Protocol = the published v6w generation: eval seed 20260722; small_scale
# 3 eps, baseline 5 eps, massive_scale 3 eps @ 25k steps, very_long and
# lifecycle 1 ep @ record-every 200; PYTHONHASHSEED=0 in every part
# (collision-free machine ids for the benchmark layouts); CPU only.
# Scenario order: lifecycle, massive_scale, very_long, baseline, small_scale.
#
# Parts write <OUT>/<agent>/<scenario>/{episodes.csv,steps.csv.gz}; a part
# with an episodes.csv is skipped (a rerun resumes).  No part starts unless
# its duration guard ends before DEADLINE_UTC.  One line per part with rc=
# (rc=97: exit 0 but no episodes.csv; rc=98: skipped by the guard).
# Markers: "SEQBASE QUEUE DONE" only when every part succeeded (or was
# cached), else "SEQBASE QUEUE FAILED".  TAG changes the marker prefix
# (e.g. TAG=SEQBASE-SIMFIX for the fixed-simulator lane).
#
# Planner parameters and reward scales: run_configs/agents/seqbase_mpc.json
# of this tree.  Layout on dgy: this tree at ~/kata_seqbase with .venv linked
# to the main checkout's venv (deliberate torch 2.7.1+cu126; never uv sync).
#
# Smoke: SIM=3000 NEPS=1 OUT=reports/seqbase_smoke Q=reports/seqbase_smoke.log
#        LOGDIR=reports/seqbase_smoke_logs scripts/dgy_seqbase_queue.sh
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="${OUT:-reports/seqbase}"
Q="${Q:-reports/seqbase_queue.log}"
LOGDIR="${LOGDIR:-reports/seqbase_logs}"
TAG="${TAG:-SEQBASE}"
SEED=20260722
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-20 11:00:00}"
MAXC="${MAXC:-8}"
AGENTS="${AGENTS-rolling_mpc greedy_train_reward}"
SCENARIOS="${SCENARIOS-lifecycle massive_scale very_long baseline small_scale}"
SIM="${SIM:-}"     # smoke only: override every horizon
NEPS="${NEPS:-}"   # smoke only: override every episode count
STATUS="$LOGDIR/status"

say() { echo "$(date -u +%FT%TZ) [seqbase] $*" | tee -a "$Q"; }

# Duration guards in minutes (conservative: ~1M decisions at very_long,
# ~560k at lifecycle, a decision ~3-8 ms on a loaded dgy CPU).
guard_min() {
  if [ -n "$SIM" ]; then echo 30; return; fi
  case "$1" in
    very_long) echo 540 ;;
    lifecycle) echo 360 ;;
    massive_scale) echo 120 ;;
    *) echo 60 ;;
  esac
}
fits() { [ $(( $(date -u +%s) + $1 * 60 )) -le "$(date -u -d "$DEADLINE_UTC" +%s)" ]; }

part() {  # $1 agent  $2 scenario
  local A=$1 S=$2 EXTRA="" G rc
  local dir="$OUT/$A"
  if [ -s "$dir/$S/episodes.csv" ]; then
    say "$A/$S cached rc=0"; echo 0 > "$STATUS/${A}_${S}"; return 0
  fi
  G=$(guard_min "$S")
  if ! fits "$G"; then
    say "$A/$S skipped: guard ${G} min ends after $DEADLINE_UTC rc=98"; echo 98 > "$STATUS/${A}_${S}"; return 0
  fi
  case "$S" in
    massive_scale) EXTRA="--steps 25000 --n-eps 3" ;;
    very_long|lifecycle) EXTRA="--record-every 200" ;;
  esac
  [ -n "$SIM" ] && EXTRA="$EXTRA --sim $SIM"
  [ -n "$NEPS" ] && EXTRA="$EXTRA --n-eps $NEPS"
  mkdir -p "$dir"
  say "$A/$S start (guard ${G} min)"
  nice -n 10 "$PY" scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$SEED" --out-root "$dir" \
    > "$LOGDIR/${A}_${S}.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ] && [ ! -s "$dir/$S/episodes.csv" ]; then rc=97; fi
  echo "$rc" > "$STATUS/${A}_${S}"
  say "$A/$S rc=$rc"
}

mkdir -p reports "$OUT" "$LOGDIR" "$STATUS"
rm -f "$STATUS"/*
say "$TAG QUEUE ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown), out $OUT, MAXC $MAXC, agents [$AGENTS], scenarios [$SCENARIOS]${SIM:+, SMOKE sim $SIM})"
[ -f run_configs/agents/seqbase_mpc.json ] || say "WARNING: run_configs/agents/seqbase_mpc.json missing (defaults, unit scales)"

N=0
for S in $SCENARIOS; do
  for A in $AGENTS; do
    N=$((N + 1))
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do wait -n; done
    part "$A" "$S" &
    sleep 2
  done
done
wait

OK=0; BAD=""
for S in $SCENARIOS; do
  for A in $AGENTS; do
    rc=$(cat "$STATUS/${A}_${S}" 2>/dev/null || echo missing)
    if [ "$rc" = 0 ]; then OK=$((OK + 1)); else BAD="$BAD $A/$S:$rc"; fi
  done
done
say "parts ok $OK/$N${BAD:+; failed:$BAD}"
if [ -f scripts/seqbase_compare.py ]; then
  PUB="${PUB:-reports/hvp_eval_v6w}"
  [ -d "$PUB" ] || PUB="$HOME/repositories/knowledge-aware-technician-assignment/reports/hvp_eval_v6w"
  nice -n 10 "$PY" scripts/seqbase_compare.py --root "$OUT" --published "$PUB" \
    --parts "$(dirname "$PUB")/hvp_v6w_parts" --extra "$PUB/mlp_last_step_metrics.csv" \
    --out "$OUT/SEQBASE_COMPARE.md" >> "$LOGDIR/compare.log" 2>&1
  say "compare rc=$?"
fi
if [ "$OK" = "$N" ]; then say "$TAG QUEUE DONE"; else say "$TAG QUEUE FAILED"; fi
