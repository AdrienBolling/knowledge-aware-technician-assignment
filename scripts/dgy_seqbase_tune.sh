#!/usr/bin/env bash
# Sequential baselines, dgy CPU: calibrate the v5 reward scales, then tune
# horizon_k and terminal_weight of the rolling-horizon planner.  Both run on
# the TRAINING worlds (train_multiscale_v5 sampler; world seeds 90000+,
# 70000+, 80000+, disjoint from the eval seed 20260722).  Run inside zellij.
#
# Writes run_configs/agents/seqbase_mpc.json (sigma, calibration, horizon_k,
# terminal_weight, tuning).  Log: reports/seqbase_tune.log.
# Markers: SEQBASE CALIBRATION rc= / SEQBASE TUNE rc= / SEQBASE TUNE CHAIN DONE
# (or SEQBASE TUNE CHAIN FAILED).
# Smoke: SMOKE=1 scripts/dgy_seqbase_tune.sh (tiny horizons, separate files).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKERS="${WORKERS:-14}"
LOG="${LOG:-reports/seqbase_tune.log}"
PARAMS="${PARAMS:-run_configs/agents/seqbase_mpc.json}"
mkdir -p reports
say() { echo "$(date -u +%FT%TZ) [seqbase-tune] $*" | tee -a "$LOG"; }

if [ -n "${SMOKE:-}" ]; then
  PARAMS=reports/seqbase_smoke_params.json
  CAL="--worlds 2 --workers 2 --sim-min 8000 --sim-max 8000"
  TUNE="--smoke --history reports/seqbase_smoke_tune_history.csv"
else
  CAL="--worlds 16 --workers $WORKERS"
  TUNE="--workers $WORKERS --history reports/seqbase_tune_history.csv"
fi

say "ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown), params $PARAMS, workers $WORKERS${SMOKE:+, SMOKE})"
nice -n 10 "$PY" scripts/calibrate_seqbase.py $CAL --out "$PARAMS" >> "$LOG" 2>&1
rc=$?
say "SEQBASE CALIBRATION rc=$rc"
if [ "$rc" != 0 ]; then say "SEQBASE TUNE CHAIN FAILED"; exit 1; fi
nice -n 10 "$PY" scripts/tune_seqbase.py $TUNE --params "$PARAMS" --out "$PARAMS" >> "$LOG" 2>&1
rc=$?
say "SEQBASE TUNE rc=$rc"
if [ "$rc" != 0 ]; then say "SEQBASE TUNE CHAIN FAILED"; exit 1; fi
say "SEQBASE TUNE CHAIN DONE"
