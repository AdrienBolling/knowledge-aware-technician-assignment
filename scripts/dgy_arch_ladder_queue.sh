#!/usr/bin/env bash
# Architecture ladder on dgy -- run inside zellij (session "ladder").
#
# Reviewer point: the MLP anchors differ from HTT-RL in algorithm, BC
# initialisation, discounting AND encoder, so they cannot attribute the
# gap to the architecture.  Every rung here uses the hc_v6 / hc_perm
# recipe -- PPO, env=train_multiscale_v5 (v5 reward, strict mask), a
# TOPSIS BC start collected with the rung's own network, semi-MDP
# discounting, 600 episodes x 5 envs, seed 42 -- and changes the encoder
# only (run_configs/agents/set_transformer_v6_ladder_<rung>.json):
#
#   flat      flattened slot embeddings through an MLP (order-aware)
#   pool      shared per-slot MLP + masked mean (Deep Sets)
#   plainset  Set Transformer (no RoPE) with plain numeric encoding
#   (top)     hc_perm_dgy = full HTT-RL, hybrid encoding, no RoPE.  It
#             trains separately in ~/kata_perm; this queue only waits
#             for it, links it and benchmarks it.
#
# Per rung: BC -> flag check -> training -> best + last (final.pt) into
# checkpoints/ladder_<rung>_final -> 10 benchmark parts (best, last x 5
# scenarios, eval seed 20260722) in reports/hvp_ladder_parts/<key>/<scenario>.
#
# Placement: flat + pool train on GPU 0 now (shared with hc_perm).  The
# plainset rung and every benchmark use GPU 1, after the live-s5 queue
# writes "LIVE LANE DONE".  GPU 2 (other run) and GPU 3 (dead) are unused.
# Booking: no stage starts that cannot finish before STOP_UTC.  All
# stages are cached, so a rerun continues where the queue stopped.
#
# Queue log: reports/arch_ladder_queue.log.  Markers:
#   LADDER <rung> BC DONE / TRAIN DONE rc=N / CANON DONE
#   LADDER <rung> BENCH DONE (all parts) or BENCH INCOMPLETE (n/total)
#   HC PERM DGY LINKED / HC PERM DGY BENCH DONE or BENCH INCOMPLETE
#   lane <name> rc=N, then ARCH LADDER QUEUE DONE (every lane rc=0, exit 0)
#   or ARCH LADDER QUEUE FAILED (lanes: ...) (exit 1)
# Training logs end with DONE_TRAIN_LADDER_<rung> rc=N.
#
# Layout expected on dgy: this tree at ~/kata_ladder, .venv linked to the
# main checkout's venv (torch 2.7.1+cu126 for the V100s).  Never run a
# bare `uv run` there.  SMOKE=1 runs every stage at tiny sizes on CPU
# (local validation only).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY="${PY:-$HOME/repositories/knowledge-aware-technician-assignment/.venv/bin/python}"
export PYTHONPATH="$ROOT/src"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2

Q=reports/arch_ladder_queue.log
PARTS=reports/hvp_ladder_parts
EVAL_SEED=20260722
STOP_UTC="${STOP_UTC:-2026-09-20 11:00:00}"
GPU_TRAIN="${GPU_TRAIN-0}"          # flat + pool training
GPU_LATE="${GPU_LATE-1}"            # plainset + every benchmark part
MAX_BENCH="${MAX_BENCH:-10}"        # concurrent benchmark parts (1 core each)
LIVE_LOG="${LIVE_LOG:-$HOME/kata_live/reports/live_s5_queue.log}"
PERM_ROOT="${PERM_ROOT:-$HOME/kata_perm}"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
# Upper-bound wall-clock estimates (hours) for the booking gate.
BC_H=2
TRAIN_H=24
part_hours() {
  case $1 in
    very_long) echo 10 ;; lifecycle) echo 8 ;; massive_scale) echo 2 ;; *) echo 1 ;;
  esac
}

# The hc_v6 BC protocol (scripts/dgy_v6_train.sh) and the hc_perm
# training recipe (scripts/paris_perm_train.sh on branch memory-ema).
BC_ARGS=(--episodes 25 --sim-time 200000 --seed 7)
TRAIN_ARGS=(episodes=600 parallel_envs=5 sim_time=275000 sim_time_min=200000
            sim_time_max=350000 eval_interval=200 checkpoint_interval=50 seed=42)
BENCH_SMOKE=()
if [ "${SMOKE:-0}" = 1 ]; then
  BC_ARGS=(--episodes 1 --sim-time 3000 --max-steps 300 --epochs 1 --seed 7)
  TRAIN_ARGS=(episodes=2 parallel_envs=2 sim_time=3000 sim_time_min=2000
              sim_time_max=4000 max_steps=400 rollout_steps=64 eval_interval=2
              eval_episodes=1 checkpoint_interval=1 seed=42 wandb=false)
  BENCH_SMOKE=(--n-eps 1 --sim 2000 --steps 60)
fi

say() { echo "$(date -u +%FT%TZ) [ladder] $*" | tee -a "$Q"; }
past_stop() { [ "$(date -u +%s)" -ge "$(date -u -d "$STOP_UTC" +%s)" ]; }
fits() {  # $1 hours: can a stage started now finish before STOP_UTC?
  [ $(( $(date -u +%s) + $1 * 3600 )) -le "$(date -u -d "$STOP_UTC" +%s)" ]
}
mkdir -p reports checkpoints "$PARTS"

# Expected improvements flags per rung (checked on BC and final checkpoints).
COMMON="numeric_encoding=plain set_positional=False slot_role_binding=True use_feature_context=True use_popart=True rnn_type=none"
flags_of() {
  case $1 in
    flat) echo "cross_slot=flat use_cross_attention=False $COMMON" ;;
    pool) echo "cross_slot=pool use_cross_attention=False $COMMON" ;;
    plainset) echo "cross_slot=attention use_cross_attention=True $COMMON" ;;
    hc_perm) echo "cross_slot=attention numeric_encoding=hybrid set_positional=False slot_role_binding=True use_feature_context=True use_popart=True rnn_type=none" ;;
  esac
}

check_flags() {  # $1 checkpoint  $2 "key=value ..." -> rc 0 when all match
  "$PY" - "$1" "$2" <<'EOF'
import sys
import torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
imp = ck.get("improvements") or {}
bad = []
for kv in sys.argv[2].split():
    k, v = kv.split("=", 1)
    got = imp.get(k, "<absent>")
    if str(got) != v:
        bad.append(f"{k}={got} (want {v})")
print("flags OK" if not bad else "flag mismatch: " + ", ".join(bad))
sys.exit(1 if bad else 0)
EOF
}

wait_live() {  # the live-s5 queue frees GPU 1 when its benchmark lane ends
  if [ ! -f "$LIVE_LOG" ]; then say "WARNING: $LIVE_LOG does not exist (gate waits for it)"; fi
  until grep -qs "LIVE LANE DONE\|LIVE S5 QUEUE DONE" "$LIVE_LOG"; do
    past_stop && { say "GPU 1 gate: past $STOP_UTC, giving up"; return 1; }
    sleep 300
  done
}

with_slot() {  # run "$@" while holding one of MAX_BENCH lock slots
  local i fd rc
  mkdir -p reports/.ladder_slots
  while :; do
    for i in $(seq 1 "$MAX_BENCH"); do
      exec {fd}>"reports/.ladder_slots/slot$i"
      if flock -n "$fd"; then
        "$@"
        rc=$?
        exec {fd}>&-
        return $rc
      fi
      exec {fd}>&-
    done
    sleep 30
  done
}

part() {  # $1 harness key  $2 scenario
  local K=$1 S=$2 EXTRA="" H rc LOG
  LOG="reports/ladder_part_${K}_${S}.log"
  if [ -s "$PARTS/$K/$S/episodes.csv" ]; then say "part $K/$S cached"; return 0; fi
  case $S in
    massive_scale) EXTRA="--steps 25000 --n-eps 3" ;;
    very_long|lifecycle) EXTRA="--record-every 200" ;;
  esac
  H=$(part_hours "$S")
  fits "$H" || { say "part $K/$S skipped: ${H} h would pass $STOP_UTC"; return 1; }
  say "part $K/$S start (GPU ${GPU_LATE:-cpu})"
  # PYTHONHASHSEED=0: collision-free machine ids for all five benchmark
  # layouts (see scripts/dgy_financial_queue.sh).
  PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=$GPU_LATE nice -n 10 "$PY" \
    scripts/eval_human_vs_performance.py --scenario "$S" $EXTRA \
    "${BENCH_SMOKE[@]}" --agents "$K" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$K" > "$LOG" 2>&1
  rc=$?
  say "part $K/$S rc=$rc"
  if grep -q "partial checkpoint load" "$LOG"; then
    say "WARNING part $K/$S: partial checkpoint load (see $LOG)"
  fi
  return $rc
}

bench() {  # $1 label  $2.. harness keys: 5 scenarios each, longest first
  local L=$1 K S n=0 total=0
  shift
  wait_live || { say "$L benchmarks not started (GPU 1 gate)"; return 1; }
  for S in $SCENARIOS; do
    for K in "$@"; do
      with_slot part "$K" "$S" &
      sleep 2
    done
  done
  wait
  for S in $SCENARIOS; do
    for K in "$@"; do
      total=$((total + 1))
      if [ -s "$PARTS/$K/$S/episodes.csv" ]; then n=$((n + 1)); else say "MISSING part $K/$S"; fi
    done
  done
  if [ "$n" -eq "$total" ]; then
    say "$L BENCH DONE ($n/$total parts)"
    return 0
  fi
  say "$L BENCH INCOMPLETE ($n/$total parts)"
  return 1
}

rung() {  # $1 rung  $2 gpu for BC + training
  local R=$1 G=$2 RC LAST F MSG
  local CFG=run_configs/agents/set_transformer_v6_ladder_$R.json
  local BC=checkpoints/bc_topsis_v6_$R/set_transformer_bc.pt
  local DIR=checkpoints/ladder_$R FINAL=checkpoints/ladder_${R}_final
  local TLOG=reports/train_ladder_$R.log BLOG=reports/ladder_bc_$R.log

  # (a) BC with the rung's own network, hc_v6's BC arguments
  if [ -s "$BC" ]; then
    say "$R BC cached"
  else
    fits "$BC_H" || { say "$R BC skipped: would pass $STOP_UTC"; return 1; }
    say "$R BC start (GPU ${G:-cpu})"
    CUDA_VISIBLE_DEVICES=$G nice -n 5 "$PY" scripts/warmstart_bc.py \
      --env-config run_configs/benchmark_suite/train_multiscale_v5.json \
      --agent-config "$CFG" "${BC_ARGS[@]}" --out "$BC" > "$BLOG" 2>&1
    say "$R BC rc=$?"
  fi
  [ -s "$BC" ] || { say "ABORT $R: $BC missing (see $BLOG)"; return 1; }
  MSG=$(check_flags "$BC" "$(flags_of "$R")") \
    || { say "ABORT $R: BC checkpoint $MSG"; return 1; }
  say "$R BC checkpoint $MSG"
  say "LADDER $R BC DONE ($(grep -o 'done: val agreement [0-9.]*%' "$BLOG" 2>/dev/null | tail -1))"

  # (b) training, hc_perm recipe
  if grep -qs "^DONE_TRAIN_LADDER_$R rc=0" "$TLOG"; then
    say "$R training cached"
  else
    fits "$TRAIN_H" || { say "$R training skipped: would pass $STOP_UTC"; return 1; }
    say "$R training start (GPU ${G:-cpu}, ${TRAIN_ARGS[*]})"
    CUDA_VISIBLE_DEVICES=$G nice -n 5 "$PY" scripts/train_hydra.py \
      env=train_multiscale_v5 agent=set_transformer_v6_ladder_$R \
      "${TRAIN_ARGS[@]}" init_checkpoint="$BC" checkpoint_dir="$DIR" \
      hydra.run.dir="outputs/ladder_$R" >> "$TLOG" 2>&1
    RC=$?
    echo "DONE_TRAIN_LADDER_$R rc=$RC $(date -u +%FT%TZ)" >> "$TLOG"
    say "LADDER $R TRAIN DONE rc=$RC"
    [ "$RC" = 0 ] || { say "ABORT $R: training failed (see $TLOG)"; return 1; }
  fi
  if grep -q "Partial checkpoint load" "$TLOG"; then
    say "WARNING $R: the BC start loaded partially (see $TLOG)"
  fi

  # (c) canonicalise best + last (final.pt); copy then rename, so a
  # benchmark process never reads a half-written file
  mkdir -p "$FINAL"
  LAST="$DIR/set_transformer_final.pt"
  if [ ! -s "$LAST" ]; then
    LAST=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
    say "WARNING $R: no final.pt, last = ${LAST:-none}"
  fi
  [ -s "$DIR/set_transformer_best.pt" ] && [ -n "$LAST" ] \
    || { say "ABORT $R: best or last checkpoint missing in $DIR"; return 1; }
  for F in best last; do
    if [ ! -s "$FINAL/set_transformer_$F.pt" ]; then
      if [ "$F" = best ]; then cp "$DIR/set_transformer_best.pt" "$FINAL/.tmp_$F"
      else cp "$LAST" "$FINAL/.tmp_$F"; fi
      mv "$FINAL/.tmp_$F" "$FINAL/set_transformer_$F.pt"
    fi
    MSG=$(check_flags "$FINAL/set_transformer_$F.pt" "$(flags_of "$R")") \
      || { say "ABORT $R: $F checkpoint $MSG"; return 1; }
  done
  say "LADDER $R CANON DONE (best + last=$(basename "$LAST"))"

  # (d) benchmarks
  bench "LADDER $R" "ladder_$R" "ladder_${R}_last"
}

perm_lane() {  # top rung: trained in ~/kata_perm, benchmarked here
  local PLOG="$PERM_ROOT/reports/train_hc_perm.log" PQ="$PERM_ROOT/reports/hc_perm_queue.log"
  local SRC="$PERM_ROOT/checkpoints/hc_perm_final" F MSG
  until grep -qs "DONE_TRAIN_HCPERM" "$PLOG"; do
    past_stop && { say "hc_perm: no DONE_TRAIN_HCPERM before $STOP_UTC"; return 1; }
    sleep 300
  done
  grep "DONE_TRAIN_HCPERM" "$PLOG" | tail -1 | grep -q "rc=0" \
    || { say "ABORT hc_perm lane: $(grep DONE_TRAIN_HCPERM "$PLOG" | tail -1)"; return 1; }
  until grep -qs "HC PERM QUEUE DONE" "$PQ"; do
    past_stop && { say "hc_perm: no HC PERM QUEUE DONE before $STOP_UTC"; return 1; }
    sleep 60
  done
  for F in best last; do
    [ -s "$SRC/set_transformer_$F.pt" ] || { say "ABORT hc_perm lane: $SRC/set_transformer_$F.pt missing"; return 1; }
    MSG=$(check_flags "$SRC/set_transformer_$F.pt" "$(flags_of hc_perm)") \
      || { say "ABORT hc_perm lane: $F checkpoint $MSG"; return 1; }
  done
  ln -sfn "$SRC" checkpoints/hc_perm_dgy_final
  say "HC PERM DGY LINKED (checkpoints/hc_perm_dgy_final -> $SRC)"
  bench "HC PERM DGY" hc_perm_dgy hc_perm_dgy_last
}

say "ARCH LADDER QUEUE ARMED (pid $$, code $(cat COMMIT 2>/dev/null || echo unknown), smoke=${SMOKE:-0}, stop $STOP_UTC)"
say "placement: flat + pool on GPU ${GPU_TRAIN:-cpu}; plainset + benchmarks on GPU ${GPU_LATE:-cpu} after LIVE LANE DONE; <= $MAX_BENCH benchmark parts at once"

# Each lane runs in the background; its exit status is collected below.
rung flat "$GPU_TRAIN" &
PID_flat=$!
sleep 5
rung pool "$GPU_TRAIN" &
PID_pool=$!
(
  say "plainset: waiting for LIVE LANE DONE in $LIVE_LOG"
  wait_live && { say "plainset: GPU 1 gate open"; rung plainset "$GPU_LATE"; }
) &
PID_plainset=$!
perm_lane &
PID_hc_perm=$!

FAILED=""
for LANE in flat pool plainset hc_perm; do
  PID_VAR="PID_$LANE"
  wait "${!PID_VAR}"
  RC=$?
  say "lane $LANE rc=$RC"
  [ "$RC" -eq 0 ] || FAILED="$FAILED $LANE"
done
if [ -n "$FAILED" ]; then
  say "ARCH LADDER QUEUE FAILED (lanes:$FAILED)"
  exit 1
fi
say "ARCH LADDER QUEUE DONE"
