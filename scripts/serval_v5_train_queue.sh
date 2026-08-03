#!/usr/bin/env bash
# Chain 3 of the v4-world generation: train hc_v5 and merge it into
# reports/hvp_eval_v4.
#
#   1. Reuse the v4-world TOPSIS behaviour-cloning warm-start
#      (checkpoints/bc_topsis_v4 — same world, so no re-collection).
#   2. Train v5 with the exact v4 recipe EXCEPT the objective:
#      env=train_multiscale_v5 (PBRS knowledge credit, workload_balance
#      0.5, fleet_availability 0.5) + semi-MDP gamma**dt discounting
#      (gamma 0.9999 per sim-t.u., conf/train.yaml default).  The
#      reward/discounting package is the only variable vs hc_v4.
#   3. Plateau gate with up to 3 x 200-episode extensions (same
#      protocol as v3/v4).
#   4. Canonicalise best+last -> checkpoints/hc_v5_final and benchmark
#      hc_v5 + hc_v5_last at all scales into the v4 generation.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
Q=reports/v5_train_queue.log
SEED=20260722
OUTROOT=reports/hvp_eval_v4
PARTS=reports/hvp_v5_vl_parts
say() { echo "$(date -u +%FT%TZ) [v5t] $*" >> "$Q"; }

say "V5 TRAIN QUEUE ARMED (pid $$)"

# ---------- 1. BC warm-start (shared with v4 — same world) ----------
if [ ! -f checkpoints/bc_topsis_v4/set_transformer_bc.pt ]; then
  say "ABORT: BC checkpoint missing (expected checkpoints/bc_topsis_v4)"
  exit 1
fi

# ---------- 2. Train v5 (+ plateau extensions) ----------
train_v5() {  # $1 episodes, $2 init ckpt, $3 ckpt dir
  nice -n 10 uv run python scripts/train_hydra.py \
    env=train_multiscale_v5 episodes="$1" parallel_envs=10 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 checkpoint_interval=50 seed=42 \
    init_checkpoint="$2" checkpoint_dir="$3" \
    >> reports/train_hc_v5.log 2>&1
}

DIR=checkpoints/hc_v5
say "v5 training start (600 eps)"
train_v5 600 checkpoints/bc_topsis_v4/set_transformer_bc.pt "$DIR"
RC=$?
say "v5 training rc=$RC"
echo "DONE_TRAIN_V5 rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v5.log
if [ "$RC" != "0" ]; then say "ABORT: training failed"; exit 1; fi

PL=0
for EXT in 1 2 3; do
  uv run python scripts/check_plateau.py \
    --log reports/train_hc_v5.log >> "$Q" 2>&1
  PL=$?
  if [ "$PL" = "0" ]; then say "plateau confirmed"; break; fi
  if [ "$PL" = "2" ]; then say "ALERT plateau indeterminate — proceeding"; break; fi
  LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$LAST_CKPT" ] && LAST_CKPT="$DIR/set_transformer_best.pt"
  NEWDIR="checkpoints/hc_v5_ext${EXT}"
  say "still improving — extension #$EXT (+200 eps) from $LAST_CKPT"
  train_v5 200 "$LAST_CKPT" "$NEWDIR"
  RC=$?
  say "extension #$EXT rc=$RC"
  echo "DONE_TRAIN_V5_EXT${EXT} rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v5.log
  if [ "$RC" != "0" ]; then say "ALERT extension failed — using what we have"; break; fi
  DIR="$NEWDIR"
done

# ---------- 3. Canonicalise ----------
mkdir -p checkpoints/hc_v5_final
BEST="$DIR/set_transformer_best.pt"
[ -f "$BEST" ] || BEST=checkpoints/hc_v5/set_transformer_best.pt
LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
[ -z "$LAST_CKPT" ] && LAST_CKPT="$BEST"
cp "$BEST" checkpoints/hc_v5_final/set_transformer_best.pt
cp "$LAST_CKPT" checkpoints/hc_v5_final/set_transformer_last.pt
say "final ckpts: best=$BEST last=$LAST_CKPT"

# ---------- 4. Benchmark v5 into the generation ----------
vl() {
  local A=$1
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "$PARTS/$A" \
    >> "reports/v5_eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
vl hc_v5 &
vl hc_v5_last &

for S in "baseline" "small_scale" "massive_scale"; do
  EXTRA=""
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  say "std $S merge start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents hc_v5,hc_v5_last \
    --eval-seed "$SEED" --merge --out-root "$OUTROOT" \
    >> reports/v5_eval_std_hcv5.log 2>&1
  say "std $S merge rc=$?"
done

wait
nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts "$PARTS" --dest "$OUTROOT/very_long" \
  >> reports/v5_eval_merge.log 2>&1
say "vl merge rc=$?"
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v5_eval_merge.log 2>&1
say "analysis rc=$?"
say "V5 TRAIN QUEUE DONE"
