#!/usr/bin/env bash
# Chain 2 of the v4 generation: train hc_v4 from scratch in the
# mechanisms-ON world and merge it into reports/hvp_eval_v4.
#
#   1. Re-collect the TOPSIS behaviour-cloning warm-start under the new
#      world dynamics (the SOM-world BC data predates decay/Kijima).
#   2. Train v4 with the exact v3 recipe (600 eps, vec10, horizons
#      U(200k, 350k), HC-v1 legacy reward, gamma 0.997 / lambda 0.98,
#      PopArt, no GRU, seed 42) — the world flip is the only variable.
#   3. Plateau gate with up to 3 x 200-episode extensions (same
#      protocol as v3).
#   4. Canonicalise best+last -> checkpoints/hc_v4_final and benchmark
#      hc_v4 + hc_v4_last at all scales into the v4 generation.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
Q=reports/v4_train_queue.log
SEED=20260722
OUTROOT=reports/hvp_eval_v4
PARTS=reports/hvp_v4_vl_parts
say() { echo "$(date -u +%FT%TZ) [v4t] $*" >> "$Q"; }

say "V4 TRAIN QUEUE ARMED (pid $$)"

# ---------- 1. BC warm-start in the new world ----------
if [ ! -f checkpoints/bc_topsis_v4/set_transformer_bc.pt ]; then
  say "BC collection start"
  nice -n 10 uv run python scripts/warmstart_bc.py \
    --env-config run_configs/benchmark_suite/train_multiscale.json \
    --episodes 25 --sim-time 200000 --seed 7 \
    --out checkpoints/bc_topsis_v4/set_transformer_bc.pt \
    >> reports/v4_bc_collect.log 2>&1
  say "BC collection rc=$?"
fi
if [ ! -f checkpoints/bc_topsis_v4/set_transformer_bc.pt ]; then
  say "ABORT: BC checkpoint missing"
  exit 1
fi

# ---------- 2. Train v4 (+ plateau extensions) ----------
train_v4() {  # $1 episodes, $2 init ckpt, $3 ckpt dir
  nice -n 10 uv run python scripts/train_hydra.py \
    env=train_multiscale episodes="$1" parallel_envs=10 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 checkpoint_interval=50 seed=42 \
    init_checkpoint="$2" checkpoint_dir="$3" \
    >> reports/train_hc_v4.log 2>&1
}

DIR=checkpoints/hc_v4
say "v4 training start (600 eps)"
train_v4 600 checkpoints/bc_topsis_v4/set_transformer_bc.pt "$DIR"
RC=$?
say "v4 training rc=$RC"
echo "DONE_TRAIN_V4 rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v4.log
if [ "$RC" != "0" ]; then say "ABORT: training failed"; exit 1; fi

PL=0
for EXT in 1 2 3; do
  uv run python scripts/check_plateau.py \
    --log reports/train_hc_v4.log >> "$Q" 2>&1
  PL=$?
  if [ "$PL" = "0" ]; then say "plateau confirmed"; break; fi
  if [ "$PL" = "2" ]; then say "ALERT plateau indeterminate — proceeding"; break; fi
  LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$LAST_CKPT" ] && LAST_CKPT="$DIR/set_transformer_best.pt"
  NEWDIR="checkpoints/hc_v4_ext${EXT}"
  say "still improving — extension #$EXT (+200 eps) from $LAST_CKPT"
  train_v4 200 "$LAST_CKPT" "$NEWDIR"
  RC=$?
  say "extension #$EXT rc=$RC"
  echo "DONE_TRAIN_V4_EXT${EXT} rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v4.log
  if [ "$RC" != "0" ]; then say "ALERT extension failed — using what we have"; break; fi
  DIR="$NEWDIR"
done

# ---------- 3. Canonicalise ----------
mkdir -p checkpoints/hc_v4_final
BEST="$DIR/set_transformer_best.pt"
[ -f "$BEST" ] || BEST=checkpoints/hc_v4/set_transformer_best.pt
LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
[ -z "$LAST_CKPT" ] && LAST_CKPT="$BEST"
cp "$BEST" checkpoints/hc_v4_final/set_transformer_best.pt
cp "$LAST_CKPT" checkpoints/hc_v4_final/set_transformer_last.pt
say "final ckpts: best=$BEST last=$LAST_CKPT"

# ---------- 4. Benchmark v4 into the generation ----------
vl() {
  local A=$1
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "$PARTS/$A" \
    >> "reports/v4_eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
vl hc_v4 &
vl hc_v4_last &

for S in "baseline" "small_scale" "massive_scale"; do
  EXTRA=""
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  say "std $S merge start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents hc_v4,hc_v4_last \
    --eval-seed "$SEED" --merge --out-root "$OUTROOT" \
    >> reports/v4_eval_std_hcv4.log 2>&1
  say "std $S merge rc=$?"
done

wait
nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts "$PARTS" --dest "$OUTROOT/very_long" \
  >> reports/v4_eval_merge.log 2>&1
say "vl merge rc=$?"
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v4_eval_merge.log 2>&1
say "analysis rc=$?"
say "V4 TRAIN QUEUE DONE"
