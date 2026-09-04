#!/usr/bin/env bash
# hc_v6_ext benchmark parts on dgy (GPU 1) — moved off the local 4080
# (user request 2026-09-03: "not on local computer but on serval dgy").
#
# Training + canonicalisation already happened locally on 2026-08-31
# (checkpoints/hc_v6_ext_final/{best,last} shipped here by rsync, md5
# verified).  This runs ONLY the 10 benchmark parts (hc_v6_ext and
# hc_v6_ext_last x 5 scenarios, longest first) into the idempotent
# parts tree reports/hvp_v6w_parts/<agent>/<scenario>.
#
# NO merge here: the parts are pulled back to the desktop and merged
# into the canonical reports/hvp_eval_v6w there (dgy's copy of that
# tree lacks the po_v6 rows, so it is not the reference any more).
# Note the parts carry the newer per-type-disruption / machine-rate
# columns; merge tolerates that via NaN for older rows.
#
# uv --no-sync everywhere (deliberate torch downgrade on dgy).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
GPU="${CUDA_VISIBLE_DEVICES:-1}"
Q=reports/hc_v6_ext_dgy_queue.log
EVAL_SEED=20260722
PARTS=reports/hvp_v6w_parts
KEYS="hc_v6_ext hc_v6_ext_last"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
say() { echo "$(date -u +%FT%TZ) [hc-v6-ext-dgy] $*" | tee -a "$Q"; }
mkdir -p reports "$PARTS"

for f in checkpoints/hc_v6_ext_final/set_transformer_best.pt \
         checkpoints/hc_v6_ext_final/set_transformer_last.pt; do
  [ -f "$f" ] || { say "ABORT: checkpoint $f missing"; exit 1; }
done
say "HC V6 EXT DGY PARTS ARMED (pid $$, GPU $GPU, 2 keys x 5 scenarios)"

part() {  # $1 agent, $2 scenario
  local A=$1 S=$2 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then
    say "part $A/$S cached"
    return
  fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "part $A/$S start (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 uv run --no-sync python \
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
    if [ $((NJOBS % 4)) = 0 ]; then wait; fi
  done
done
wait
for S in $SCENARIOS; do
  for A in $KEYS; do
    [ -s "$PARTS/$A/$S/episodes.csv" ] || say "MISSING part $A/$S (see reports/v6w_part_${A}_${S}.log)"
  done
done
say "HC V6 EXT DGY PARTS DONE"
