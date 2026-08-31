#!/usr/bin/env bash
# hc_v6_ext queue on the LOCAL machine (gourmetdesk, RTX 4080) — the
# training-budget extension of hc_v6.
#
# Purpose (user request 2026-08-31): take the best HTT weights we have
# (hc_v6, 600 eps) and train 600 MORE episodes with the normal
# human-centric reward set (env=train_multiscale_v5, unchanged), to
# see whether performance still improves past the ~ep-400 plateau.
#
# Deliberate choices (documented for the read):
#   - init = checkpoints/hc_v6/set_transformer_final.pt (the ep-600
#     end state, continuation semantics; the noisy-eval best.pt at
#     ep-401 would discard 199 episodes of training).
#   - seed 43 (NOT 42): seed 42 would REPLAY hc_v6's exact training
#     world sequence; 43 gives fresh draws — these are episodes
#     601-1200 of a notional longer run, not a repeat of 1-600.
#   - agent.params.lr=1e-4: the D1 fix re-arms an exhausted cosine
#     schedule at the CONSTRUCTOR lr on load, so this runs a warm
#     restart at 1/3 of the original 3e-4 peak — hot enough to move a
#     plateaued policy, cool enough not to blow it up (KL early stop
#     + clip guard the rest).
#
# After training: canonicalise best+last, benchmark hc_v6_ext/
# hc_v6_ext_last on all 5 scenarios (longest first) into the
# idempotent parts tree, merge into reports/hvp_eval_v6w, re-analyze.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/hc_v6_ext_queue.log
INIT=checkpoints/hc_v6/set_transformer_final.pt
SEED=43
EVAL_SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
KEYS="hc_v6_ext hc_v6_ext_last"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
say() { echo "$(date -u +%FT%TZ) [hc-v6-ext] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints

[ -f "$INIT" ] || { say "ABORT: init checkpoint $INIT missing"; exit 1; }
say "HC V6 EXT QUEUE ARMED (pid $$, seed $SEED, lr 1e-4, +600 eps from hc_v6 final)"

# ---------- 1. Train (hc_v6 recipe, 600 more eps) ----------
CKDIR=checkpoints/hc_v6_ext
if [ -f "$CKDIR/set_transformer_final.pt" ]; then
  say "train cached (final.pt exists)"
else
  say "hc_v6_ext training start (600 eps, parallel_envs=5)"
  nice -n 5 uv run python scripts/train_hydra.py \
    env=train_multiscale_v5 agent=set_transformer_v6 \
    episodes=600 parallel_envs=5 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=200 checkpoint_interval=50 seed=$SEED \
    init_checkpoint="$INIT" \
    checkpoint_dir="$CKDIR" \
    agent.params.lr=1e-4 \
    > reports/train_hc_v6_ext.log 2>&1
  RC=$?
  say "training rc=$RC"
  echo "DONE_TRAIN_HC_V6_EXT rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v6_ext.log
fi

# ---------- 2. Canonicalise best + last ----------
pick_last() {  # $1 = checkpoint dir
  local d=$1 p
  if [ -f "$d/set_transformer_final.pt" ]; then
    echo "$d/set_transformer_final.pt"
    return
  fi
  p=$(ls -1 "$d"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p=$(ls -1 "$d"/set_transformer_ep*.pt 2>/dev/null | sort | tail -1)
  [ -z "$p" ] && p="$d/set_transformer_best.pt"
  echo "$p"
}
if [ ! -f "$CKDIR/set_transformer_best.pt" ] && [ ! -f "$CKDIR/set_transformer_final.pt" ]; then
  say "ABORT: no checkpoints in $CKDIR (training failed)"
  exit 1
fi
mkdir -p checkpoints/hc_v6_ext_final
cp "$CKDIR/set_transformer_best.pt" \
   checkpoints/hc_v6_ext_final/set_transformer_best.pt
LAST=$(pick_last "$CKDIR")
cp "$LAST" checkpoints/hc_v6_ext_final/set_transformer_last.pt
say "canonicalised: best + last=$(basename "$LAST")"
say "HC V6 EXT TRAIN DONE"

# ---------- 3. Benchmark parts (idempotent) ----------
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
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
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
say "all hc_v6_ext parts done"

# ---------- 4. Merge + analyze ----------
for S in $SCENARIOS; do
  nice -n 10 uv run python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/hc_v6_ext_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/hc_v6_ext_merge.log 2>&1
say "analysis rc=$?"
say "HC V6 EXT QUEUE DONE"
