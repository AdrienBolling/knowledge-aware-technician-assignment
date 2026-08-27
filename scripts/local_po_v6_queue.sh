#!/usr/bin/env bash
# po_v6 queue on the LOCAL machine (gourmetdesk, RTX 4080) — the
# production-only ablation of hc_v6.
#
# Purpose (user request 2026-08-27): train an agent IDENTICAL to the
# best HTT-RL (hc_v6 recipe: v6 architecture, corrected world,
# train_multiscale_v5 curriculum, BC warm-start, 600 eps vec5, seed
# 42) but with ONLY performance-oriented reward components — the
# human-centric stack (knowledge_increment PBRS, terminal fleet
# knowledge, fatigue_cost, workload_balance, fleet_availability) is
# disabled in run_configs/benchmark_suite/train_multiscale_v5_po.json;
# throughput_delta, repair_quality, terminal_finished_products (and
# the inert busy_technician) remain.  Goal: showcase what the
# human-centric objectives buy over long horizons.
#
# Caveat recorded for the writeup: dgy (and its
# checkpoints/bc_topsis_v6) is unreachable, so the BC warm-start is
# RE-COLLECTED here with the exact dgy protocol and seed (25 eps,
# sim-time 200000, seed 7, v6 architecture).  Collection data is
# seed-deterministic; the fitted BC weights can differ from dgy's file
# at GPU-kernel-nondeterminism level.
#
# After training: canonicalise best+last, benchmark po_v6/po_v6_last
# on all 5 scenarios into the idempotent parts tree, merge into
# reports/hvp_eval_v6w, re-analyze.  Scenarios ordered longest-first
# so both 5M episodes run in the same concurrency batch.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
Q=reports/po_v6_queue.log
BC=checkpoints/bc_topsis_v6_local/set_transformer_bc.pt
SEED=42
EVAL_SEED=20260722
OUTROOT=reports/hvp_eval_v6w
PARTS=reports/hvp_v6w_parts
KEYS="po_v6 po_v6_last"
SCENARIOS="very_long lifecycle massive_scale baseline small_scale"
say() { echo "$(date -u +%FT%TZ) [po-v6-local] $*" | tee -a "$Q"; }
mkdir -p reports checkpoints

say "PO V6 LOCAL QUEUE ARMED (pid $$, seed $SEED, env=train_multiscale_v5_po)"

# ---------- 1. BC warm-start (dgy protocol, local re-collect) ----------
if [ ! -f "$BC" ]; then
  say "BC collection start (v6 architecture, seed 7 — dgy protocol)"
  nice -n 5 uv run python scripts/warmstart_bc.py \
    --env-config run_configs/benchmark_suite/train_multiscale_v5.json \
    --agent-config run_configs/agents/set_transformer_v6.json \
    --episodes 25 --sim-time 200000 --seed 7 \
    --out "$BC" \
    >> reports/po_v6_bc_collect.log 2>&1
  say "BC collection rc=$?"
fi
if [ ! -f "$BC" ]; then
  say "ABORT: BC checkpoint missing (see reports/po_v6_bc_collect.log)"
  exit 1
fi

# ---------- 2. Train (exact hc_v6 recipe, PO reward) ----------
CKDIR=checkpoints/po_v6
if [ -f "$CKDIR/set_transformer_final.pt" ]; then
  say "train cached (final.pt exists)"
else
  say "po_v6 training start (600 eps, parallel_envs=5)"
  nice -n 5 uv run python scripts/train_hydra.py \
    env=train_multiscale_v5_po agent=set_transformer_v6 \
    episodes=600 parallel_envs=5 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=200 checkpoint_interval=50 seed=$SEED \
    init_checkpoint="$BC" \
    checkpoint_dir="$CKDIR" \
    > reports/train_po_v6.log 2>&1
  RC=$?
  say "training rc=$RC"
  echo "DONE_TRAIN_PO_V6 rc=$RC $(date -u +%FT%TZ)" >> reports/train_po_v6.log
fi

# ---------- 3. Canonicalise best + last ----------
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
mkdir -p checkpoints/po_v6_final
cp "$CKDIR/set_transformer_best.pt" \
   checkpoints/po_v6_final/set_transformer_best.pt
LAST=$(pick_last "$CKDIR")
cp "$LAST" checkpoints/po_v6_final/set_transformer_last.pt
say "canonicalised: best + last=$(basename "$LAST")"
say "PO V6 TRAIN DONE"

# ---------- 4. Benchmark parts (idempotent) ----------
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
say "all po_v6 parts done"

# ---------- 5. Merge + analyze ----------
for S in $SCENARIOS; do
  nice -n 10 uv run python scripts/merge_hvp_parts.py \
    --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
    >> reports/po_v6_merge.log 2>&1
  say "merge $S rc=$?"
done
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/po_v6_merge.log 2>&1
say "analysis rc=$?"
say "PO V6 QUEUE DONE"
