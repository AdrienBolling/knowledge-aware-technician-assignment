#!/usr/bin/env bash
# Chain 4 of the v4-world generation: train hc_v6 and merge it into
# reports/hvp_eval_v4, then run the NEW lifecycle scenario for the
# whole agent roster.
#
# v6 = v5 objective/world UNCHANGED (env=train_multiscale_v5) with the
# 2026-08-04 fix package as the only variables:
#   * D1  — LR schedule sized from measured decision density
#           (tu_per_decision=24) + floored tail + extension re-arm.
#   * D2  — dropout 0.0 (agent config) + eval-mode deterministic acting.
#   * D3  — role-bound slot fusion + feature-context view
#           (agent config: slot_role_binding, use_feature_context).
#   * D11 — boolean obs tokens now match the frozen vocab (BUSY/BROKEN/
#           DISRUPT/... were <UNK> for every earlier agent).
#   * D12 — machine BD_COUNT/DOWNTIME/MEAN_TBF now read real values.
# The architecture change invalidates bc_topsis_v4 -> fresh BC collect.
#
#   1. Collect TOPSIS BC with the v6 architecture -> bc_topsis_v6.
#   2. Train 600 eps, vec10, horizons U(200k,350k), seed 42.
#   3. Plateau gate, up to 3 x 200-ep extensions (extensions now train
#      at a live LR thanks to the D1 re-arm).
#   4. Canonicalise best+last -> checkpoints/hc_v6_final; benchmark
#      hc_v6 + hc_v6_last at all standard scales into the generation.
#   5. Lifecycle scenario (5M + fleet/park mutations) for the FULL
#      roster (learned + baselines), parallel per-agent parts.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
Q=reports/v6_train_queue.log
SEED=20260722
OUTROOT=reports/hvp_eval_v4
PARTS=reports/hvp_v6_vl_parts
LC_PARTS=reports/hvp_lifecycle_parts
say() { echo "$(date -u +%FT%TZ) [v6t] $*" >> "$Q"; }
# End-of-run checkpoint naming depends on the loop: the vec loop writes
# round*.pt + final.pt, the serial loop (parallel_envs=1) ep*.pt +
# final.pt.
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

say "V6 TRAIN QUEUE ARMED (pid $$)"

# ---------- 1. BC warm-start with the v6 architecture ----------
if [ ! -f checkpoints/bc_topsis_v6/set_transformer_bc.pt ]; then
  say "BC collection start (v6 architecture)"
  nice -n 10 uv run python scripts/warmstart_bc.py \
    --env-config run_configs/benchmark_suite/train_multiscale_v5.json \
    --agent-config run_configs/agents/set_transformer_v6.json \
    --episodes 25 --sim-time 200000 --seed 7 \
    --out checkpoints/bc_topsis_v6/set_transformer_bc.pt \
    >> reports/v6_bc_collect.log 2>&1
  say "BC collection rc=$?"
fi
if [ ! -f checkpoints/bc_topsis_v6/set_transformer_bc.pt ]; then
  say "ABORT: BC checkpoint missing"
  exit 1
fi

# ---------- 2. Train v6 (+ plateau extensions) ----------
train_v6() {  # $1 episodes, $2 init ckpt, $3 ckpt dir
  nice -n 10 uv run python scripts/train_hydra.py \
    env=train_multiscale_v5 agent=set_transformer_v6 \
    episodes="$1" parallel_envs=10 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 checkpoint_interval=50 seed=42 \
    init_checkpoint="$2" checkpoint_dir="$3" \
    >> reports/train_hc_v6.log 2>&1
}

DIR=checkpoints/hc_v6
say "v6 training start (600 eps)"
train_v6 600 checkpoints/bc_topsis_v6/set_transformer_bc.pt "$DIR"
RC=$?
say "v6 training rc=$RC"
echo "DONE_TRAIN_V6 rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v6.log
if [ "$RC" != "0" ]; then say "ABORT: training failed"; exit 1; fi

PL=0
for EXT in 1 2 3; do
  uv run python scripts/check_plateau.py \
    --log reports/train_hc_v6.log >> "$Q" 2>&1
  PL=$?
  if [ "$PL" = "0" ]; then say "plateau confirmed"; break; fi
  if [ "$PL" = "2" ]; then say "ALERT plateau indeterminate — proceeding"; break; fi
  LAST_CKPT=$(pick_last "$DIR")
  NEWDIR="checkpoints/hc_v6_ext${EXT}"
  say "still improving — extension #$EXT (+200 eps) from $LAST_CKPT"
  train_v6 200 "$LAST_CKPT" "$NEWDIR"
  RC=$?
  say "extension #$EXT rc=$RC"
  echo "DONE_TRAIN_V6_EXT${EXT} rc=$RC $(date -u +%FT%TZ)" >> reports/train_hc_v6.log
  if [ "$RC" != "0" ]; then say "ALERT extension failed — using what we have"; break; fi
  DIR="$NEWDIR"
done

# ---------- 3. Canonicalise ----------
mkdir -p checkpoints/hc_v6_final
BEST="$DIR/set_transformer_best.pt"
[ -f "$BEST" ] || BEST=checkpoints/hc_v6/set_transformer_best.pt
LAST_CKPT=$(pick_last "$DIR")
[ -f "$LAST_CKPT" ] || LAST_CKPT="$BEST"
cp "$BEST" checkpoints/hc_v6_final/set_transformer_best.pt
cp "$LAST_CKPT" checkpoints/hc_v6_final/set_transformer_last.pt
say "final ckpts: best=$BEST last=$LAST_CKPT"

# ---------- 4. Benchmark v6 into the generation ----------
vl() {
  local A=$1
  if [ -s "$PARTS/$A/very_long/episodes.csv" ]; then
    say "vl $A cached (part exists)"
    return
  fi
  say "vl $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "$PARTS/$A" \
    >> "reports/v6_eval_vl_${A}.log" 2>&1
  say "vl $A rc=$?"
}
vl hc_v6 &
vl hc_v6_last &

for S in "baseline" "small_scale" "massive_scale"; do
  EXTRA=""
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  say "std $S merge start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents hc_v6,hc_v6_last \
    --eval-seed "$SEED" --merge --out-root "$OUTROOT" \
    >> reports/v6_eval_std_hcv6.log 2>&1
  say "std $S merge rc=$?"
done
wait
nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts "$PARTS" --dest "$OUTROOT/very_long" \
  >> reports/v6_eval_merge.log 2>&1
say "vl merge rc=$?"

# ---------- 5. Lifecycle scenario: full roster, per-agent parts ----------
# 5M horizon + mid-episode fleet/park mutations; every agent faces the
# identical schedule (same eval seed -> same layout, same targets).
# Parts are idempotent (skipped if episodes.csv exists), max 4 at once.
LC_AGENTS="hc_v6 hc_v6_last hc_v5 hc_v5_last hc_v4 hc_v4_last hc_v3 hc_v3_last gaefix human topsis empirical_topsis empirical_spt shortest_processing optimal_assignment batch_milp greedy_reward shortest_queue least_busy least_fatigued round_robin random train_weakest reserve_specialist"
lc() {
  local A=$1
  if [ -s "$LC_PARTS/$A/lifecycle/episodes.csv" ]; then
    say "lifecycle $A cached (part exists)"
    return
  fi
  say "lifecycle $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario lifecycle --agents "$A" --eval-seed "$SEED" \
    --record-every 200 --out-root "$LC_PARTS/$A" \
    >> "reports/v6_eval_lc_${A}.log" 2>&1
  say "lifecycle $A rc=$?"
}
NJOBS=0
for A in $LC_AGENTS; do
  lc "$A" &
  NJOBS=$((NJOBS + 1))
  if [ $((NJOBS % 4)) = 0 ]; then wait; fi
done
wait
nice -n 10 uv run python scripts/merge_hvp_parts.py \
  --parts "$LC_PARTS" --dest "$OUTROOT/lifecycle" --scenario lifecycle \
  >> reports/v6_eval_merge.log 2>&1
say "lifecycle merge rc=$?"

nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root "$OUTROOT" >> reports/v6_eval_merge.log 2>&1
say "analysis rc=$?"
say "V6 TRAIN QUEUE DONE"
