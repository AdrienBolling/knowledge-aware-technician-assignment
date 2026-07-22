#!/usr/bin/env bash
# Post-v3 unattended queue (runs ON serval; survives VPN loss / local sleep).
#
# Stage 0  wait for DONE_TRAIN_V3 in reports/train_hc_v3.log; abort on rc!=0
# Stage 0b plateau gate: while the per-decision train return is still
#          improving (scripts/check_plateau.py), extend training by 200
#          episodes from the latest round checkpoint (max 3 extensions),
#          then canonicalise best+last into checkpoints/hc_v3_final/
# Stage 1  standard ladder: baseline / small_scale / massive_scale,
#          all agents (14 heuristics + set-transformer ckpts incl. both
#          hc_v3 best AND last), fresh seed, out-root reports/hvp_eval_v3
# Stage 2  RL anchors trained on the go (sequential, GPU-polite):
#          ppo_transformer then rainbow_dqn on baseline.json (HC-v1 reward)
# Stage 3  anchors evaluated on the baseline scenario (their training
#          world — fixed action head + fleet-sized vocab cannot transfer)
# Stage 4  very-long 5M ladder, PER-AGENT invocations with --merge so each
#          agent's rows land incrementally (crash/restart resilient)
#
# All progress + rc codes append to reports/post_v3_queue.log; a heartbeat
# subshell logs liveness every 10 min. Nothing here ever kills processes.
set -u
cd "$(dirname "$0")/.."
Q=reports/post_v3_queue.log
RUNLOG=reports/hvp_eval_v3_run.log
SEED=20260722
say() { echo "$(date -u +%FT%TZ) $*" >> "$Q"; }

say "QUEUE ARMED (pid $$, seed $SEED)"

# Heartbeat: liveness + last line of the active run log every 10 min.
(
  while true; do
    sleep 600
    echo "$(date -u +%FT%TZ) [hb] $(tail -c 120 "$RUNLOG" 2>/dev/null | tr '\n' ' ')" >> "$Q"
  done
) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT

# ---------- Stage 0: gate on training completion ----------
while ! grep -q DONE_TRAIN_V3 reports/train_hc_v3.log 2>/dev/null; do
  sleep 300
done
RC=$(grep DONE_TRAIN_V3 reports/train_hc_v3.log | tail -1 | grep -oE "rc=[0-9]+" | cut -d= -f2)
say "STAGE0 v3 training finished rc=${RC:-?}"
if [ "${RC:-1}" != "0" ]; then
  say "ABORT: v3 rc=${RC:-?} — refusing to benchmark a failed training"
  exit 1
fi
if [ ! -f checkpoints/hc_v3_bc/set_transformer_best.pt ]; then
  say "ABORT: checkpoints/hc_v3_bc/set_transformer_best.pt missing"
  exit 1
fi

# ---------- Stage 0b: plateau gate + incremental extension ----------
DIR=checkpoints/hc_v3_bc
for EXT in 1 2 3; do
  uv run python scripts/check_plateau.py \
    --log reports/train_hc_v3.log >> "$Q" 2>&1
  PL=$?
  if [ "$PL" = "0" ]; then say "STAGE0b plateau confirmed"; break; fi
  if [ "$PL" = "2" ]; then
    say "STAGE0b ALERT: plateau indeterminate — proceeding to benchmarks"
    break
  fi
  LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
  [ -z "$LAST_CKPT" ] && LAST_CKPT="$DIR/set_transformer_best.pt"
  NEWDIR="checkpoints/hc_v3_bc_ext${EXT}"
  say "STAGE0b still improving — extension #$EXT (+200 eps) from $LAST_CKPT"
  nice -n 10 uv run python scripts/train_hydra.py \
    env=train_multiscale episodes=200 parallel_envs=10 \
    sim_time=275000 sim_time_min=200000 sim_time_max=350000 \
    eval_interval=50 \
    init_checkpoint="$LAST_CKPT" checkpoint_dir="$NEWDIR" \
    >> reports/train_hc_v3.log 2>&1
  RC=$?
  say "STAGE0b extension #$EXT rc=$RC"
  if [ "$RC" != "0" ]; then
    say "STAGE0b ALERT: extension failed — benchmarking what we have"
    break
  fi
  DIR="$NEWDIR"
done
[ "$EXT" = "3" ] && [ "$PL" = "1" ] && \
  say "STAGE0b extension cap reached (3) — proceeding"

# Canonicalise: the eval script benchmarks BOTH the best checkpoint and
# the latest round checkpoint (inline eval is sparse — 5 eps / 50
# rounds — so 'best' is a noisy selection).
mkdir -p checkpoints/hc_v3_final
BEST="$DIR/set_transformer_best.pt"
[ -f "$BEST" ] || BEST=checkpoints/hc_v3_bc/set_transformer_best.pt
LAST_CKPT=$(ls -1 "$DIR"/set_transformer_round*.pt 2>/dev/null | sort | tail -1)
[ -z "$LAST_CKPT" ] && LAST_CKPT="$BEST"
cp "$BEST" checkpoints/hc_v3_final/set_transformer_best.pt
cp "$LAST_CKPT" checkpoints/hc_v3_final/set_transformer_last.pt
say "STAGE0b final ckpts: best=$BEST last=$LAST_CKPT"

# ---------- Stage 1: standard ladder ----------
for S in baseline small_scale massive_scale; do
  EXTRA=""
  # industrial: 100k sim needs >10k decisions — keep the 25k non-binding
  # cap; 3 episodes instead of the historical 1 for spread.
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  say "STAGE1 $S start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents all \
    --eval-seed "$SEED" --out-root reports/hvp_eval_v3 \
    >> "$RUNLOG" 2>&1
  say "STAGE1 $S rc=$?"
done
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root reports/hvp_eval_v3 >> "$RUNLOG" 2>&1
say "STAGE1 analysis rc=$? (tables in reports/hvp_eval_v3)"

# ---------- Stage 2: RL anchors, trained on the go ----------
for A in ppo_transformer rainbow_dqn; do
  say "STAGE2 train $A start"
  nice -n 10 uv run python -m experiment.cli \
    --env run_configs/benchmark_suite/baseline.json \
    --agent "run_configs/agents/${A}.json" \
    --experiment run_configs/experiments/anchor_train.json \
    --exp-id "anchor_${A}" \
    >> "reports/train_anchor_${A}.log" 2>&1
  say "STAGE2 train $A rc=$?"
done

# ---------- Stage 3: anchors on their training world ----------
say "STAGE3 anchor eval start"
nice -n 10 uv run python scripts/eval_human_vs_performance.py \
  --scenario baseline --agents ppo_transformer,rainbow_dqn \
  --eval-seed "$SEED" --merge --out-root reports/hvp_eval_v3 \
  >> "$RUNLOG" 2>&1
say "STAGE3 rc=$?"
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root reports/hvp_eval_v3 >> "$RUNLOG" 2>&1

# ---------- Stage 4: 5M very-long ladder, one agent at a time ----------
HEUR="random round_robin least_busy least_fatigued shortest_queue \
shortest_processing optimal_assignment batch_milp topsis \
reserve_specialist greedy_reward train_weakest empirical_spt empirical_topsis"
for A in $HEUR human performance hc_v3 hc_v3_last; do
  say "STAGE4 very_long $A start"
  nice -n 10 uv run python scripts/eval_human_vs_performance.py \
    --scenario very_long --agents "$A" \
    --eval-seed "$SEED" --merge --record-every 200 \
    --out-root reports/hvp_eval_v3 >> "$RUNLOG" 2>&1
  say "STAGE4 $A rc=$?"
done
nice -n 10 uv run python scripts/analyze_hvp_results.py \
  --root reports/hvp_eval_v3 >> "$RUNLOG" 2>&1
say "QUEUE DONE"
