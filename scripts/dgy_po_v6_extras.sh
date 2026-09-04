#!/usr/bin/env bash
# po_v6 extras on dgy (GPU 2), 2026-09-03.
#
# (1) The lifecycle deep-dive part the ddlc queue reported MISSING:
#     po_v6 was trained on the desktop and its checkpoint had never
#     been shipped here ("unknown agent keys: ['po_v6']" was the
#     harness skipping a key with no checkpoint).  Now shipped
#     (checkpoints/po_v6_final, md5 verified) — same invocation as
#     scripts/dgy_deepdive_lifecycle.sh.
# (2) po_v6 rows for the disruption-instrumented tree: the 14-agent
#     disr roster omitted the PO-HTT-RL twin, but the per-type
#     decomposition of its lifecycle illness blowup (ill/1k 1,777 vs
#     hc_v6 1,435 in v6w) is exactly what that tree is for.  Parts go
#     to reports/hvp_disr_parts/po_v6/<scenario> and are merged into
#     reports/hvp_eval_disr (merge is agent-keyed + non-destructive,
#     so re-merging the 14 existing agents is a no-op).  Best ckpt
#     only, like the rest of the disr roster (best==last for po_v6).
#
# uv --no-sync everywhere (deliberate torch downgrade on dgy).
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
Q=reports/po_v6_extras_queue.log
EVAL_SEED=20260722
PARTS=reports/hvp_disr_parts
OUTROOT=reports/hvp_eval_disr
DD=reports/deepdive_lc_parts/po_v6
say() { echo "$(date -u +%FT%TZ) [po6x] $*" | tee -a "$Q"; }
mkdir -p reports "$PARTS"

[ -f checkpoints/po_v6_final/set_transformer_best.pt ] \
  || { say "ABORT: checkpoints/po_v6_final/set_transformer_best.pt missing"; exit 1; }
say "PO V6 EXTRAS ARMED (pid $$, GPU $CUDA_VISIBLE_DEVICES: deep-dive lifecycle + 5 disr parts)"

deepdive() {
  if [ -s "$DD/lifecycle/episodes.csv" ]; then say "deepdive po_v6 cached"; return; fi
  say "deepdive po_v6 start"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario lifecycle --record-every 200 \
    --agents po_v6 --eval-seed "$EVAL_SEED" \
    --out-root "$DD" \
    > reports/deepdive_lc_po_v6.log 2>&1
  say "deepdive po_v6 rc=$?"
}

part() {  # $1 scenario
  local A=po_v6 S=$1 EXTRA=""
  if [ -s "$PARTS/$A/$S/episodes.csv" ]; then say "disr part $A/$S cached"; return; fi
  [ "$S" = massive_scale ] && EXTRA="--steps 25000 --n-eps 3"
  [ "$S" = very_long ] && EXTRA="--record-every 200"
  [ "$S" = lifecycle ] && EXTRA="--record-every 200"
  say "disr part $A/$S start"
  nice -n 10 uv run --no-sync python scripts/eval_human_vs_performance.py \
    --scenario "$S" $EXTRA --agents "$A" --eval-seed "$EVAL_SEED" \
    --out-root "$PARTS/$A" \
    >> "reports/disr_part_${A}_${S}.log" 2>&1
  say "disr part $A/$S rc=$?"
}

# three long evaluations in parallel, then the three short ones
deepdive & part very_long & part lifecycle & wait
part massive_scale & part baseline & part small_scale & wait

if [ -s "$DD/lifecycle/episodes.csv" ]; then
  echo "$(date -u +%FT%TZ) [ddlc] po_v6 done (re-run via scripts/dgy_po_v6_extras.sh)" \
    >> reports/deepdive_lc_queue.log
  say "deepdive po_v6 done"
else
  say "deepdive po_v6 MISSING (see reports/deepdive_lc_po_v6.log)"
fi

for S in very_long lifecycle massive_scale baseline small_scale; do
  if [ -s "$PARTS/po_v6/$S/episodes.csv" ]; then
    nice -n 10 uv run --no-sync python scripts/merge_hvp_parts.py \
      --parts "$PARTS" --dest "$OUTROOT/$S" --scenario "$S" \
      >> reports/disr_merge.log 2>&1
    say "disr merge $S rc=$?"
  else
    say "disr part po_v6/$S MISSING — merge skipped for $S"
  fi
done
echo "$(date -u +%FT%TZ) [disr] po_v6 parts merged (scripts/dgy_po_v6_extras.sh)" \
  >> reports/disr_bench_queue.log
say "PO V6 EXTRAS DONE"
