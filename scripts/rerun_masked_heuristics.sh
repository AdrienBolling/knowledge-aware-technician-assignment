#!/usr/bin/env bash
# Re-run ONLY the heuristic baselines with the availability-aware
# (action-mask-reading) implementations of agents/baselines/heuristics.py,
# merging the fresh rows into the existing benchmark artifacts so the
# expensive trained-agent episodes are reused untouched.
#
# Rationale: the original heuristics read only `technician_busy`, which is
# blind to disruptions (injury / vacation / exhaustion); at industrial
# scale the deterministic rules funnel every ticket raised during an
# absence onto the same absent technician and the line collapses.  The
# mask-aware variants see exactly what the masked RL policies see.
#
# Usage (from the repo root; ~40-60 min for the three scales, plus
# ~2h for the very_long heuristic legs):
#
#   bash scripts/rerun_masked_heuristics.sh            # three scales
#   bash scripts/rerun_masked_heuristics.sh very_long  # + very-long study
#
# Afterwards regenerate tables and figures:
#   uv run python scripts/analyze_hvp_results.py
#   uv run python scripts/plot_hvp_figures.py
#
# NOTE: Section 6's prose (industrial-scale heuristic collapse, the
# "fatigue-exhaustion spiral" sentence, and the setup paragraph) must be
# updated to the new numbers after this run — the collapse mechanism was
# disruption-blindness, not fatigue (see analysis/heuristic_collapse_probe
# notes in the session log / memory).

set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  echo "=== $1 (heuristics only, merge) ==="
  uv run python scripts/eval_human_vs_performance.py \
    --scenario "$1" --agents heuristics --merge "${@:2}"
}

run baseline
run small_scale
run massive_scale --n-eps 2 --steps 25000

if [[ "${1:-}" == "very_long" ]]; then
  run very_long --record-every 5
fi

echo "=== re-analyzing ==="
uv run python scripts/analyze_hvp_results.py
uv run python scripts/plot_hvp_figures.py
echo "Done. Remember to update Section 6 prose to the new numbers."
