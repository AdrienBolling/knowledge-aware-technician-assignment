"""Generate per-scenario benchmark notebooks from the master template.

For each scenario in :data:`SCENARIOS` this stamps out a self-contained
notebook in ``benchmarks/`` that runs the trained agent against every
heuristic baseline on that scenario's env config.  The notebooks are
modelled on the manual ``examples/benchmark_agent.ipynb`` workflow but
with two simplifications:

* The user-config cell is **pre-filled** with the scenario's env config
  path and a horizon profile chosen to match the scenario size.
* Part 3 (the massive-scale cross-scenario stress test in the master
  notebook) is **stripped** — each per-scenario notebook focuses on one
  factory configuration, so a cross-scenario section would be
  redundant.

Run this script whenever you tweak the master template or want to add
a new scenario::

    python benchmarks/_generate.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    """Per-scenario customisation knobs applied to the master template."""

    name: str                       # filename stem, e.g. "baseline"
    env_config: str                 # path to the env JSON
    description: str                # one-line description for the title cell
    short_n_episodes: int           # Part 1: number of multi-episode rollouts
    short_max_sim_time: float       # Part 1: per-episode horizon
    short_max_steps: int            # Part 1: per-episode step cap
    long_max_sim_time: float        # Part 2: single-episode horizon
    long_max_steps: int             # Part 2: single-episode step cap


SCENARIOS: list[Scenario] = [
    Scenario(
        name="baseline",
        env_config="run_configs/benchmark_suite/baseline.json",
        description="The reference scenario — 4 techs, 12-22 machines, default disruptions, 200k sim time.",
        short_n_episodes=5,
        short_max_sim_time=200_000.0,
        short_max_steps=20_000,
        long_max_sim_time=2_000_000.0,
        long_max_steps=200_000,
    ),
    Scenario(
        name="short",
        env_config="run_configs/benchmark_suite/short.json",
        description="Fast smoke / sweep scenario — same as baseline but at 1/10 the horizon.",
        short_n_episodes=10,        # lower-variance because each episode is cheap
        short_max_sim_time=20_000.0,
        short_max_steps=2_000,
        long_max_sim_time=200_000.0,
        long_max_steps=20_000,
    ),
    Scenario(
        name="no_fatigue",
        env_config="run_configs/benchmark_suite/no_fatigue.json",
        description="Isolates the knowledge-matching dimension by zeroing fatigue dynamics + dropping the exhaustion disruption.",
        short_n_episodes=5,
        short_max_sim_time=200_000.0,
        short_max_steps=20_000,
        long_max_sim_time=2_000_000.0,
        long_max_steps=200_000,
    ),
    Scenario(
        name="no_disruption",
        env_config="run_configs/benchmark_suite/no_disruption.json",
        description="Isolates steady-state assignment quality by dropping every disruption process.",
        short_n_episodes=5,
        short_max_sim_time=200_000.0,
        short_max_steps=20_000,
        long_max_sim_time=2_000_000.0,
        long_max_steps=200_000,
    ),
    Scenario(
        name="long_horizon",
        env_config="run_configs/benchmark_suite/long_horizon.json",
        description="Knowledge-investment payoff regime — 2M sim time with all techs starting as trainees.",
        short_n_episodes=3,
        short_max_sim_time=200_000.0,        # downsampled from the env's nominal 2M
        short_max_steps=20_000,
        long_max_sim_time=2_000_000.0,
        long_max_steps=200_000,
    ),
    Scenario(
        name="small_scale",
        env_config="run_configs/benchmark_suite/small_scale.json",
        description="Action-space lower bound — 2 techs (generalist + junior), 3-5 machines from 2 templates.",
        short_n_episodes=3,
        short_max_sim_time=200_000.0,        # downsampled from the env's nominal 2M
        short_max_steps=20_000,
        long_max_sim_time=2_000_000.0,
        long_max_steps=200_000,
    ),
    Scenario(
        name="massive_scale",
        env_config="run_configs/benchmark_suite/massive_scale.json",
        description="Stress test — 30 techs, 80-100 machines from all 12 component-bearing templates.",
        short_n_episodes=1,                  # each rollout is expensive
        short_max_sim_time=100_000.0,        # downsampled from the env's nominal 2M
        short_max_steps=10_000,
        long_max_sim_time=500_000.0,
        long_max_steps=50_000,
    ),
]


TEMPLATE_NB = Path("examples/benchmark_agent.ipynb")
OUT_DIR = Path("benchmarks")


def _config_cell_source(s: Scenario) -> str:
    """Return the source for the user-config cell, tailored to the scenario."""
    return f"""from pathlib import Path

# --- scenario --------------------------------------------------------------
# Auto-generated by benchmarks/_generate.py.
# Re-run that script after editing examples/benchmark_agent.ipynb to refresh.
SCENARIO_NAME      = {s.name!r}
TRAINED_ENV_CONFIG = Path({s.env_config!r})

# --- trained agent ---------------------------------------------------------
TRAINED_AGENT_CONFIG = Path("run_configs/agents/set_transformer.json")
TRAINED_CHECKPOINT   = Path("checkpoints/set_transformer_final.pt")
TRAINED_AGENT_LABEL  = f"SetTransformer ({{SCENARIO_NAME}})"

# --- baselines to compare against ------------------------------------------
BASELINES: list[str] = [
    "random",
    "round_robin",
    "least_busy",
    "least_fatigued",
    "shortest_queue",
]

# --- evaluation horizon (Part 1: multi-episode KPI comparison) -------------
N_EVAL_EPISODES    = {s.short_n_episodes}
MAX_EVAL_SIM_TIME  = {s.short_max_sim_time:_}
MAX_EVAL_STEPS     = {s.short_max_steps:_}
EVAL_SEED          = 4321
DETERMINISTIC      = True

# --- output ----------------------------------------------------------------
REPORTS_ROOT = Path("reports") / f"benchmarks_{{SCENARIO_NAME}}"

print(f"Scenario           : {{SCENARIO_NAME}}")
print(f"Trained env config : {{TRAINED_ENV_CONFIG}}")
print(f"Trained agent cfg  : {{TRAINED_AGENT_CONFIG}}")
print(f"Checkpoint         : {{TRAINED_CHECKPOINT}}")
print(f"Baselines          : {{BASELINES}}")
print(f"Episodes / agent   : {{N_EVAL_EPISODES}}")
print(f"Max sim time       : {{MAX_EVAL_SIM_TIME}}")
print(f"Max env steps      : {{MAX_EVAL_STEPS}}")
"""


def _long_config_cell_source(s: Scenario) -> str:
    """Return the source for the Part 2 long-episode config cell."""
    return f"""# --- Part 2 horizon (single long episode per agent) ------------------------
LONG_MAX_SIM_TIME = {s.long_max_sim_time:_}
LONG_MAX_STEPS    = {s.long_max_steps:_}
LONG_SEED         = EVAL_SEED + 7

print(f"Long-episode horizon : sim_time={{LONG_MAX_SIM_TIME:.0f}}  steps={{LONG_MAX_STEPS}}")
print(f"Long-episode seed    : {{LONG_SEED}}")
"""


def _title_cell_source(s: Scenario) -> list[str]:
    return [
        f"# Benchmark — `{s.name}` scenario\n",
        "\n",
        f"{s.description}\n",
        "\n",
        f"This notebook is **auto-generated** from `examples/benchmark_agent.ipynb` by `benchmarks/_generate.py`.  It runs the trained SetTransformer against the heuristic baselines on the `{s.name}` env config in two passes — multi-episode KPI comparison (Part 1) and one long single-episode trace (Part 2).  Run-all-cells is safe.\n",
        "\n",
        "Edit the master template (`examples/benchmark_agent.ipynb`) and re-run the generator to refresh.\n",
    ]


def _lines(text: str) -> list[str]:
    """Convert a multiline string into Jupyter's list-of-lines source format."""
    lines = text.split("\n")
    # Trailing-newline convention: every line except the last gets \n
    out = [line + "\n" for line in lines[:-1]]
    if lines[-1] != "":
        out.append(lines[-1])
    return out


def _is_p3_cell(cell: dict) -> bool:
    """Return True if the cell belongs to Part 3 (massive-scale section)."""
    cid = str(cell.get("id", ""))
    if cid.startswith("p3-"):
        return True
    # Some master-template iterations may have used different ids — guard
    # by also checking the markdown title content.
    src = "".join(cell.get("source") or [])
    return "PART 3" in src and cell["cell_type"] == "markdown"


def _patch_template(template: dict, s: Scenario) -> dict:
    """Return a deep copy of ``template`` customised for scenario ``s``."""
    nb = deepcopy(template)

    # 1. Strip Part 3 cells -- they belong to the master notebook only.
    nb["cells"] = [c for c in nb["cells"] if not _is_p3_cell(c)]

    # 2. Patch the title cell (c00), the Part 1 user-config cell (c02),
    #    and the Part 2 long-episode config cell (99856de0).
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid == "c00":
            cell["source"] = _title_cell_source(s)
            cell["metadata"] = {}
        elif cid == "c02":
            cell["source"] = _lines(_config_cell_source(s))
            cell["execution_count"] = None
            cell["outputs"] = []
        elif cid == "99856de0":
            cell["source"] = _lines(_long_config_cell_source(s))
            cell["execution_count"] = None
            cell["outputs"] = []

    # 3. Belt-and-braces: clear all execution counts and outputs across the
    #    notebook so the generated artefacts are "pre-run" clean.
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    return nb


def main() -> int:
    if not TEMPLATE_NB.is_file():
        msg = (
            f"Template not found: {TEMPLATE_NB}.  Run from the repo root."
        )
        raise SystemExit(msg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = json.loads(TEMPLATE_NB.read_text())

    for s in SCENARIOS:
        nb = _patch_template(template, s)
        out = OUT_DIR / f"{s.name}.ipynb"
        out.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"  wrote {out}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
