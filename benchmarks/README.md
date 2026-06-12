# Per-scenario benchmark notebooks

One self-contained notebook per scenario in the benchmark suite.  Each runs
the trained SetTransformer checkpoint against every heuristic baseline on
the scenario's env config, in two passes:

* **Part 1** — `N_EVAL_EPISODES` short multi-episode rollouts; produces a
  KPI summary table, a six-panel bar chart, and per-step trend plots.
* **Part 2** — one long single-episode rollout per agent; produces
  within-episode trends and a cumulative-throughput plot.

Output artefacts land in `reports/benchmarks_<scenario>/<sub>/` (CSVs +
manifest).  No code changes are required to run — just point
`TRAINED_CHECKPOINT` in the config cell at the checkpoint you want.

## Scenarios

| Notebook | Env config | Highlights |
|---|---|---|
| `baseline.ipynb` | `benchmark_suite/baseline.json` | Reference — 4 techs, 12-22 machines, default disruptions |
| `short.ipynb` | `benchmark_suite/short.json` | 1/10 horizon — for quick iteration |
| `no_fatigue.ipynb` | `benchmark_suite/no_fatigue.json` | Fatigue disabled — isolates knowledge-matching |
| `no_disruption.ipynb` | `benchmark_suite/no_disruption.json` | Disruptions disabled — isolates steady-state quality |
| `long_horizon.ipynb` | `benchmark_suite/long_horizon.json` | 2M sim time, all-trainee fleet — knowledge-investment payoff |
| `small_scale.ipynb` | `benchmark_suite/small_scale.json` | 2 techs, 3-5 machines — action-space lower bound |
| `massive_scale.ipynb` | `benchmark_suite/massive_scale.json` | 30 techs, 80-100 machines, 12 templates — scale stress test |

## Choosing a checkpoint

The notebooks default to `checkpoints/set_transformer_final.pt`, which is
whatever the last training run produced.  Edit the `TRAINED_CHECKPOINT`
line in cell `c02` to point at a specific saved checkpoint
(`set_transformer_ep00100.pt`, `set_transformer_best.pt`, …) if you want
to benchmark a specific epoch or the save-best snapshot.

The set tokenizer is resolved via the canonical artefact
(`run_configs/vocab/set_vocab.json`) using the same three-layer load
order the master benchmark notebook uses — embedded > canonical >
rebuild — so the same checkpoint loads consistently regardless of which
scenario you point at.

## Regeneration

Edit the master template (`examples/benchmark_agent.ipynb`) and re-run
the generator from the repo root:

```bash
python benchmarks/_generate.py
```

This overwrites every `<scenario>.ipynb` in this folder with a fresh
copy patched to its scenario.  To add a new scenario, append an entry to
the `SCENARIOS` list at the top of `_generate.py`.

## What got stripped from the template

The master `benchmark_agent.ipynb` ends with a Part 3 block that runs
the same agents on the `massive_scale` env regardless of which scenario
the rest of the notebook was pointed at.  That section is **stripped**
from the per-scenario notebooks here — running the massive scenario
gets its own dedicated `massive_scale.ipynb` instead, and bolting it
onto, say, the `short.ipynb` notebook would be confusing.
