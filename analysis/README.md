# Analysis notebooks

A suite of Jupyter notebooks for exploring the per-experiment CSV
reports written by `Experiment.run()` under `reports/<exp_id>/`.

## Layout

| File | Purpose |
|---|---|
| `utils.py` | Shared loaders (`load_run`, `list_runs`, `latest_run`, …) and column-grouping helpers. Imported by every notebook. |
| `_build_notebooks.py` | Single source of truth that re-generates every `.ipynb` in this folder. Re-run after editing it. |
| `00_overview.ipynb` | What runs are available · how to load one · column structure · run summary. |
| `01_learning_curves.ipynb` | Episode return + PPO update metrics over training. Train-vs-eval gap. |
| `02_reward_decomposition.ipynb` | Where the return comes from: per-component contributions (total + per-step). |
| `03_episode_kpis.ipynb` | Manufacturing KPIs: MTTR, fleet availability, throughput, breakdowns. |
| `04_per_technician.ipynb` | Workload distribution, knowledge / fatigue evolution, specialisation cross-tab. |
| `05_per_machine.ipynb` | Maintenance / production time per machine, breakdown counts, utilisation. |
| `06_compare_runs.ipynb` | Side-by-side comparison across `exp_id`s (returns, distributions, summary table). |

## Quickstart

```bash
# 1. Generate the notebooks (already done, regenerate after editing _build_notebooks.py)
uv run python analysis/_build_notebooks.py

# 2. Open them
uv run jupyter lab    # or: code . / pycharm / your editor of choice
```

Every notebook defaults to **`latest_run()`** so opening 00, 01, … in
order against your most recent training run "just works".  Override
`EXP_ID = "your_run"` in the second cell to pin a specific run.

## Conventions in the dataframes

* **`phase`** ∈ `train` / `inline_eval` / `eval` distinguishes the
  source of the row.
* Columns are namespaced by prefix — use `group_columns(df, "metric")`
  to pull all metric columns at once.

| Prefix | Source |
|---|---|
| `metric/<name>` | episode-level KPI from `EPISODE_METRICS` |
| `step_mean/<name>` | mean of a step series over the episode |
| `reward_sum/<comp>` / `reward_mean/<comp>` | reward components |
| `update/<name>` | PPO agent-update metrics (`approx_kl`, `lr`, …) |
| `assignments/<tech>` | repairs assigned to each tech |
| `machine_maintenance/<m>` / `machine_production/<m>` / `machine_breakdowns/<m>` | per-machine stats |

## Extending the suite

Notebooks are generated from `_build_notebooks.py` — adding a new one is:

1. Define `_make_NN_my_topic() -> Cells` in the script.
2. Append `("NN_my_topic.ipynb", _make_NN_my_topic)` to the `NOTEBOOKS` list.
3. Re-run `uv run python analysis/_build_notebooks.py`.

Or just edit the generated `.ipynb` directly — the script is only a
one-shot generator, not a watcher.
