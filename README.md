# Knowledge-Aware Technician Allocation: code release

This branch (`journal`) is the code release of the article
*Knowledge-Aware Technician Allocation: The Long-Term Impact of Technician
Upskilling* (Adrien Bolling, Sylvain Kubler, Marcelo Luis Ruiz-Rodríguez,
and Yves Le Traon, SnT, University of Luxembourg). It keeps the code that the
article uses and removes the other experiments of the project.

- For the complete experiment settings of the article, see
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
- To cite this code, use [CITATION.cff](CITATION.cff).
- The code is released under the MIT license ([LICENSE](LICENSE)).

## Contents

| Path | Contents |
|---|---|
| `src/kata/` | FactoReal, the simulator of the article (code name `kata`): the `KataEnv` Gymnasium environment, technicians, machines, reward components, and metrics |
| `src/agents/ppo/ppo_set_transformer.py` | HTT-RL (`SetTransformerAgent`); its network is in `src/agents/networks/` |
| `src/agents/a2c/`, `src/agents/grpo/`, `src/agents/dqn/` | the MLP anchors A2C-MLP, GRPO-MLP, and DDQN-MLP |
| `src/agents/baselines/heuristics.py` | the rule-based, multicriteria, and optimization baselines |
| `src/experiment/` | the training loop and the parallel environments |
| `conf/`, `scripts/train_hydra.py` | the training launcher (Hydra) |
| `scripts/warmstart_bc.py` | the behavior-cloning initialization of HTT-RL |
| `run_configs/` | the training and evaluation configurations, the HTT-RL and anchor configurations, the vocabulary, and the ticket embedding |
| `scripts/eval_human_vs_performance.py`, `scripts/merge_hvp_parts.py` | the evaluation harness and the merge of its parts |
| `scripts/*.sh` | the queue scripts of the published training and evaluation runs |
| `scripts/summary_scores.py`, `scripts/make_*.py`, `scripts/financial_analysis.py`, `scripts/kpi_weight_sensitivity.py`, `scripts/compare_wall_time.py` | the generators of the tables and figures of the article |
| `tests/` | the unit and regression tests |

The evaluation records (`reports/`) and the checkpoints (`checkpoints/`) are
not in this repository. The table and figure generators read the records
from `reports/hvp_eval_v6w/`, `reports/hvp_v6w_parts/`, and
`reports/hvp_eval_disr/`.

## Installation and tests

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
uv run pytest -q
```

## Time units

Simulation time has no unit. The article reads **1 time unit ≈ 1 minute**.
With this anchor:

| Quantity | Value | Reads as |
|---|---|---|
| Technician travel delay | 15 t.u. | 15 minutes |
| Mean MTTR at the industrial scale (S3) | about 80–110 t.u. | ≈ 1.3–1.8 h |
| Operational horizons (S1–S3) | 1–2 × 10⁵ t.u. | ≈ 10 weeks – 4.5 months |
| Long scenarios (S4, S5) | 5 × 10⁶ t.u. | ≈ 9.5 years (career scale) |

The simulator models continuous coverage (no shift patterns), so the anchor
is indicative and not literal. The top axes of the scenario figures of the
article use the same anchor.

## Observations

`KataEnv` supports five observation representations, selected with
`gym.observation_representation` in `GymEnvConfig`: `structured` (default),
`tokens`, `token_ids`, `hybrid`, and `set`. In the article, HTT-RL and the MLP
anchors use `set`, and the baselines use `structured`.
