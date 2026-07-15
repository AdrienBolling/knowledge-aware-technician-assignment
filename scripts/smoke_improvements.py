"""End-to-end smoke tests for the opt-in PPO improvements.

Runs four tiny trainings (a few episodes at a 3k-t.u. horizon) and one
metric-based mini-eval each, verifying that nothing is broken:

  A. regression   — original agent config, single env (classic path)
  B. vectorised   — parallel_envs=2 (async workers, step-based rounds)
  C. popart       — use_popart=true
  D. gru          — rnn_type=gru

Success criteria per variant: training completes, PPO losses are
finite, a checkpoint save/load round-trip works, and a deterministic
eval episode reports sane *metrics* (finished products, MTTR) — per
the project rule, improvements are judged on metrics, not rewards.

Usage::

    uv run python scripts/smoke_improvements.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")
os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

import numpy as np

SMOKE_SIM, SMOKE_STEPS = 3_000.0, 300


def _tiny_env_cfg(src: str) -> dict:
    cfg = json.loads(Path(src).read_text())
    cfg["gym"]["max_sim_time"] = SMOKE_SIM
    cfg["gym"]["max_episode_steps"] = SMOKE_STEPS
    # The set-transformer agent trains on the grouped 'set' representation
    # (training env configs like factory_set.json set this; the benchmark
    # suite configs default to token_ids for the notebook workflows).
    cfg["gym"]["observation_representation"] = "set"
    return cfg


def _tiny_agent_cfg(src: str, **overrides) -> dict:
    cfg = json.loads(Path(src).read_text())
    cfg["params"].update(
        rollout_steps=64,
        minibatch_size=32,
        n_epochs=2,
        total_updates=10,
        warmup_updates=1,
        **overrides,
    )
    return cfg


def _tiny_exp_cfg(**overrides) -> dict:
    return {
        "mode": "train",
        "seed": 7,
        "n_episodes": 3,
        "log_interval": 100,
        "eval": {"enabled": False},
        "checkpoint": {"enabled": False},
        "wandb": {"enabled": False},
        "reports": {"enabled": False},
        **overrides,
    }


def run_variant(name, env_cfg, agent_cfg, exp_cfg) -> bool:
    from experiment.config import AgentConfig, ExperimentConfig
    from experiment.runner import Experiment
    from kata.core.config import KATAConfig

    print(f"\n=== {name} ===", flush=True)
    exp = Experiment(
        env_config=KATAConfig(**env_cfg),
        agent_config=AgentConfig(**agent_cfg),
        experiment_config=ExperimentConfig(**exp_cfg),
        quiet=True,
    )
    history = exp._train_loop()
    losses = [x for x in history.get("loss", []) if not math.isnan(x)]
    n_eps = len(history.get("return", []))
    print(f"  episodes trained : {n_eps}")
    print(f"  finite losses    : {len(losses)} {losses[:3]}")
    ok = n_eps >= 1 and all(math.isfinite(x) for x in losses)

    # checkpoint round-trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    exp.agent.save(ckpt_path)
    exp.agent.load(ckpt_path)
    os.unlink(ckpt_path)
    print("  save/load        : OK")

    # metric-based mini-eval (deterministic, on the fixed eval scenario)
    ep = exp._run_episode(training=False, seed=123)
    em = ep.get("episode_metrics", {})
    finished = em.get("finished_products", float("nan"))
    mttr = em.get("mttr", float("nan"))
    print(f"  eval metrics     : finished={finished:.0f}  mttr={mttr:.1f}  "
          f"steps={ep.get('length', 0):.0f}")
    ok &= math.isfinite(float(finished))
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    env_plain = _tiny_env_cfg("run_configs/benchmark_suite/baseline.json")
    env_crit = _tiny_env_cfg("run_configs/benchmark_suite/baseline_crit.json")

    results = {}
    results["A_regression_single_env"] = run_variant(
        "A. regression (original config, single env)",
        env_plain,
        _tiny_agent_cfg("run_configs/agents/set_transformer.json"),
        _tiny_exp_cfg(),
    )
    results["B_vectorized_2_envs"] = run_variant(
        "B. vectorised (parallel_envs=2, gamma/lambda long-horizon)",
        env_plain,
        _tiny_agent_cfg("run_configs/agents/set_transformer_long.json"),
        _tiny_exp_cfg(parallel_envs=2, n_episodes=4),
    )
    results["C_popart"] = run_variant(
        "C. PopArt value normalisation",
        env_crit,
        _tiny_agent_cfg("run_configs/agents/set_transformer_popart.json"),
        _tiny_exp_cfg(),
    )
    results["D_gru"] = run_variant(
        "D. GRU recurrent context",
        env_crit,
        _tiny_agent_cfg("run_configs/agents/set_transformer_gru.json"),
        _tiny_exp_cfg(),
    )

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
