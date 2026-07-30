"""Hydra-composed training launcher.

A thin composition front-end over the exact same training stack as
``scripts/train_hc_improved.py``: Hydra assembles the configuration
(groups ``env``/``agent`` are symlinks to the canonical JSON configs, so
there is a single source of truth), the pydantic models still validate
everything, and ``Experiment`` runs unchanged.  Nothing under ``src/``
knows Hydra exists.

What Hydra adds for ablations and tracking:

* deep CLI overrides at any config depth
  (``'env.gym.reward.fatigue_cost.coefficient=0.5'``);
* ``--multirun`` sweeps that expand an ablation grid into one job per
  combination (``use_popart=true,false seed=42,43,44``);
* a per-run output directory containing the fully-resolved config
  snapshot (``.hydra/config.yaml``) and, by default, the run's
  checkpoints -- every experiment is self-contained and reproducible.

See ``conf/train.yaml`` for the full parameter list and examples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> int:
    from experiment.config import AgentConfig, ExperimentConfig
    from experiment.runner import Experiment
    from kata.core.config import KATAConfig

    # ----- environment: composed group + trainer-level mutations -------
    env_data = OmegaConf.to_container(cfg.env, resolve=True)
    env_data["gym"]["max_sim_time"] = float(cfg.sim_time)
    env_data["gym"]["max_episode_steps"] = int(cfg.max_steps)
    if cfg.get("sim_time_min") is not None and cfg.get("sim_time_max") is not None:
        env_data["gym"]["max_sim_time_min"] = float(cfg.sim_time_min)
        env_data["gym"]["max_sim_time_max"] = float(cfg.sim_time_max)
    env_data["gym"]["observation_representation"] = "set"
    env_data.setdefault("randomized_scenario", {})[
        "episodes_per_scenario"
    ] = int(cfg.episodes_per_scenario)
    env_data["gym"].setdefault("reward", {})[
        "knowledge_increment_potential_based"
    ] = bool(cfg.potential_knowledge_reward)

    # ----- agent: composed group + improvement toggles -----------------
    agent_data = OmegaConf.to_container(cfg.agent, resolve=True)
    params = agent_data["params"]
    params["use_popart"] = bool(cfg.use_popart)
    if params.get("use_popart"):
        params["normalize_rewards"] = False  # mutually exclusive
    if not bool(cfg.use_gru):
        params["rnn_type"] = "none"
    if cfg.gamma is not None:
        params["gamma"] = float(cfg.gamma)
    params["time_based_discount"] = bool(cfg.get("time_based_discount", False))
    if params["time_based_discount"] and params.get("gamma", 1.0) < 0.999:
        raise ValueError(
            "time_based_discount=true interprets gamma per sim-TIME-UNIT: "
            f"gamma={params.get('gamma')} would give a ~"
            f"{1.0 / (1.0 - float(params.get('gamma', 0.99))):.0f} t.u. "
            "horizon (likely a per-decision value passed by mistake). "
            "Use e.g. gamma=0.9999, or set time_based_discount=false."
        )
    if cfg.gae_lambda is not None:
        params["gae_lambda"] = float(cfg.gae_lambda)
    params["rollout_steps"] = int(cfg.rollout_steps)
    # LR schedule sized to the actual update budget (same heuristic as
    # train_hc_improved.py: ~1 decision / 60 t.u. at baseline scale).
    mean_sim = (
        (float(cfg.sim_time_min) + float(cfg.sim_time_max)) / 2.0
        if cfg.get("sim_time_min") is not None and cfg.get("sim_time_max") is not None
        else float(cfg.sim_time)
    )
    est_steps_per_ep = mean_sim / 60.0
    est_rounds = max(
        200,
        int(
            int(cfg.episodes)
            / max(1, int(cfg.parallel_envs))
            * est_steps_per_ep
            / int(cfg.rollout_steps)
        ),
    )
    params["total_updates"] = est_rounds
    params["warmup_updates"] = max(10, est_rounds // 25)
    if cfg.init_checkpoint:
        agent_data["checkpoint"] = str(cfg.init_checkpoint)

    # ----- experiment ---------------------------------------------------
    checkpoint_dir = str(cfg.checkpoint_dir)
    exp_data = {
        "mode": "train",
        "seed": int(cfg.seed),
        "n_episodes": int(cfg.episodes),
        "log_interval": 10,
        "parallel_envs": int(cfg.parallel_envs),
        "eval": {
            "enabled": True,
            "interval": int(cfg.eval_interval),
            "n_episodes": int(cfg.eval_episodes),
            "deterministic": True,
        },
        "checkpoint": {
            "enabled": True,
            "interval": int(cfg.checkpoint_interval),
            "dir": checkpoint_dir,
            "save_best": True,
        },
        "wandb": {
            "enabled": bool(cfg.wandb),
            "project": str(cfg.wandb_project),
            "log_interval": 1,
            "tags": ["hydra", f"vec{int(cfg.parallel_envs)}"],
        },
        "reports": {"enabled": False},
    }

    print("=== Hydra training configuration ===")
    print(f"  env group            : {cfg.env.get('name', '(group)')}")
    print(f"  episodes             : {int(cfg.episodes)}  @ {float(cfg.sim_time):,.0f} t.u.")
    print(f"  parallel envs        : {int(cfg.parallel_envs)}")
    print(f"  gamma / gae_lambda   : {params['gamma']} / {params['gae_lambda']}"
          + (" (semi-MDP: gamma per t.u.)" if params.get("time_based_discount") else " (per decision)"))
    print(f"  popart / rnn         : {params.get('use_popart')} / {params.get('rnn_type')}")
    print(f"  knowledge reward     : "
          f"{'potential-based' if cfg.potential_knowledge_reward else 'HC-v1 legacy (floored)'}")
    print(f"  init checkpoint      : {cfg.init_checkpoint or '(from scratch)'}")
    print(f"  checkpoints          : {checkpoint_dir}")
    print(f"  mca encoder          : {env_data['gym'].get('use_mca_encoder')}")

    # Validate through the same pydantic models regardless of dry_run,
    # so a sweep of dry runs is a full configuration-grid check.
    env_config = KATAConfig(**env_data)
    agent_config = AgentConfig(**agent_data)
    experiment_config = ExperimentConfig(**exp_data)
    if bool(cfg.dry_run):
        print("DRY RUN OK — configuration composed and validated; not training.")
        return 0

    exp = Experiment(
        env_config=env_config,
        agent_config=agent_config,
        experiment_config=experiment_config,
    )
    exp.run()
    print(f"Training complete.  Best checkpoint: {checkpoint_dir}/set_transformer_best.pt")
    return 0


if __name__ == "__main__":
    main()
