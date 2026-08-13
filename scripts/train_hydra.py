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
    # null = respect the env config's own reward stack (v5+ configs set
    # knowledge_increment_potential_based themselves); an explicit
    # true/false still force-overrides for ablations.
    if cfg.get("potential_knowledge_reward") is not None:
        env_data["gym"].setdefault("reward", {})[
            "knowledge_increment_potential_based"
        ] = bool(cfg.potential_knowledge_reward)

    # ----- agent: composed group + improvement toggles -----------------
    agent_data = OmegaConf.to_container(cfg.agent, resolve=True)
    params = agent_data["params"]
    agent_type = str(agent_data.get("agent_type", "set_transformer"))
    # LR schedule sized to the actual update budget.  Decision density is
    # configurable (``tu_per_decision``): the multiscale event-driven world
    # measures ~24 t.u./decision (2026-07-22 probe: 8,249 decisions per
    # 200k t.u.); the historical 60-t.u. constant under-estimated rounds
    # ~3x and let the cosine schedule hit its (now floored) tail mid-run.
    mean_sim = (
        (float(cfg.sim_time_min) + float(cfg.sim_time_max)) / 2.0
        if cfg.get("sim_time_min") is not None and cfg.get("sim_time_max") is not None
        else float(cfg.sim_time)
    )
    tu_per_decision = float(cfg.get("tu_per_decision") or 24.0)
    est_steps_per_ep = mean_sim / tu_per_decision

    def _sized_rounds(rollout_steps: int) -> int:
        if int(cfg.parallel_envs) == 1:
            # The serial loop updates exactly once per EPISODE and
            # never consults ``rollout_steps``.
            return int(cfg.episodes)
        # 1.25 safety inflation: the decision-density constant is an
        # estimate (v5 corrected world measures ~20 t.u./decision vs
        # the 24 configured) — a slightly slower cosine descent is
        # benign, while undershooting parks the tail on the
        # lr_min_factor floor for the last stretch of the run.
        return max(
            200,
            int(
                1.25
                * int(cfg.episodes)
                / max(1, int(cfg.parallel_envs))
                * est_steps_per_ep
                / rollout_steps
            ),
        )

    if agent_type == "set_transformer":
        params["use_popart"] = bool(cfg.use_popart)
        if params.get("use_popart"):
            params["normalize_rewards"] = False  # mutually exclusive
        if not bool(cfg.use_gru):
            params["rnn_type"] = "none"
        if cfg.gamma is not None:
            params["gamma"] = float(cfg.gamma)
        params["time_based_discount"] = bool(
            cfg.get("time_based_discount", False)
        )
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
        est_rounds = _sized_rounds(int(cfg.rollout_steps))
        params["total_updates"] = est_rounds
        params["warmup_updates"] = max(10, est_rounds // 25)
    elif agent_type == "a2c_mlp":
        # Traditional MLP baseline: no PopArt/GRU/semi-MDP injection —
        # the JSON's per-decision gamma and rollout length rule.  The
        # cosine schedule is sized with the AGENT's rollout length
        # (128), not the PPO default 2048: A2C's single full-batch step
        # makes the rollout the batch.
        if cfg.gae_lambda is not None:
            params["gae_lambda"] = float(cfg.gae_lambda)
        est_rounds = _sized_rounds(int(params.get("rollout_steps", 128)))
        params["total_updates"] = est_rounds
        params["warmup_updates"] = max(10, est_rounds // 25)
    elif agent_type == "grpo_mlp":
        # One gradient phase per GROUP of complete episodes; the group
        # must share one sampled scenario, so the sampler's rotation
        # cadence is pinned to the group size regardless of the
        # episodes_per_scenario trainer default.
        group = int(params.get("group_size", 8))
        est_rounds = max(4, int(cfg.episodes) // group)
        params["total_updates"] = est_rounds
        params["warmup_updates"] = max(3, est_rounds // 25)
        env_data.setdefault("randomized_scenario", {})[
            "episodes_per_scenario"
        ] = group
        # GRPO's outcome is the EPISODE-SUM of rewards z-scored within
        # the group — a per-episode horizon draw U(min, max) would make
        # the dominant term of that statistic an exogenous ±26% length
        # spread the policy cannot influence (confirmed empirically:
        # corr(horizon, return) ≈ −0.7 under the v5 reward drift).  Pin
        # every episode to the fixed mean horizon instead; scenario
        # SIZE stays multiscale.
        env_data["gym"].pop("max_sim_time_min", None)
        env_data["gym"].pop("max_sim_time_max", None)
    elif agent_type == "dql_mlp":
        # Constant LR, own gamma, cadence self-managed in
        # observe_transition.  The agent's exploration/replay RNGs are
        # PRIVATE streams (they must not perturb the simulator's seeded
        # global stream), so the experiment seed has to reach the ctor
        # explicitly or real runs draw from OS entropy.
        params.setdefault("seed", int(cfg.seed))
    else:
        raise ValueError(
            f"train_hydra has no parameter-injection policy for "
            f"agent_type={agent_type!r}; add a branch before training "
            "with it (unconditional PPO params would reach its ctor)."
        )
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
    print(f"  agent type           : {agent_type}")
    print(f"  episodes             : {int(cfg.episodes)}  @ {float(cfg.sim_time):,.0f} t.u.")
    print(f"  parallel envs        : {int(cfg.parallel_envs)}")
    print(f"  gamma / gae_lambda   : {params.get('gamma')} / {params.get('gae_lambda')}"
          + (" (semi-MDP: gamma per t.u.)" if params.get("time_based_discount") else " (per decision)"))
    print(f"  popart / rnn         : {params.get('use_popart')} / {params.get('rnn_type')}")
    _pbrs = env_data["gym"].get("reward", {}).get(
        "knowledge_increment_potential_based", False
    )
    print(f"  knowledge reward     : "
          f"{'potential-based' if _pbrs else 'HC-v1 legacy (floored)'}")
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
    print(f"Training complete.  Best checkpoint: {checkpoint_dir}/{agent_type}_best.pt")
    return 0


if __name__ == "__main__":
    main()
