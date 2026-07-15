"""Train the human-centric (HC) agent with the opt-in PPO improvements.

One-liner launch (from the repo root, on any machine with the repo +
``uv sync`` done)::

    uv run python scripts/train_hc_improved.py

Defaults bundle every improvement of commit 2358d64 on top of the HC
reward stack, and the training regime requested for the second-
generation agent:

* human-centric rewards + machine_criticality + downtime_cost
  (``baseline_crit.json``) — the bottleneck-blindness fix;
* **longer training**: 2000 episodes (vs 1500 historically);
* **longer episodes**: 500k simulated time units (vs 200k) so the
  policy sees deeper into the knowledge-saturation curve;
* **fresh factory layout every 5 episodes** (``episodes_per_scenario``);
* vectorised collection over N parallel simulators (default 6);
* long-horizon credit (gamma 0.997, GAE-lambda 0.97);
* PopArt value normalisation;
* GRU recurrent context.

Every improvement can be switched off individually for ablations
(``--no-popart``, ``--no-gru``, ``--parallel-envs 1``,
``--gamma 0.99 --gae-lambda 0.95``, ``--env-config
run_configs/benchmark_suite/baseline.json``), so the same script drives
the whole ablation grid on the remote machine.

Checkpoints land in ``--checkpoint-dir`` as
``set_transformer_best.pt`` (best periodic eval) and
``set_transformer_round*.pt``; rename after the run to
``hc_v2_set_transformer_best.pt`` or similar before benchmarking.
Progress is judged by the periodic deterministic eval's *metrics*
(finished products, MTTR, availability) — not by the training reward.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the HC agent with the opt-in PPO improvements."
    )
    ap.add_argument("--env-config",
                    default="run_configs/benchmark_suite/baseline_crit.json",
                    help="training env config (default: HC rewards + criticality/downtime)")
    ap.add_argument("--agent-config",
                    default="run_configs/agents/set_transformer_gru.json",
                    help="base agent config (default: long-horizon gamma/lambda + GRU)")
    ap.add_argument("--episodes", type=int, default=2000,
                    help="total training episodes (default 2000)")
    ap.add_argument("--sim-time", type=float, default=500_000.0,
                    help="episode horizon in simulated time units (default 500k)")
    ap.add_argument("--max-steps", type=int, default=25_000,
                    help="decision cap per episode; keep non-binding (default 25k)")
    ap.add_argument("--episodes-per-scenario", type=int, default=5,
                    help="reuse each sampled factory layout for N episodes (default 5)")
    ap.add_argument("--parallel-envs", type=int, default=6,
                    help="parallel simulator workers; 1 = classic loop (default 6)")
    ap.add_argument("--rollout-steps", type=int, default=2048,
                    help="PPO round length per worker (default 2048)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="override gamma (default: from agent config, 0.997)")
    ap.add_argument("--gae-lambda", type=float, default=None,
                    help="override GAE lambda (default: from agent config, 0.97)")
    ap.add_argument("--no-popart", action="store_true", help="disable PopArt")
    ap.add_argument("--no-gru", action="store_true", help="disable the GRU context")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-dir", default="checkpoints/hc_v2")
    ap.add_argument("--checkpoint-interval", type=int, default=50,
                    help="checkpoint every N PPO rounds (vec) / episodes (classic)")
    ap.add_argument("--eval-interval", type=int, default=25,
                    help="deterministic metric eval every N rounds/episodes")
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="kata-set-transformer")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    from experiment.config import AgentConfig, ExperimentConfig
    from experiment.runner import Experiment
    from kata.core.config import KATAConfig

    # ----- environment: HC rewards, long episodes, rotating layouts -----
    env_data = json.loads(Path(args.env_config).read_text())
    env_data["gym"]["max_sim_time"] = float(args.sim_time)
    env_data["gym"]["max_episode_steps"] = int(args.max_steps)
    # The set-transformer trains on the grouped 'set' representation.
    env_data["gym"]["observation_representation"] = "set"
    env_data.setdefault("randomized_scenario", {})
    env_data["randomized_scenario"]["episodes_per_scenario"] = int(
        args.episodes_per_scenario
    )

    # ----- agent: long-horizon credit + PopArt + GRU (each optional) -----
    agent_data = json.loads(Path(args.agent_config).read_text())
    params = agent_data["params"]
    params["use_popart"] = not args.no_popart
    if params.get("use_popart"):
        params["normalize_rewards"] = False
    if args.no_gru:
        params["rnn_type"] = "none"
    if args.gamma is not None:
        params["gamma"] = float(args.gamma)
    if args.gae_lambda is not None:
        params["gae_lambda"] = float(args.gae_lambda)
    params["rollout_steps"] = int(args.rollout_steps)
    # LR schedule sized to the actual update budget:
    # rounds ~= episodes/worker * steps/episode / rollout_steps.
    est_steps_per_ep = args.sim_time / 60.0  # ~1 decision / 60 t.u. at baseline scale
    est_rounds = max(
        200,
        int(args.episodes / max(1, args.parallel_envs)
            * est_steps_per_ep / args.rollout_steps),
    )
    params["total_updates"] = est_rounds
    params["warmup_updates"] = max(10, est_rounds // 25)

    # ----- experiment: vectorised training, metric-based monitoring -----
    exp_data = {
        "mode": "train",
        "seed": int(args.seed),
        "n_episodes": int(args.episodes),
        "log_interval": 10,
        "parallel_envs": int(args.parallel_envs),
        "eval": {
            "enabled": True,
            "interval": int(args.eval_interval),
            "n_episodes": int(args.eval_episodes),
            "deterministic": True,
        },
        "checkpoint": {
            "enabled": True,
            "interval": int(args.checkpoint_interval),
            "dir": args.checkpoint_dir,
            "save_best": True,
        },
        "wandb": {
            "enabled": not args.no_wandb,
            "project": args.wandb_project,
            "log_interval": 1,
            "tags": ["hc-v2", "improvements", f"vec{args.parallel_envs}"],
        },
        "reports": {"enabled": False},
    }

    print("=== HC-v2 training configuration ===")
    print(f"  env config           : {args.env_config}")
    print(f"  episodes             : {args.episodes}  @ {args.sim_time:,.0f} t.u.")
    print(f"  scenario rotation    : every {args.episodes_per_scenario} episodes")
    print(f"  parallel envs        : {args.parallel_envs}")
    print(f"  gamma / gae_lambda   : {params['gamma']} / {params['gae_lambda']}")
    print(f"  popart / rnn         : {params.get('use_popart')} / {params.get('rnn_type')}")
    print(f"  rollout / lr budget  : {params['rollout_steps']} steps, "
          f"{params['total_updates']} updates")
    print(f"  checkpoints          : {args.checkpoint_dir}")
    print(f"  wandb                : {not args.no_wandb}")

    exp = Experiment(
        env_config=KATAConfig(**env_data),
        agent_config=AgentConfig(**agent_data),
        experiment_config=ExperimentConfig(**exp_data),
        quiet=bool(args.quiet),
    )
    exp.run()
    print("Training complete.  Best checkpoint: "
          f"{args.checkpoint_dir}/set_transformer_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
