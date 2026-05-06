"""CLI entrypoint for running KATA experiments.

Usage
-----
::

    # Train a GRPO agent
    kata-experiment \\
        --env run_configs/complex_factory.json \\
        --agent run_configs/agents/grpo.json \\
        --experiment run_configs/experiments/default.json

    # Evaluate a checkpoint
    kata-experiment \\
        --env run_configs/complex_factory.json \\
        --agent run_configs/agents/grpo.json \\
        --experiment run_configs/experiments/default.json \\
        --mode eval --checkpoint checkpoints/grpo_best.pt

    # Train + full evaluation (separate W&B runs)
    kata-experiment \\
        --env run_configs/complex_factory.json \\
        --agent run_configs/agents/grpo.json \\
        --experiment run_configs/experiments/default.json \\
        --mode evaluated_training
"""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and run the experiment."""
    parser = argparse.ArgumentParser(
        prog="kata-experiment",
        description="Train or evaluate a KATA agent on a factory environment.",
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Path to the environment JSON config (KATAConfig).",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Path to the agent JSON config (AgentConfig).",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to the experiment JSON config (ExperimentConfig).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "evaluated_training"],
        default=None,
        help="Override the experiment mode from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the experiment seed.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to an agent checkpoint to load (overrides agent config).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all stdout output (SimPy noise is always suppressed).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging regardless of config.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Override the W&B project name.",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="Override the W&B group name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Extra W&B tags to append.",
    )

    args = parser.parse_args(argv)

    # Configure logging based on --quiet flag
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    # Lazy imports to keep CLI startup fast
    from experiment.runner import Experiment

    exp = Experiment.from_configs(
        env_config=args.env,
        agent_config=args.agent,
        experiment_config=args.experiment,
        quiet=args.quiet,
    )

    # Apply CLI overrides
    if args.mode is not None:
        exp.exp_cfg.mode = args.mode

    if args.seed is not None:
        exp.exp_cfg.seed = args.seed

    if args.checkpoint is not None:
        exp.agent_cfg.checkpoint = args.checkpoint
        exp.agent.load(args.checkpoint)
        logging.getLogger("experiment.runner").info(
            "Loaded checkpoint: %s", args.checkpoint
        )

    if args.no_wandb:
        exp.exp_cfg.wandb.enabled = False

    if args.wandb_project is not None:
        exp.exp_cfg.wandb.project = args.wandb_project

    if args.wandb_group is not None:
        exp.exp_cfg.wandb.group = args.wandb_group

    if args.wandb_tags:
        exp.exp_cfg.wandb.tags.extend(args.wandb_tags)

    exp.run()


if __name__ == "__main__":
    main()
