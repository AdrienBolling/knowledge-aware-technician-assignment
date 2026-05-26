"""CLI entrypoint for pretraining an MLM encoder on KATA rollouts.

Usage
-----
::

    kata-pretrain-encoder \\
        --env run_configs/factory_long.json \\
        --out checkpoints/mtm_encoder.pt \\
        --n-rollouts 4000 \\
        --epochs 20

By default the collector uses the ``LeastBusyAgent`` heuristic so
trajectories spread well across the state space.  The resulting
``encoder.pt`` (the file written at ``--out``) can be plugged into a
``ppo_latent`` agent config via the ``encoder_path`` parameter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("agents.representation.cli")


@contextmanager
def _quiet_stdout() -> Any:
    """Suppress noisy SimPy prints from machines / routers / sources."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def main(argv: list[str] | None = None) -> None:
    """Parse args and run the pretraining pipeline."""
    parser = argparse.ArgumentParser(
        prog="kata-pretrain-encoder",
        description=(
            "Collect token-obs rollouts with a heuristic agent and "
            "train a masked-token-modelling encoder on them."
        ),
    )
    parser.add_argument("--env", required=True, help="Path to env JSON (KATAConfig).")
    parser.add_argument(
        "--out",
        default="checkpoints/mtm_encoder.pt",
        help="Output path for the MLM checkpoint.",
    )
    parser.add_argument(
        "--collector",
        default="random",
        choices=["random", "round_robin", "least_busy", "least_fatigued", "shortest_queue"],
        help=(
            "Policy used to collect rollouts.  Defaults to ``random`` so "
            "the buffer covers a broad slice of state space — heuristic "
            "collectors over-sample 'sensible' regions and bias the "
            "pretraining distribution."
        ),
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=4000,
        help="Total environment steps recorded into the buffer.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    # KATAConfig must NOT load any other config file
    os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

    # -- lazy imports keep CLI startup fast and avoid torch unless we run
    from agents import (
        LeastBusyAgent,
        LeastFatiguedAgent,
        RandomAgent,
        RoundRobinAgent,
        ShortestQueueAgent,
    )
    from agents.representation.data import collect_token_rollouts
    from agents.representation.mtm import (
        MaskedTokenModel,
        MTMTrainConfig,
        MTMTrainer,
    )
    from kata.core.config import KATAConfig
    from kata.core.tokenizer import StateTokenizer
    from kata.env import KataEnv
    from kata.EntityFactories import RandomScenarioSampler
    from kata.scenario import ScenarioBuilder

    # -- env --
    env_path = Path(args.env)
    cfg = KATAConfig(**json.loads(env_path.read_text()))
    rcfg = cfg.randomized_scenario
    if rcfg.enabled:
        sampler = RandomScenarioSampler(cfg, rcfg, seed=args.seed)
        machine_types = sampler.all_machine_types()
        component_types = sampler.all_component_types()
        n_techs = rcfg.n_technicians

        def factory():
            return ScenarioBuilder(sampler.sample_config()).build()
    else:
        machine_types = sorted({m.machine_type for m in cfg.machines.values()})
        component_types = sorted(
            {
                c.component_type
                for m in cfg.machines.values()
                for c in m.components.values()
            }
        )
        n_techs = len(cfg.technicians)

        def factory():
            return ScenarioBuilder(cfg).build()

    tokenizer = StateTokenizer.build_vocab(
        machine_types=machine_types,
        n_technicians=n_techs,
        seq_length=cfg.gym.tokenizer_seq_length,
        component_types=component_types,
        next_ticket_lookahead=cfg.gym.next_ticket_lookahead,
    )
    target_repr = cfg.gym.observation_representation
    if target_repr not in {"hybrid", "token_ids"}:
        target_repr = "token_ids"
    gym_cfg = cfg.gym.model_copy(update={"observation_representation": target_repr})
    hybrid_mode = target_repr == "hybrid"
    with _quiet_stdout():
        env = KataEnv(scenario_factory=factory, config=gym_cfg, tokenizer=tokenizer)

    logger.info(
        "Env ready — vocab=%d, seq_len=%d, n_techs=%d",
        tokenizer.vocab_size,
        gym_cfg.tokenizer_seq_length,
        n_techs,
    )

    # -- collect --
    collector_cls = {
        "random": RandomAgent,
        "round_robin": RoundRobinAgent,
        "least_busy": LeastBusyAgent,
        "least_fatigued": LeastFatiguedAgent,
        "shortest_queue": ShortestQueueAgent,
    }[args.collector]
    collector = collector_cls(n_actions=n_techs)
    logger.info("Collecting %d token-obs steps with %s...", args.n_rollouts, args.collector)
    with _quiet_stdout():
        buf = collect_token_rollouts(env, collector, n_steps=args.n_rollouts, seed=args.seed)
    seqs = buf.as_array()
    if isinstance(seqs, dict):
        n_seqs = len(seqs["token_ids"])
        seq_len = int(seqs["token_ids"].shape[1])
        non_pad = float(np.mean((seqs["token_ids"] != 0).sum(axis=1)))
    else:
        n_seqs = len(seqs)
        seq_len = int(seqs.shape[1])
        non_pad = float(np.mean((seqs != 0).sum(axis=1)))
    logger.info(
        "Collected %d sequences of length %d (avg non-pad tokens: %.1f)",
        n_seqs,
        seq_len,
        non_pad,
    )

    # -- train --
    model = MaskedTokenModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=cfg.gym.tokenizer_seq_length,
        dropout=args.dropout,
        hybrid=hybrid_mode,
        sim_time_scale=float(cfg.gym.max_sim_time),
    )
    train_cfg = MTMTrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_prob=args.mask_prob,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        device=args.device,
    )
    trainer = MTMTrainer(model, vocab_size=tokenizer.vocab_size, cfg=train_cfg)
    logger.info(
        "Training MLM: epochs=%d, batch=%d, lr=%g, mask_prob=%g, device=%s",
        train_cfg.n_epochs,
        train_cfg.batch_size,
        train_cfg.lr,
        train_cfg.mask_prob,
        trainer.device,
    )
    history = trainer.fit(seqs)
    if history:
        last = history[-1]
        logger.info(
            "Final  loss=%.3f  mlm_accuracy=%.3f  (%d log-steps total)",
            last["loss"],
            last["mlm_accuracy"],
            len(history),
        )

    out_path = Path(args.out)
    trainer.save(
        out_path,
        extras={
            "env_config": str(env_path),
            "collector": args.collector,
            "n_rollouts": args.n_rollouts,
            "vocab_size": tokenizer.vocab_size,
            "machine_types": machine_types,
            "component_types": component_types,
            "n_techs": n_techs,
        },
    )
    logger.info("Saved MLM checkpoint -> %s", out_path)


if __name__ == "__main__":
    main()
