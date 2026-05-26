#!/usr/bin/env python
"""Standalone encoder pretraining for KATA.

A self-contained script that:

1. **Collects** a large pool of token observations from a KATA env by
   running a ``RandomAgent`` (broad state coverage — heuristic
   collectors over-represent "sensible" regions and bias the
   pretraining distribution).
2. **Trains** a 6-layer transformer (``d_model=192``) with the same
   modern recipe used by ``PPOTransformerAgent`` — RMSNorm, SwiGLU,
   RoPE, CLS pooling — on a BERT-style masked-token-modelling
   objective.  Output is ~2.7M parameters by default.
3. **Reports** per-step loss + MLM accuracy as a PNG curve, and a
   per-token-id reconstruction breakdown so you can see exactly which
   parts of the observation alphabet the encoder learned vs which
   collapsed to a majority class.
4. **Saves** the checkpoint at ``<out>/encoder.pt`` ready for use by
   ``PPOLatentAgent`` (set ``encoder_path`` in
   ``run_configs/agents/ppo_latent.json``).

Usage
-----
::

    python scripts/pretrain_encoder.py \\
        --env run_configs/factory_long.json \\
        --out reports/encoder_v1 \\
        --n-rollouts 8000 \\
        --epochs 30

All knobs have sensible defaults; the file is designed to be edited
directly when you need behaviour the CLI flags don't cover (e.g.
mixing collectors, freezing parts of the architecture).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Anchor to the repo root so relative paths work regardless of CWD.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))


@contextmanager
def quiet_stdout() -> Any:
    """Suppress noisy SimPy prints from machines / routers / sources."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a transformer encoder on KATA token observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", required=True, help="Path to env JSON (KATAConfig).")
    parser.add_argument(
        "--out",
        default="reports/encoder_pretrain",
        help="Output directory.  Saved artefacts: encoder.pt, curve.png, "
        "per_token_report.csv, manifest.json.",
    )

    g_collect = parser.add_argument_group("data collection")
    g_collect.add_argument("--n-rollouts", type=int, default=8000)
    g_collect.add_argument(
        "--collector",
        default="random",
        choices=["random", "round_robin", "least_busy", "least_fatigued", "shortest_queue"],
        help="Policy used to drive the env during collection.",
    )
    g_collect.add_argument("--eval-fraction", type=float, default=0.1)

    g_arch = parser.add_argument_group(
        "architecture (defaults match PPOTransformerAgent → ~2.7M params)"
    )
    g_arch.add_argument("--d-model", type=int, default=192)
    g_arch.add_argument("--n-layers", type=int, default=6)
    g_arch.add_argument("--n-heads", type=int, default=6)
    g_arch.add_argument("--dropout", type=float, default=0.1)

    g_train = parser.add_argument_group("training")
    g_train.add_argument("--epochs", type=int, default=30)
    g_train.add_argument("--batch-size", type=int, default=64)
    g_train.add_argument("--lr", type=float, default=3e-4)
    g_train.add_argument("--mask-prob", type=float, default=0.15)
    g_train.add_argument("--warmup-steps", type=int, default=500)
    g_train.add_argument("--log-every", type=int, default=50)
    g_train.add_argument("--seed", type=int, default=0)
    g_train.add_argument("--device", default="auto")
    g_train.add_argument("--quiet", "-q", action="store_true")
    return parser.parse_args(argv)


def build_env(env_cfg_path: Path, seed: int):
    """Construct the env + tokenizer used for both collection and eval.

    Returns ``(env, tokenizer, vocab_size, n_techs, machine_types, component_types)``.
    """
    from kata.core.config import KATAConfig
    from kata.core.tokenizer import StateTokenizer
    from kata.env import KataEnv
    from kata.EntityFactories import RandomScenarioSampler
    from kata.scenario import ScenarioBuilder

    cfg = KATAConfig(**json.loads(env_cfg_path.read_text()))
    rcfg = cfg.randomized_scenario
    if rcfg.enabled:
        sampler = RandomScenarioSampler(cfg, rcfg, seed=seed)
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
    # Honour the env config's observation_representation: ``hybrid`` if the
    # user wants PLE / Time2Vec / Fourier continuous channels, otherwise
    # fall back to the legacy ``token_ids`` mode for back-compat.
    target_repr = cfg.gym.observation_representation
    if target_repr not in {"hybrid", "token_ids"}:
        target_repr = "token_ids"
    gym_cfg = cfg.gym.model_copy(update={"observation_representation": target_repr})
    with quiet_stdout():
        env = KataEnv(scenario_factory=factory, config=gym_cfg, tokenizer=tokenizer)
    return env, tokenizer, n_techs, machine_types, component_types


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")
    logger = logging.getLogger("pretrain_encoder")

    # Don't let an unrelated KATA_CONF_PATH leak in.
    os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

    import numpy as np

    from agents.representation.data import collect_token_rollouts
    from agents.representation.mtm import (
        MaskedTokenModel,
        MTMTrainConfig,
        MTMTrainer,
    )

    # ----- output dir -----------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    # ----- env + tokenizer ------------------------------------------------
    env_path = Path(args.env).resolve()
    env, tokenizer, n_techs, machine_types, component_types = build_env(
        env_path, seed=args.seed
    )
    hybrid_mode = env.config.observation_representation == "hybrid"
    logger.info(
        "Env ready — vocab=%d, seq_len=%d, n_techs=%d, machines=%d, components=%d, mode=%s",
        tokenizer.vocab_size,
        tokenizer.seq_length,
        n_techs,
        len(machine_types),
        len(component_types),
        "hybrid" if hybrid_mode else "token_ids",
    )

    # ----- collect token observations -------------------------------------
    from agents import (
        LeastBusyAgent,
        LeastFatiguedAgent,
        RandomAgent,
        RoundRobinAgent,
        ShortestQueueAgent,
    )

    collector_cls = {
        "random": RandomAgent,
        "round_robin": RoundRobinAgent,
        "least_busy": LeastBusyAgent,
        "least_fatigued": LeastFatiguedAgent,
        "shortest_queue": ShortestQueueAgent,
    }[args.collector]
    collector = collector_cls(n_actions=n_techs)
    logger.info(
        "Collecting %d token-obs steps with %s (this is the slow part)...",
        args.n_rollouts,
        args.collector,
    )
    t0 = time.time()
    with quiet_stdout():
        buf = collect_token_rollouts(
            env, collector, n_steps=args.n_rollouts, seed=args.seed
        )
    raw = buf.as_array()

    def _index_seqs(seqs, idx):
        if isinstance(seqs, dict):
            return {k: v[idx] for k, v in seqs.items()}
        return seqs[idx]

    def _seq_count(seqs) -> int:
        return len(seqs["token_ids"]) if isinstance(seqs, dict) else len(seqs)

    if isinstance(raw, dict):
        n_seqs = _seq_count(raw)
        seq_len = int(raw["token_ids"].shape[1])
        non_pad = float(np.mean((raw["token_ids"] != 0).sum(axis=1)))
    else:
        n_seqs = len(raw)
        seq_len = int(raw.shape[1])
        non_pad = float(np.mean((raw != 0).sum(axis=1)))
    logger.info(
        "Collected %d sequences in %.1fs (avg non-pad tokens: %.1f / %d)",
        n_seqs,
        time.time() - t0,
        non_pad,
        seq_len,
    )

    # ----- train / eval split --------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_seqs)
    n_eval = max(1, int(n_seqs * args.eval_fraction))
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
    train_seqs = _index_seqs(raw, train_idx)
    eval_seqs = _index_seqs(raw, eval_idx)
    logger.info("Split: train=%d, eval=%d", _seq_count(train_seqs), _seq_count(eval_seqs))

    # ----- build model ----------------------------------------------------
    model = MaskedTokenModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=tokenizer.seq_length,
        dropout=args.dropout,
        hybrid=hybrid_mode,
        sim_time_scale=float(env.config.max_sim_time),
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "MaskedTokenModel: %d params  (d_model=%d, n_layers=%d, n_heads=%d)",
        n_params,
        args.d_model,
        args.n_layers,
        args.n_heads,
    )

    # ----- train ---------------------------------------------------------
    train_cfg = MTMTrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_prob=args.mask_prob,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
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
    t0 = time.time()
    history = trainer.fit(train_seqs)
    train_dur = time.time() - t0
    if history:
        last = history[-1]
        logger.info(
            "Training finished in %.1fs — final loss=%.3f  mlm_accuracy=%.3f",
            train_dur,
            last["loss"],
            last["mlm_accuracy"],
        )

    # ----- held-out evaluation -------------------------------------------
    logger.info("Evaluating on held-out set (%d sequences)...", len(eval_seqs))
    report = trainer.evaluate(eval_seqs, n_passes=3)
    logger.info(
        "Held-out MLM accuracy: %.3f over %d predictions",
        report["mlm_accuracy"],
        report["n_predictions"],
    )

    # ----- save artefacts ------------------------------------------------
    ckpt_path = out_dir / "encoder.pt"
    trainer.save(
        ckpt_path,
        extras={
            "env_config": str(env_path),
            "collector": args.collector,
            "n_rollouts": int(args.n_rollouts),
            "eval_fraction": float(args.eval_fraction),
            "vocab_size": tokenizer.vocab_size,
            "machine_types": machine_types,
            "component_types": component_types,
            "n_techs": n_techs,
            "held_out_mlm_accuracy": float(report["mlm_accuracy"]),
        },
    )
    logger.info("Saved checkpoint -> %s", ckpt_path)

    # Curve PNG
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        accs = [h["mlm_accuracy"] for h in history]
        fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_l.plot(steps, losses, color="tab:blue")
        ax_l.set_ylabel("MLM loss")
        ax_l.grid(True, alpha=0.3)
        ax_a.plot(steps, accs, color="tab:orange")
        ax_a.axhline(
            report["mlm_accuracy"],
            color="red",
            linestyle="--",
            label=f"held-out: {report['mlm_accuracy']:.3f}",
        )
        ax_a.set_ylabel("MLM accuracy")
        ax_a.set_xlabel("training step")
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Encoder pretraining ({n_params:,} params, vocab={tokenizer.vocab_size})"
        )
        fig.tight_layout()
        curve_path = out_dir / "curve.png"
        fig.savefig(curve_path, dpi=120)
        plt.close(fig)
        logger.info("Saved curve  -> %s", curve_path)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("Could not save curve.png: %s", exc)

    # Per-token CSV (sorted by accuracy: hardest tokens first)
    csv_path = out_dir / "per_token_report.csv"
    rows = []
    for tid, stats in report["per_token_id"].items():
        rows.append(
            (
                int(tid),
                tokenizer.id_to_token(int(tid)),
                int(stats["total"]),
                float(stats["accuracy"]),
            )
        )
    rows.sort(key=lambda r: (r[3], -r[2]))  # ascending accuracy
    with csv_path.open("w") as f:
        f.write("token_id,token,n_evaluated,accuracy\n")
        for tid, tok, n, acc in rows:
            f.write(f"{tid},{tok},{n},{acc:.4f}\n")
    logger.info("Saved per-token report -> %s  (%d unique tokens)", csv_path, len(rows))

    # Manifest
    manifest = {
        "env_config": str(env_path),
        "out_dir": str(out_dir),
        "args": vars(args),
        "vocab_size": tokenizer.vocab_size,
        "n_params": int(n_params),
        "machine_types": machine_types,
        "component_types": component_types,
        "n_techs": n_techs,
        "hybrid": bool(hybrid_mode),
        "train_size": int(_seq_count(train_seqs)),
        "eval_size": int(_seq_count(eval_seqs)),
        "training_duration_seconds": float(train_dur),
        "final_train_loss": float(history[-1]["loss"]) if history else None,
        "final_train_accuracy": (
            float(history[-1]["mlm_accuracy"]) if history else None
        ),
        "held_out_mlm_accuracy": float(report["mlm_accuracy"]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Saved manifest -> %s", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
