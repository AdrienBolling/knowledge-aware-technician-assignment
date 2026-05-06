"""Configuration models for the experiment framework.

Three JSON files drive an experiment:

1. **Environment config** — loaded directly as a ``KATAConfig`` (reuses the
   existing config schema under ``run_configs/``).
2. **Agent config** — specifies the agent type and its constructor kwargs.
3. **Experiment config** — experiment mode, training/eval parameters, logging,
   and checkpoint settings.

Experiment modes
----------------
- ``train``  — train an agent from scratch, optionally with periodic eval.
- ``eval``   — evaluate a pre-trained checkpoint (no weight updates).
- ``evaluated_training`` — train with interleaved evaluation rounds, each
  logged as a separate W&B run in the same group.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ======================================================================
# Agent configuration
# ======================================================================


class AgentConfig(BaseModel):
    """Which agent to instantiate and with what hyperparameters.

    ``agent_type`` selects the class from the agent registry.  All
    remaining fields in ``params`` are forwarded as keyword arguments
    to the constructor (``n_actions`` is injected automatically by the
    experiment runner).
    """

    agent_type: Literal[
        "random",
        "round_robin",
        "least_busy",
        "least_fatigued",
        "shortest_queue",
        "rainbow_dqn",
        "grpo",
    ] = Field(description="Agent class to instantiate.")

    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor keyword arguments forwarded to the agent.",
    )

    checkpoint: str | None = Field(
        default=None,
        description="Path to a checkpoint file to load before running.",
    )


# ======================================================================
# Experiment configuration
# ======================================================================


class EvalConfig(BaseModel):
    """Settings for evaluation (standalone or periodic during training)."""

    enabled: bool = Field(default=True, description="Run evaluation episodes.")
    interval: int = Field(
        default=10,
        gt=0,
        description="Evaluate every N training episodes.",
    )
    n_episodes: int = Field(
        default=5,
        gt=0,
        description="Number of evaluation episodes per evaluation round.",
    )
    deterministic: bool = Field(
        default=True,
        description="Use deterministic policy during evaluation.",
    )


class CheckpointConfig(BaseModel):
    """Settings for model checkpointing."""

    enabled: bool = Field(default=True, description="Save agent checkpoints.")
    interval: int = Field(
        default=50,
        gt=0,
        description="Checkpoint every N training episodes.",
    )
    dir: str = Field(
        default="checkpoints",
        description="Directory for saving checkpoints (relative to CWD).",
    )
    save_best: bool = Field(
        default=True,
        description="Additionally save the best model by mean eval reward.",
    )


class WandbConfig(BaseModel):
    """Weights & Biases logging settings."""

    enabled: bool = Field(default=True, description="Log to W&B.")
    project: str = Field(
        default="kata-experiments",
        description="W&B project name.",
    )
    entity: str | None = Field(
        default="bolling-adrien",
        description="W&B entity (team or user). None = default entity.",
    )
    group: str | None = Field(
        default=None,
        description=(
            "W&B run group. Auto-generated from agent type + seed when None. "
            "In evaluated_training mode, train and eval runs share the same group."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Extra tags attached to the W&B run. Auto-tags (agent type, "
            "experiment mode, obs representation, …) are always appended."
        ),
    )
    name: str | None = Field(
        default=None,
        description="Run display name. None = auto-generated.",
    )
    log_interval: int = Field(
        default=1,
        gt=0,
        description="Log training metrics every N episodes.",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment settings."""

    mode: Literal["train", "eval", "evaluated_training"] = Field(
        default="train",
        description=(
            "Experiment type. "
            "'train' = training only, "
            "'eval' = evaluate a checkpoint, "
            "'evaluated_training' = train with interleaved eval runs."
        ),
    )
    seed: int = Field(default=42, description="Global random seed.")
    n_episodes: int = Field(
        default=200,
        gt=0,
        description="Total number of training episodes (ignored in eval mode).",
    )
    log_interval: int = Field(
        default=10,
        gt=0,
        description="Print training progress every N episodes to stdout.",
    )

    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# ======================================================================
# Loader
# ======================================================================


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dict."""
    with open(path) as f:
        return json.load(f)
