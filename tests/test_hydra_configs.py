"""Composition tests for the Hydra config tree.

The ``conf/env`` and ``conf/agent`` groups are symlinks onto the
canonical JSON configs, so these tests guard two invariants: every
group option still composes AND validates through the pydantic models
(catching drift between the JSONs and the models), and deep CLI
overrides reach the composed tree.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from kata.core.config import KATAConfig

CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")

ENV_GROUPS = [
    "baseline",
    "baseline_crit",
    "small_scale",
    "massive_scale",
    "train_multiscale",
]


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="train", overrides=overrides)


def test_every_env_group_composes_and_validates():
    for env in ENV_GROUPS:
        cfg = _compose([f"env={env}"])
        env_data = OmegaConf.to_container(cfg.env, resolve=True)
        KATAConfig(**env_data)  # pydantic is still the validator


def test_deep_override_reaches_composed_tree():
    cfg = _compose([
        "agent.params.gae_lambda=0.99",
        "env.gym.reward.fatigue_cost.coefficient=0.25",
    ])
    assert cfg.agent.params.gae_lambda == 0.99
    assert cfg.env.gym.reward.fatigue_cost.coefficient == 0.25


def test_trainer_defaults_mirror_launcher():
    cfg = _compose([])
    assert cfg.episodes == 2000
    assert cfg.eval_episodes == 5          # hardened selection default
    assert cfg.use_gru is False            # current-generation stack
    # null = respect the env config's reward stack (v5 sets PBRS itself)
    assert cfg.potential_knowledge_reward is None
