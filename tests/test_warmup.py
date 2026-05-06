"""Tests for WarmupEnv and KataEnv MCA integration."""

import os

from kata.core.config import (
    GymEnvConfig,
    KATAConfig,
    MachineConfig,
    ProductConfig,
    TechnicianConfig,
)
from kata.core.tokenizer import StateTokenizer
from kata.entities.encoder.mca_encoder import MCAEncoder
from kata.env import KataEnv
from kata.env_warmup import WarmupEnv
from kata.scenario import ScenarioBuilder


def _isolated_config(**overrides) -> KATAConfig:
    old = os.environ.get("KATA_CONF_PATH")
    os.environ["KATA_CONF_PATH"] = "/dev/null/no_such_file.json"
    try:
        defaults = {
            "technicians": {
                "t0": TechnicianConfig(name="t0"),
                "t1": TechnicianConfig(name="t1"),
            },
            "machines": {"m0": MachineConfig(machine_type="generic", process_time=10)},
            "products": {
                "p0": ProductConfig(product_type="generic", route=["generic"])
            },
        }
        defaults.update(overrides)
        return KATAConfig(**defaults)
    finally:
        if old is None:
            os.environ.pop("KATA_CONF_PATH", None)
        else:
            os.environ["KATA_CONF_PATH"] = old


def _factory(cfg: KATAConfig):
    def _build():
        return ScenarioBuilder(cfg).build()

    return _build


class TestWarmupEnv:
    def test_warmup_runs_and_returns_encoder_tokenizer(self):
        cfg = _isolated_config(
            gym=GymEnvConfig(
                max_episode_steps=50,
                max_sim_time=500.0,
                observation_representation="tokens",
            ),
        )
        warmup = WarmupEnv(
            scenario_factory=_factory(cfg),
            config=cfg.gym,
            n_warmup_steps=30,
        )
        encoder, tokenizer = warmup.run()
        assert isinstance(encoder, MCAEncoder)
        assert isinstance(tokenizer, StateTokenizer)
        assert tokenizer.frozen

    def test_warmup_populates_tokenizer_vocab(self):
        cfg = _isolated_config(
            gym=GymEnvConfig(
                max_episode_steps=50,
                max_sim_time=500.0,
                observation_representation="tokens",
            ),
        )
        warmup = WarmupEnv(
            scenario_factory=_factory(cfg),
            config=cfg.gym,
            n_warmup_steps=20,
        )
        _, tokenizer = warmup.run()
        # Should have learned tokens beyond the 4 special ones
        assert tokenizer.vocab_size > 4


class TestKataEnvTokenIds:
    def test_token_ids_observation(self):
        cfg = _isolated_config(
            gym=GymEnvConfig(
                max_episode_steps=10,
                max_sim_time=200.0,
                observation_representation="token_ids",
                tokenizer_seq_length=32,
            ),
        )
        env = KataEnv(scenario_factory=_factory(cfg), config=cfg.gym)
        obs, info = env.reset()
        assert "token_ids" in obs
        assert obs["token_ids"].shape == (32,)

    def test_token_ids_with_mca_warmup(self):
        cfg = _isolated_config(
            gym=GymEnvConfig(
                max_episode_steps=10,
                max_sim_time=200.0,
                observation_representation="token_ids",
                use_mca_encoder=True,
                warmup_steps=20,
                tokenizer_seq_length=32,
            ),
        )
        env = KataEnv(scenario_factory=_factory(cfg), config=cfg.gym)
        obs, info = env.reset()
        assert "token_ids" in obs
        assert obs["token_ids"].shape == (32,)
        # Warmup should have run
        assert env._warmup_done

    def test_step_returns_token_ids(self):
        cfg = _isolated_config(
            gym=GymEnvConfig(
                max_episode_steps=10,
                max_sim_time=200.0,
                observation_representation="token_ids",
                tokenizer_seq_length=16,
            ),
        )
        env = KataEnv(scenario_factory=_factory(cfg), config=cfg.gym)
        obs, _ = env.reset()
        if env.current_request is not None:
            obs, reward, terminated, truncated, info = env.step(0)
            assert "token_ids" in obs
            assert obs["token_ids"].shape == (16,)
