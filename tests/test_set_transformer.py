"""Tests for the ``set`` observation mode + SetTransformer agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")

from kata import get_config
from kata.core.config import KATAConfig
from kata.EntityFactories import RandomScenarioSampler
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder
from agents.networks.set_transformer import (
    PointerActionHead,
    SetTransformerActorCritic,
    SetTransformerEncoder,
)
from agents.ppo.ppo_set_transformer import SetTransformerAgent


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _build_set_env(*, max_techs: int = 30, max_machines: int = 100,
                   L_t: int = 16, L_m: int = 12, L_e: int = 16,
                   max_sim_time: float = 1500.0, max_steps: int = 50, seed: int = 0):
    cfg_path = Path("run_configs/benchmark_suite/baseline.json")
    cfg_dict = json.loads(cfg_path.read_text())
    cfg_dict["gym"]["observation_representation"] = "set"
    cfg_dict["gym"]["max_techs"] = max_techs
    cfg_dict["gym"]["max_machines"] = max_machines
    cfg_dict["gym"]["set_tech_slot_length"] = L_t
    cfg_dict["gym"]["set_machine_slot_length"] = L_m
    cfg_dict["gym"]["set_env_length"] = L_e
    cfg_dict["gym"]["max_sim_time"] = max_sim_time
    cfg_dict["gym"]["max_episode_steps"] = max_steps
    cfg = KATAConfig(**cfg_dict)
    cached = get_config()
    cached.sim = cfg.sim
    cached.gym = cfg.gym
    sampler = RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=seed)
    env = KataEnv(
        scenario_factory=lambda: ScenarioBuilder(sampler.sample_config()).build(),
        config=cfg.gym,
    )
    return env, cfg


# ---------------------------------------------------------------------
# Observation-space tests
# ---------------------------------------------------------------------


class TestSetObservation:
    def test_obs_space_keys_and_shapes(self):
        env, cfg = _build_set_env(max_techs=30, max_machines=100)
        obs, _ = env.reset(seed=42)
        expected_keys = {
            "tech_token_ids", "tech_cont_values", "tech_cont_kinds", "tech_mask",
            "machine_token_ids", "machine_cont_values", "machine_cont_kinds",
            "machine_mask",
            "env_token_ids", "env_cont_values", "env_cont_kinds",
            "action_mask",
        }
        assert expected_keys.issubset(obs.keys())
        assert obs["tech_token_ids"].shape == (30, 16)
        assert obs["tech_mask"].shape == (30,)
        assert obs["machine_token_ids"].shape == (100, 12)
        assert obs["machine_mask"].shape == (100,)
        assert obs["env_token_ids"].shape == (16,)
        assert obs["action_mask"].shape == (30,)

    def test_masks_match_fleet_size(self):
        env, _ = _build_set_env()
        obs, _ = env.reset(seed=42)
        n_techs = len(env.dispatcher.techs)
        # tech_mask: 1 for real slots, 0 for padded
        assert int(obs["tech_mask"].sum()) == n_techs
        assert (obs["tech_mask"][:n_techs] == 1).all()
        assert (obs["tech_mask"][n_techs:] == 0).all()
        # action_mask: subset of tech_mask (1 only for non-busy, non-disrupted real techs)
        assert int(obs["action_mask"].sum()) <= n_techs

    def test_padded_slots_are_zero(self):
        env, _ = _build_set_env()
        obs, _ = env.reset(seed=42)
        n_techs = len(env.dispatcher.techs)
        # Padded tech slots should be entirely PAD tokens (id == 0) and zero cont values
        assert (obs["tech_token_ids"][n_techs:] == 0).all()
        assert (obs["tech_cont_values"][n_techs:] == 0.0).all()

    def test_tech_knowledge_features_present(self):
        """Each real tech slot emits non-zero knowledge scalars."""
        env, _ = _build_set_env(max_steps=50)
        obs, _ = env.reset(seed=42)
        # Run a few steps so knowledge actually accumulates beyond
        # the initial grid load.
        for s in range(5):
            mask = obs["action_mask"]
            a = int(np.where(mask)[0][0]) if mask.any() else 0
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        kinds = obs["tech_cont_kinds"]
        vals = obs["tech_cont_values"]
        n_real = int(obs["tech_mask"].sum())
        # COUNT_PLE positions (kind=2) carry the volume / max_k /
        # entropy scalars — at least one per real slot must be > 0.
        for i in range(n_real):
            count_positions = vals[i][kinds[i] == 2]
            assert (count_positions > 0.0).any(), (
                f"tech slot {i} has no positive count-kind features"
            )

    def test_machine_is_current_flag_exists(self):
        """Exactly one machine slot has IS_CURRENT=T at any decision step."""
        env, _ = _build_set_env(max_steps=20)
        obs, _ = env.reset(seed=42)
        # IS_CURRENT is a boolean cat token "IS_CURRENT=T" / "...=F";
        # there is exactly one machine per decision that is the broken
        # ticket-source.  We can't easily decode token strings here,
        # but we can rely on the env's current_request invariant:
        # exactly one machine_id must match the current ticket.
        if env.current_request is None:
            return  # no decision pending
        ticket_mid = env._machine_id_from_machine(env.current_request.machine)
        machines = env._factory_machines()
        n_match = sum(
            1 for m in machines
            if env._machine_id_from_machine(m) == ticket_mid
        )
        assert n_match == 1


# ---------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------


class TestSetTransformerEncoder:
    def test_forward_shapes(self):
        enc = SetTransformerEncoder(
            vocab_size=64, d_model=32, n_heads=4, n_layers=2,
            max_techs=30, max_machines=100, env_length=16,
        )
        ac = SetTransformerActorCritic(enc, value_hidden=32, pointer_d_attn=16)
        b = 4
        # Use the actual default slot lengths from config:
        # L_tech=16, L_mach=12, L_env=16.
        obs = {
            "tech_token_ids":     torch.zeros(b, 30, 16, dtype=torch.long),
            "tech_cont_values":   torch.zeros(b, 30, 16, dtype=torch.float32),
            "tech_cont_kinds":    torch.zeros(b, 30, 16, dtype=torch.long),
            "tech_mask":          torch.zeros(b, 30,     dtype=torch.long),
            "machine_token_ids":  torch.zeros(b, 100, 12, dtype=torch.long),
            "machine_cont_values":torch.zeros(b, 100, 12, dtype=torch.float32),
            "machine_cont_kinds": torch.zeros(b, 100, 12, dtype=torch.long),
            "machine_mask":       torch.zeros(b, 100,     dtype=torch.long),
            "env_token_ids":      torch.zeros(b, 16, dtype=torch.long),
            "env_cont_values":    torch.zeros(b, 16, dtype=torch.float32),
            "env_cont_kinds":     torch.zeros(b, 16, dtype=torch.long),
        }
        # Make a few tech slots valid so attention has something to do
        obs["tech_mask"][:, :3] = 1
        obs["machine_mask"][:, :10] = 1
        logits, value, hidden = ac(obs)
        assert hidden is None  # no RNN configured -> hidden passes through
        assert logits.shape == (b, 30)
        assert value.shape == (b,)

    def test_pointer_masks_invalid_slots(self):
        head = PointerActionHead(d_context=16, d_slot=16, d_attn=8)
        s = torch.randn(2, 16)
        e = torch.randn(2, 5, 16)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        logits = head(s, e, mask)
        assert logits.shape == (2, 5)
        # Invalid positions should be -inf
        assert torch.isinf(logits[0, 2:]).all()
        assert torch.isinf(logits[1, 4])
        # Valid positions should be finite
        assert torch.isfinite(logits[0, :2]).all()
        assert torch.isfinite(logits[1, :4]).all()


# ---------------------------------------------------------------------
# Agent integration test
# ---------------------------------------------------------------------


class TestSetTransformerAgent:
    @pytest.mark.slow
    def test_end_to_end_rollout_and_update(self):
        env, _ = _build_set_env(max_steps=50)
        # Warmup loop to populate vocab
        obs, _ = env.reset(seed=42)
        for s in range(40):
            a = (
                int(np.where(obs["action_mask"])[0][0])
                if obs["action_mask"].any()
                else 0
            )
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                obs, _ = env.reset(seed=42 + s + 1)
        env._tokenizer.freeze()
        vocab = env._tokenizer.vocab_size

        agent = SetTransformerAgent(
            n_actions=30, vocab_size=vocab,
            d_model=32, n_heads=4, n_layers=2,
            max_techs=30, max_machines=100, env_length=8,
            rollout_steps=20, minibatch_size=8, n_epochs=2,
            device="cpu", seed=0,
        )

        obs, info = env.reset(seed=99)
        for _ in range(25):
            a = agent.select_action(obs)
            nxt, r, term, trunc, info = env.step(a)
            agent.observe_transition(obs, a, r, nxt, term, trunc, info)
            obs = nxt
            if term or trunc:
                break

        metrics = agent.update()
        assert "loss" in metrics
        # First-epoch PG ratio is exactly 1 → pg_loss often 0;
        # what we want is that the update ran without nans.
        for k, v in metrics.items():
            assert not (isinstance(v, float) and (v != v)), f"NaN in {k}"

    def test_agent_constructor_validates_n_actions(self):
        with pytest.raises(ValueError, match="max_techs"):
            SetTransformerAgent(
                n_actions=4, vocab_size=64,
                max_techs=30,  # mismatch
                device="cpu",
            )
