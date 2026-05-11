"""Tests for the shared resolve_device helper and per-agent auto-detection."""

from __future__ import annotations

from unittest.mock import patch

import torch

from agents import GRPOAgent, PPOTransformerAgent, RainbowDQNAgent
from agents.base import resolve_device


class TestResolveDevice:
    def test_explicit_device_passes_through(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:0") == "cuda:0"
        assert resolve_device("mps") == "mps"

    def test_auto_prefers_cuda(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            assert resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_mps_when_no_cuda(self):
        class _MPSStub:
            @staticmethod
            def is_available() -> bool:
                return True

        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends, "mps", _MPSStub, create=True),
        ):
            assert resolve_device("auto") == "mps"

    def test_auto_falls_back_to_cpu(self):
        class _MPSStub:
            @staticmethod
            def is_available() -> bool:
                return False

        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends, "mps", _MPSStub, create=True),
        ):
            assert resolve_device("auto") == "cpu"


class TestAgentAutoDetect:
    """All three NN agents must default to ``auto`` and end up on a real device."""

    def test_ppo_defaults_to_auto(self):
        agent = PPOTransformerAgent(
            n_actions=2, vocab_size=16, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8, rollout_steps=4, minibatch_size=2,
            total_updates=2, warmup_updates=1,
        )
        # Whatever auto resolved to, it must be a real torch.device
        assert isinstance(agent.device, torch.device)
        assert agent.device.type in ("cpu", "cuda", "mps")

    def test_grpo_defaults_to_auto(self):
        agent = GRPOAgent(
            n_actions=2, vocab_size=16, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8,
        )
        assert isinstance(agent.device, torch.device)
        assert agent.device.type in ("cpu", "cuda", "mps")

    def test_rainbow_defaults_to_auto(self):
        agent = RainbowDQNAgent(
            n_actions=2, vocab_size=16, d_model=16, n_heads=2, n_layers=1,
            max_seq_len=8, min_replay_size=2,
        )
        assert isinstance(agent.device, torch.device)
        assert agent.device.type in ("cpu", "cuda", "mps")

    def test_explicit_cpu_override_respected(self):
        for cls, kw in (
            (PPOTransformerAgent, dict(
                vocab_size=16, d_model=16, n_heads=2, n_layers=1, max_seq_len=8,
                rollout_steps=4, minibatch_size=2, total_updates=2, warmup_updates=1,
            )),
            (GRPOAgent, dict(vocab_size=16, d_model=16, n_heads=2, n_layers=1, max_seq_len=8)),
            (RainbowDQNAgent, dict(
                vocab_size=16, d_model=16, n_heads=2, n_layers=1, max_seq_len=8,
                min_replay_size=2,
            )),
        ):
            agent = cls(n_actions=2, device="cpu", **kw)
            assert agent.device == torch.device("cpu")
