"""Tests for the PPOTransformerAgent and ModernTransformerEncoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from agents.networks.modern_transformer import ModernTransformerEncoder
from agents.ppo.ppo_transformer import PPOTransformerAgent


# ---------------------------------------------------------------------------
# ModernTransformerEncoder
# ---------------------------------------------------------------------------


class TestModernTransformerEncoder:
    def test_forward_shape_and_dtype(self):
        torch.manual_seed(0)
        enc = ModernTransformerEncoder(
            vocab_size=64, d_model=64, n_heads=4, n_layers=2, max_seq_len=16
        )
        ids = torch.randint(1, 64, (3, 12), dtype=torch.long)
        out = enc(ids)
        assert out.shape == (3, 64)
        assert out.dtype == torch.float32

    def test_pad_mask_isolates_pad_tokens(self):
        """A sequence of all-PAD tokens still produces a finite output."""
        enc = ModernTransformerEncoder(
            vocab_size=32, d_model=32, n_heads=4, n_layers=2, max_seq_len=8,
            pad_token_id=0,
        )
        ids = torch.zeros(2, 8, dtype=torch.long)  # all padding
        out = enc(ids)
        assert torch.isfinite(out).all()

    def test_param_count_grows_with_depth(self):
        small = ModernTransformerEncoder(vocab_size=32, d_model=64, n_heads=4, n_layers=2, max_seq_len=8)
        big = ModernTransformerEncoder(vocab_size=32, d_model=64, n_heads=4, n_layers=6, max_seq_len=8)
        n_small = sum(p.numel() for p in small.parameters())
        n_big = sum(p.numel() for p in big.parameters())
        assert n_big > n_small


# ---------------------------------------------------------------------------
# PPOTransformerAgent
# ---------------------------------------------------------------------------


def _make_agent(**overrides):
    defaults = dict(
        n_actions=4,
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        head_hidden_dim=32,
        max_seq_len=16,
        dropout=0.0,
        rollout_steps=32,
        n_epochs=2,
        minibatch_size=8,
        total_updates=4,
        warmup_updates=1,
        device="cpu",
        seed=0,
    )
    defaults.update(overrides)
    return PPOTransformerAgent(**defaults)


class TestPPOTransformerAgent:
    def _stub_obs(self, agent: PPOTransformerAgent) -> dict:
        rng = np.random.default_rng(0)
        return {
            "token_ids": rng.integers(
                1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64
            )
        }

    def test_select_action_returns_valid_index(self):
        agent = _make_agent()
        obs = self._stub_obs(agent)
        a = agent.select_action(obs)
        assert isinstance(a, int)
        assert 0 <= a < agent.n_actions

    def test_deterministic_action_is_stable(self):
        agent = _make_agent()
        obs = self._stub_obs(agent)
        a1 = agent.select_action(obs, deterministic=True)
        a2 = agent.select_action(obs, deterministic=True)
        assert a1 == a2

    def test_update_runs_and_emits_metrics(self):
        agent = _make_agent()
        rng = np.random.default_rng(1)
        for _ in range(agent.rollout_steps):
            obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            a = agent.select_action(obs)
            next_obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            agent.observe_transition(obs, a, float(rng.normal()), next_obs, False, False, {})
        metrics = agent.update()
        for key in ("loss", "pg_loss", "vf_loss", "entropy", "approx_kl", "clip_fraction", "lr"):
            assert key in metrics
            assert np.isfinite(metrics[key]) or key == "early_stop"
        # Buffers reset
        assert agent._obs_buffer == []

    def test_save_load_round_trip_preserves_logits(self):
        agent = _make_agent()
        rng = np.random.default_rng(2)
        for _ in range(agent.rollout_steps):
            obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            a = agent.select_action(obs)
            next_obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            agent.observe_transition(obs, a, float(rng.normal()), next_obs, False, False, {})
        agent.update()

        # Eval mode disables dropout so the comparison is exact.
        agent.net.eval()
        ids = torch.from_numpy(self._stub_obs(agent)["token_ids"]).unsqueeze(0)
        with torch.no_grad():
            ref_logits, _ = agent.net(ids)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            agent.save(path)
            agent2 = _make_agent()
            agent2.load(path)
            agent2.net.eval()
            with torch.no_grad():
                new_logits, _ = agent2.net(ids)
        assert torch.allclose(ref_logits, new_logits, atol=1e-6)

    def test_target_kl_early_stop(self):
        # Aggressive setup: a single, tiny rollout pumped through many
        # epochs should typically trip target_kl early-stop.
        agent = _make_agent(target_kl=1e-9, n_epochs=8, rollout_steps=16, minibatch_size=4)
        rng = np.random.default_rng(3)
        for _ in range(agent.rollout_steps):
            obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            a = agent.select_action(obs)
            next_obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            agent.observe_transition(obs, a, float(rng.normal()), next_obs, False, False, {})
        metrics = agent.update()
        # ``early_stop`` is 1.0 when KL exceeded the threshold during the loop.
        assert metrics["early_stop"] == 1.0

    def test_lr_schedule_steps_each_update(self):
        agent = _make_agent(total_updates=10, warmup_updates=2, lr=1e-3)
        lr_before = agent.optimizer.param_groups[0]["lr"]
        # Push a small rollout and update
        rng = np.random.default_rng(4)
        for _ in range(agent.rollout_steps):
            obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            a = agent.select_action(obs)
            next_obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            agent.observe_transition(obs, a, float(rng.normal()), next_obs, False, False, {})
        agent.update()
        lr_after = agent.optimizer.param_groups[0]["lr"]
        assert lr_after != lr_before  # scheduler advanced

    def test_default_size_reaches_hundreds_of_thousands_of_params(self):
        # Constructor defaults — no overrides — should produce the
        # advertised "complex and big" architecture.
        agent = PPOTransformerAgent(n_actions=4, vocab_size=512, device="cpu")
        n = agent.num_parameters()
        assert n > 500_000

    def test_action_mask_forbids_invalid_choice(self):
        """Stochastic and deterministic sampling never pick masked-out actions."""
        agent = _make_agent(n_actions=4, use_action_mask=True)
        rng = np.random.default_rng(7)
        token_ids = rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)
        # Mask out actions 0 and 2 → only 1 and 3 are valid
        mask = np.array([0, 1, 0, 1], dtype=np.int8)
        obs = {"token_ids": token_ids, "action_mask": mask}
        for _ in range(50):
            a = agent.select_action(obs)
            assert a in (1, 3)
        # Deterministic path also respects the mask
        for _ in range(5):
            a = agent.select_action(obs, deterministic=True)
            assert a in (1, 3)

    def test_action_mask_fallback_when_all_invalid(self):
        """A degenerate all-zero mask shouldn't crash — fall back to all-valid."""
        agent = _make_agent(n_actions=3, use_action_mask=True)
        rng = np.random.default_rng(8)
        obs = {
            "token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64),
            "action_mask": np.zeros(3, dtype=np.int8),
        }
        a = agent.select_action(obs)
        assert 0 <= a < 3

    def test_mask_stored_and_applied_in_update(self):
        """Buffered masks must be respected during the policy update."""
        agent = _make_agent(n_actions=4, use_action_mask=True, rollout_steps=16, minibatch_size=4)
        rng = np.random.default_rng(9)
        # Use a fixed restrictive mask the whole rollout so update sees it
        mask = np.array([0, 1, 1, 0], dtype=np.int8)
        for _ in range(agent.rollout_steps):
            obs = {
                "token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64),
                "action_mask": mask,
            }
            a = agent.select_action(obs)
            assert a in (1, 2)
            next_obs = {
                "token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64),
                "action_mask": mask,
            }
            agent.observe_transition(obs, a, float(rng.normal()), next_obs, False, False, {})
        # Buffer should have stored masks alongside the actions
        assert len(agent._mask_buffer) == agent.rollout_steps
        metrics = agent.update()
        # Buffers reset; metrics finite
        assert agent._mask_buffer == []
        assert np.isfinite(metrics["loss"])

    def test_normalize_rewards_changes_buffered_rewards(self):
        """With ``normalize_rewards=True`` the buffered reward differs from the raw value."""
        agent = _make_agent(normalize_rewards=True)
        rng = np.random.default_rng(10)
        # Feed in 1.0 rewards — running std will diverge from 1 quickly
        raw_rewards: list[float] = []
        for _ in range(40):
            obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            a = agent.select_action(obs)
            next_obs = {"token_ids": rng.integers(1, agent.vocab_size, size=agent.max_seq_len, dtype=np.int64)}
            agent.observe_transition(obs, a, 1.0, next_obs, False, False, {})
            raw_rewards.append(1.0)
        # At least one buffered reward should differ noticeably from 1.0
        diffs = [abs(b - 1.0) for b in agent._reward_buffer]
        assert max(diffs) > 1e-3
        # The running-return statistics should be initialised away from defaults
        assert agent._return_rms.count > 1
        assert agent._return_rms.var != 1.0 or agent._return_rms.mean != 0.0
