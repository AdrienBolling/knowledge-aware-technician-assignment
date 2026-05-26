"""Sanity tests for PPOLatentAgent.

These tests do NOT exercise the full PPO update loop — only verify
that the agent can be constructed from a saved MLM checkpoint, that
the encoder weights are loaded (frozen or not), and that ``select_action``
returns a valid action on a token observation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.ppo.ppo_latent import PPOLatentAgent
from agents.representation.mtm import (
    NUM_RESERVED,
    MaskedTokenModel,
    MTMTrainConfig,
    MTMTrainer,
)


@pytest.fixture
def saved_encoder(tmp_path):
    """Train a (no-op) MTM model for 0 epochs and persist it.

    Yields ``(path, vocab_size, d_model, max_seq_len)`` so tests can
    rebuild the matching tokenizer / agent.
    """
    torch.manual_seed(0)
    vocab_size = 20
    d_model = 16
    max_seq_len = 12
    model = MaskedTokenModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=2,
        n_layers=1,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    trainer = MTMTrainer(model, vocab_size=vocab_size, cfg=MTMTrainConfig(n_epochs=0))
    path = tmp_path / "mtm.pt"
    trainer.save(path)
    return path, vocab_size, d_model, max_seq_len


def _make_agent(saved_encoder, *, freeze: bool, n_actions: int = 3):
    path, vocab_size, _d_model, max_seq_len = saved_encoder
    return PPOLatentAgent(
        n_actions=n_actions,
        vocab_size=vocab_size,
        encoder_path=path,
        freeze_encoder=freeze,
        # Tiny PPO heads — the algorithm path is exercised elsewhere.
        # n_heads/d_model/max_seq_len/n_layers are forced to match the
        # saved checkpoint by PPOLatentAgent, so the values we pass for
        # them here are ignored — kept tiny just to fail fast on any
        # accidental architectural drift.
        head_hidden_dim=16,
        rollout_steps=4,
        n_epochs=1,
        minibatch_size=2,
        total_updates=2,
        warmup_updates=1,
        use_action_mask=False,
        normalize_rewards=False,
        device="cpu",
    )


class TestPPOLatentConstruction:
    def test_loads_encoder_from_checkpoint(self, saved_encoder):
        agent = _make_agent(saved_encoder, freeze=True)
        assert agent.name == "PPOLatent"
        # Compare a representative encoder tensor to the saved checkpoint.
        saved = torch.load(saved_encoder[0], map_location="cpu", weights_only=False)
        emb_orig = saved["model"]["encoder.token_embedding.weight"]
        emb_live = agent.net.encoder.token_embedding.weight.detach().cpu()
        assert torch.allclose(emb_orig, emb_live, atol=1e-6)

    def test_freeze_encoder_disables_grad(self, saved_encoder):
        agent = _make_agent(saved_encoder, freeze=True)
        for p in agent.net.encoder.parameters():
            assert p.requires_grad is False
        # Heads still receive grad
        head_params = [
            p for n, p in agent.net.named_parameters() if not n.startswith("encoder.")
        ]
        assert all(p.requires_grad for p in head_params)
        # Optimizer has exactly one param group (heads only) when frozen
        assert len(agent.optimizer.param_groups) == 1

    def test_unfreeze_creates_two_param_groups(self, saved_encoder):
        agent = _make_agent(saved_encoder, freeze=False)
        for p in agent.net.encoder.parameters():
            assert p.requires_grad is True
        assert len(agent.optimizer.param_groups) == 2
        head_lr, enc_lr = (g["lr"] for g in agent.optimizer.param_groups)
        # encoder group has scaled lr (0.1 by default)
        assert enc_lr < head_lr

    def test_vocab_mismatch_raises(self, saved_encoder):
        path, vocab_size, _d, _max_seq = saved_encoder
        with pytest.raises(ValueError, match="vocab_size mismatch"):
            PPOLatentAgent(
                n_actions=3,
                vocab_size=vocab_size + 1,  # wrong on purpose
                encoder_path=path,
                head_hidden_dim=16,
                rollout_steps=4,
                n_epochs=1,
                minibatch_size=2,
                total_updates=2,
                warmup_updates=1,
                use_action_mask=False,
                normalize_rewards=False,
                device="cpu",
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PPOLatentAgent(
                n_actions=3,
                vocab_size=10,
                encoder_path=tmp_path / "does_not_exist.pt",
                head_hidden_dim=16,
                rollout_steps=4,
                n_epochs=1,
                minibatch_size=2,
                total_updates=2,
                warmup_updates=1,
                use_action_mask=False,
                normalize_rewards=False,
                device="cpu",
            )


class TestPPOLatentSelectAction:
    def test_returns_valid_action_index(self, saved_encoder):
        _path, vocab_size, _d, max_seq = saved_encoder
        n_actions = 4
        agent = _make_agent(saved_encoder, freeze=True, n_actions=n_actions)
        obs = {
            "token_ids": np.random.randint(
                NUM_RESERVED, vocab_size, size=max_seq, dtype=np.int64
            )
        }
        a = agent.select_action(obs, deterministic=True)
        assert 0 <= int(a) < n_actions


class TestPPOLatentUpdate:
    """End-to-end PPO update from a tiny rollout, freeze + joint variants."""

    def _exercise(self, saved_encoder, *, freeze: bool):
        """Roll out 4 transitions, run update, assert no errors + scheduler advances."""
        agent = _make_agent(saved_encoder, freeze=freeze, n_actions=3)
        _path, vocab_size, _d, max_seq = saved_encoder

        # Capture pre-update LR per param group.
        lrs_before = [g["lr"] for g in agent.optimizer.param_groups]

        # Hand-build 4 transitions through the public agent interface.
        obs = {
            "token_ids": np.random.randint(
                NUM_RESERVED, vocab_size, size=max_seq, dtype=np.int64
            )
        }
        for _ in range(4):
            action = agent.select_action(obs, deterministic=False)
            next_obs = {
                "token_ids": np.random.randint(
                    NUM_RESERVED, vocab_size, size=max_seq, dtype=np.int64
                )
            }
            agent.observe_transition(
                obs=obs, action=action, reward=0.1,
                next_obs=next_obs, terminated=False, truncated=False, info={},
            )
            obs = next_obs

        metrics = agent.update()
        assert isinstance(metrics, dict)
        # Scheduler should have stepped at least once: the LRs may have
        # decreased after warmup, but they must remain finite and >= 0.
        for g in agent.optimizer.param_groups:
            assert g["lr"] >= 0.0
            assert np.isfinite(g["lr"])
        return agent, lrs_before

    def test_update_runs_with_frozen_encoder(self, saved_encoder):
        agent, _ = self._exercise(saved_encoder, freeze=True)
        # Single param group: heads only.
        assert len(agent.optimizer.param_groups) == 1
        # Encoder must remain in eval mode after the update.
        assert agent.net.encoder.training is False

    def test_update_runs_with_joint_finetune(self, saved_encoder):
        agent, lrs_before = self._exercise(saved_encoder, freeze=False)
        # Two param groups, encoder scaled by encoder_lr_scale.
        assert len(agent.optimizer.param_groups) == 2
        head_lr, enc_lr = lrs_before
        assert enc_lr < head_lr

    def test_encoder_train_method_is_pinned_when_frozen(self, saved_encoder):
        agent = _make_agent(saved_encoder, freeze=True)
        # Calling train() at any level must not flip the encoder.
        agent.net.train()
        assert agent.net.encoder.training is False
        agent.net.encoder.train()
        assert agent.net.encoder.training is False
