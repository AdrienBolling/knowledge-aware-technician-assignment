"""Tests for the masked-token-modelling encoder + trainer + data utils."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.networks.modern_transformer import ModernTransformerEncoder
from agents.representation.data import TokenObsBuffer
from agents.representation.mtm import (
    MASK_ID,
    NUM_RESERVED,
    PAD_ID,
    MaskedTokenModel,
    MTMTrainConfig,
    MTMTrainer,
    mlm_collate,
)


class TestEncoderEncode:
    """``ModernTransformerEncoder.encode`` returns both pool + per-token states."""

    def test_returns_cls_and_token_states(self):
        enc = ModernTransformerEncoder(vocab_size=20, d_model=16, n_layers=1, n_heads=2)
        ids = torch.randint(0, 20, (3, 7))
        cls, tokens = enc.encode(ids)
        assert cls.shape == (3, 16)
        assert tokens.shape == (3, 7, 16)

    def test_forward_matches_cls_slot(self):
        torch.manual_seed(0)
        enc = ModernTransformerEncoder(
            vocab_size=20, d_model=16, n_layers=1, n_heads=2, dropout=0.0
        )
        enc.eval()
        ids = torch.randint(0, 20, (2, 7))
        cls, _ = enc.encode(ids)
        pooled = enc.forward(ids)
        assert torch.allclose(pooled, cls, atol=1e-6)


class TestMLMCollate:
    """The masking helper produces the right shapes and respects the recipe."""

    def test_shapes(self):
        seqs = np.random.randint(NUM_RESERVED, 50, size=(8, 32)).astype(np.int64)
        inp, tgt, mask = mlm_collate(seqs, vocab_size=50, mask_prob=0.5)
        assert inp.shape == seqs.shape
        assert tgt.shape == seqs.shape
        assert mask.shape == seqs.shape

    def test_loss_positions_have_targets(self):
        seqs = np.random.randint(NUM_RESERVED, 50, size=(8, 32)).astype(np.int64)
        inp, tgt, mask = mlm_collate(seqs, vocab_size=50, mask_prob=0.5)
        # Where mask=True, targets carry the original token id (>=0)
        assert (tgt[mask] >= 0).all()
        # Where mask=False, targets are -100 (ignored by CE loss)
        assert (tgt[~mask] == -100).all()

    def test_specials_are_never_masked(self):
        seqs = np.zeros((4, 8), dtype=np.int64)
        seqs[:, :NUM_RESERVED] = np.arange(NUM_RESERVED, dtype=np.int64)
        _, tgt, mask = mlm_collate(seqs, vocab_size=20, mask_prob=1.0)
        # PAD/BOS/EOS/UNK positions must never be selected for loss
        assert not mask[:, :NUM_RESERVED].any()
        assert (tgt[:, :NUM_RESERVED] == -100).all()


class TestMaskedTokenModel:
    """Forward + tied-embedding head produce per-position vocab logits."""

    def test_logits_shape(self):
        m = MaskedTokenModel(vocab_size=30, d_model=16, n_layers=1, n_heads=2)
        ids = torch.randint(NUM_RESERVED, 30, (2, 7))
        logits, cls = m(ids)
        assert logits.shape == (2, 7, 30)
        assert cls.shape == (2, 16)


class TestMTMTrainerOverfit:
    """A tiny model + structured data should overfit MLM accuracy well above chance."""

    def test_overfits_structured_data(self):
        """Each sequence is a simple arithmetic progression: ``seq[i] = base + i``.

        The MLM objective is then "given some positions, predict the
        missing ones" — perfectly learnable from sequence position.
        If even this collapses, something is wrong with the encoder /
        masking / loss plumbing.
        """
        rng = np.random.default_rng(0)
        vocab_size = 32
        seq_len = 10
        n_seqs = 64
        # seq[i] = (base + i) mod (vocab_size - NUM_RESERVED) + NUM_RESERVED
        # — each sequence is a structured progression; position predicts value.
        bases = rng.integers(
            NUM_RESERVED, vocab_size, size=(n_seqs, 1), dtype=np.int64
        )
        offsets = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        seqs = (bases + offsets - NUM_RESERVED) % (vocab_size - NUM_RESERVED) + NUM_RESERVED

        model = MaskedTokenModel(
            vocab_size=vocab_size,
            d_model=24,
            n_layers=2,
            n_heads=2,
            max_seq_len=seq_len,
            dropout=0.0,  # tiny test data — turn off dropout regularisation
        )
        cfg = MTMTrainConfig(
            n_epochs=80, batch_size=16, lr=3e-3, warmup_steps=10, log_every=10
        )
        trainer = MTMTrainer(model, vocab_size=vocab_size, cfg=cfg)
        history = trainer.fit(seqs)
        assert history
        final_acc = history[-1]["mlm_accuracy"]
        # Chance accuracy on (vocab_size - NUM_RESERVED) = 28 classes is ~3.6%.
        # The structured data is exactly learnable so a small encoder
        # should comfortably clear 50% after 80 epochs.
        assert final_acc > 0.5, (
            f"MLM accuracy stuck at {final_acc:.3f} — model couldn't "
            f"learn the position-determined progression"
        )


class TestMTMTrainerPersistence:
    """``save`` + ``load_encoder`` round-trip preserves the encoder weights."""

    def test_save_load_round_trip(self, tmp_path):
        torch.manual_seed(0)
        model = MaskedTokenModel(
            vocab_size=20,
            d_model=16,
            n_layers=1,
            n_heads=2,
            max_seq_len=8,
            dropout=0.0,
        )
        # Pin both ends to CPU so the comparison is on identical devices
        # regardless of the host machine's accelerator.
        trainer = MTMTrainer(
            model, vocab_size=20, cfg=MTMTrainConfig(n_epochs=0, device="cpu")
        )
        out = tmp_path / "mtm.pt"
        trainer.save(out, extras={"label": "test"})
        encoder = MTMTrainer.load_encoder(out, device="cpu")
        trainer.model.eval()
        encoder.eval()

        ids = torch.randint(NUM_RESERVED, 20, (2, 8))
        cls_orig, _ = trainer.model.encoder.encode(ids)
        cls_loaded, _ = encoder.encode(ids)
        assert torch.allclose(cls_orig, cls_loaded, atol=1e-6)


class TestTokenObsBuffer:
    def test_round_trip(self):
        buf = TokenObsBuffer(capacity=3, seq_length=4)
        buf.add(np.array([1, 2, 3, 4]))
        buf.add(np.array([5, 6, 7, 8]))
        out = buf.as_array()
        assert out.shape == (2, 4)
        assert (out[0] == [1, 2, 3, 4]).all()
        assert (out[1] == [5, 6, 7, 8]).all()

    def test_ring_overflow(self):
        buf = TokenObsBuffer(capacity=2, seq_length=3)
        for v in range(5):
            buf.add(np.array([v, v, v]))
        out = buf.as_array()
        assert out.shape == (2, 3)
        # Last two writes were [3,3,3] and [4,4,4] — both should be present
        assert sorted(out[:, 0].tolist()) == [3, 4]

    def test_rejects_wrong_shape(self):
        buf = TokenObsBuffer(capacity=2, seq_length=4)
        with pytest.raises(ValueError, match="does not match"):
            buf.add(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Hybrid-mode MLM (mlm_collate + MaskedTokenModel + MTMTrainer.fit)
# ---------------------------------------------------------------------------


class TestMLMCollateHybrid:
    """Mask only categorical positions when ``cont_kinds`` is provided."""

    def test_continuous_positions_are_never_masked(self):
        rng = np.random.default_rng(0)
        # Mix of categorical (kind=0) and continuous (kind=3) positions.
        # All token ids are above NUM_RESERVED so the special-token
        # filter would otherwise leave them eligible.
        seqs = rng.integers(NUM_RESERVED, 30, size=(8, 16), dtype=np.int64)
        kinds = np.zeros((8, 16), dtype=np.int8)
        kinds[:, ::2] = 3  # every other position is Time2Vec-routed
        _, _, mask = mlm_collate(
            seqs, vocab_size=30, mask_prob=1.0,
            rng=rng, cont_kinds=kinds,
        )
        # No continuous position was selected.
        assert not mask[kinds != 0].any()
        # Categorical positions WERE all selected (mask_prob=1.0).
        assert mask[kinds == 0].all()


class TestMTMTrainerHybrid:
    """End-to-end fit on a hybrid (dict-of-arrays) input."""

    def _build(self):
        from agents.networks.hybrid_encoder import HybridTokenEncoder

        m = MaskedTokenModel(
            vocab_size=24, d_model=24, n_heads=2, n_layers=1,
            max_seq_len=8, dropout=0.0, hybrid=True, sim_time_scale=100.0,
        )
        assert isinstance(m.encoder, HybridTokenEncoder)
        return m

    def test_fit_accepts_hybrid_dict_and_overfits(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        vocab = 24
        # Structured progression in the categorical channel so the
        # MLM head has actual signal to fit.
        n, S = 64, 8
        bases = rng.integers(NUM_RESERVED, vocab, size=(n, 1), dtype=np.int64)
        offsets = np.arange(S, dtype=np.int64).reshape(1, S)
        tokens = (bases + offsets - NUM_RESERVED) % (vocab - NUM_RESERVED) + NUM_RESERVED
        # All-categorical so the masking is unconstrained.
        seqs = {
            "token_ids": tokens,
            "cont_values": np.zeros((n, S), dtype=np.float32),
            "cont_kinds": np.zeros((n, S), dtype=np.int8),
        }
        model = self._build()
        cfg = MTMTrainConfig(n_epochs=80, batch_size=16, lr=3e-3, warmup_steps=10,
                              log_every=10, device="cpu")
        trainer = MTMTrainer(model, vocab_size=vocab, cfg=cfg)
        history = trainer.fit(seqs)
        assert history
        final = history[-1]["mlm_accuracy"]
        assert final > 0.5, f"hybrid MLM stuck at {final:.3f}"

    def test_fit_rejects_dict_with_non_hybrid_model(self):
        torch.manual_seed(0)
        model = MaskedTokenModel(vocab_size=24, d_model=16, n_heads=2,
                                 n_layers=1, max_seq_len=8, dropout=0.0)
        trainer = MTMTrainer(model, vocab_size=24,
                             cfg=MTMTrainConfig(n_epochs=1, device="cpu"))
        with pytest.raises(ValueError, match="hybrid=True"):
            trainer.fit(
                {
                    "token_ids": np.zeros((2, 8), dtype=np.int64),
                    "cont_values": np.zeros((2, 8), dtype=np.float32),
                    "cont_kinds": np.zeros((2, 8), dtype=np.int8),
                }
            )
