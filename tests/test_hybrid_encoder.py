"""Tests for HybridTokenEncoder and its integration with the obs pipeline."""

from __future__ import annotations

import numpy as np
import torch

from agents.networks.continuous_features import ContKind
from agents.networks.hybrid_encoder import HybridTokenEncoder


class TestHybridTokenEncoder:
    def _build(self, vocab_size: int = 32, d_model: int = 16, seq: int = 8):
        return HybridTokenEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=2,
            n_layers=1,
            max_seq_len=seq,
            dropout=0.0,
        )

    def test_encode_shapes(self):
        enc = self._build(vocab_size=32, d_model=16, seq=8)
        tok = torch.randint(5, 32, (2, 8))
        cv = torch.zeros((2, 8))
        ck = torch.zeros((2, 8), dtype=torch.int8)
        cls, per_tok = enc.encode(tok, cv, ck)
        assert cls.shape == (2, 16)
        assert per_tok.shape == (2, 8, 16)

    def test_continuous_kind_changes_output(self):
        """Changing ``cont_kinds`` at one position should produce different CLS embeddings."""
        torch.manual_seed(0)
        enc = self._build(vocab_size=32, d_model=16, seq=4)
        enc.eval()
        tok = torch.tensor([[10, 4, 11, 4]])  # 4 is the <NUM> placeholder
        cv = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
        # Variant A: numerical position 1 routed to PLE-ratio
        ck_a = torch.tensor(
            [[ContKind.CATEGORICAL, ContKind.RATIO_PLE, ContKind.CATEGORICAL, ContKind.RATIO_PLE]],
            dtype=torch.int8,
        )
        # Variant B: same position but routed to Time2Vec
        ck_b = torch.tensor(
            [[ContKind.CATEGORICAL, ContKind.TIME2VEC, ContKind.CATEGORICAL, ContKind.TIME2VEC]],
            dtype=torch.int8,
        )
        cls_a, _ = enc.encode(tok, cv, ck_a)
        cls_b, _ = enc.encode(tok, cv, ck_b)
        assert not torch.allclose(cls_a, cls_b, atol=1e-4), (
            "PLE-routed and Time2Vec-routed embeddings should diverge"
        )

    def test_purely_categorical_matches_modern_encoder(self):
        """When every position is categorical the hybrid encoder agrees with the backbone."""
        torch.manual_seed(0)
        enc = self._build(vocab_size=32, d_model=16, seq=6)
        enc.eval()
        tok = torch.randint(5, 32, (2, 6))
        cv = torch.zeros((2, 6))
        ck = torch.zeros((2, 6), dtype=torch.int8)  # all categorical
        cls_hybrid, _ = enc.encode(tok, cv, ck)
        cls_backbone, _ = enc.backbone.encode(tok)
        assert torch.allclose(cls_hybrid, cls_backbone, atol=1e-5)


class TestHybridEnvObs:
    """The env's ``hybrid`` obs mode emits aligned three-channel observations."""

    def _build_env(self):
        from kata.env import KataEnv
        from kata.core.config import GymEnvConfig
        from conftest import FakeDispatcher, FakeMachine, FakeRequest, FakeSimEnv

        sim_env = FakeSimEnv()
        dispatcher = FakeDispatcher(tech_count=2)
        machine = FakeMachine(machine_id=1)
        req = FakeRequest(machine_id=1, created_at=0.0)
        req.machine = machine
        dispatcher.repair_queue.items.append(req)
        dispatcher.machines = [machine]
        env = KataEnv(
            sim_env=sim_env,
            dispatcher=dispatcher,
            config=GymEnvConfig(
                observation_representation="hybrid",
                observation_mode="tech_aware",
                include_technician_fatigue_tokens=True,
                include_technician_knowledge_tokens=True,
                next_ticket_lookahead=2,
                tokenizer_seq_length=64,
                token_observation_length=64,
            ),
        )
        return env

    def test_hybrid_obs_contract(self):
        env = self._build_env()
        obs, _ = env.reset()
        assert "token_ids" in obs and "cont_values" in obs and "cont_kinds" in obs
        tid, cv, ck = obs["token_ids"], obs["cont_values"], obs["cont_kinds"]
        assert tid.shape == cv.shape == ck.shape == (64,)
        # cont_kinds must be valid codes
        assert set(np.unique(ck).tolist()).issubset({0, 1, 2, 3, 4})
        # Categorical positions carry no continuous payload
        assert (cv[ck == 0] == 0).all()

    def test_num_placeholder_present_only_at_numerical_positions(self):
        from kata.core.tokenizer import NUM_ID

        env = self._build_env()
        obs, _ = env.reset()
        tid, ck = obs["token_ids"], obs["cont_kinds"]
        # Wherever cont_kinds != CATEGORICAL, the token id is <NUM>.
        non_cat = ck != 0
        assert (tid[non_cat] == NUM_ID).all()
        # And nowhere else.
        cat = ck == 0
        assert (tid[cat] != NUM_ID).all() or not non_cat.any()
