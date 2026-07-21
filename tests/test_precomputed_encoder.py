"""Tests for the canonical precomputed ticket->grid encoder."""

from __future__ import annotations

import numpy as np
from conftest import FakeDispatcher, FakeRequest, FakeSimEnv

from kata.core.config import GymEnvConfig
from kata.entities.encoder import base as encoder_base
from kata.entities.encoder.precomputed import PrecomputedEncoder
from kata.env import KataEnv

ARTIFACT = "run_configs/embeddings/ticket_grid_som.json"


class _Req:
    def __init__(self, mtype, ctype):
        self.machine = type("M", (), {"mtype": mtype})()
        self._ct = ctype

    def get_failed_component_info(self):
        return {"component_type": self._ct, "component_id": 0, "repair_time": 1}


def test_known_key_maps_to_interior_cell_center():
    enc = PrecomputedEncoder(ARTIFACT)
    key = next(iter(enc._table))
    mt, ct = key.split(":")
    emb = enc.encode(_Req(mt, ct))
    assert emb.shape == (2,)
    cell = tuple(int(v // 10) for v in emb)
    assert all(1 <= c <= 8 for c in cell)  # curated keys live in the interior
    # deterministic
    assert np.allclose(emb, enc.encode(_Req(mt, ct)))


def test_unknown_key_falls_back_to_border_ring():
    enc = PrecomputedEncoder(ARTIFACT)
    for mt, ct in (("Mystery", "gizmo"), ("Alien", "flux"), ("New", "widget")):
        emb = enc.encode(_Req(mt, ct))
        cell = tuple(int(v // 10) for v in emb)
        assert min(cell) == 0 or max(cell) == 9, (mt, ct, cell)
    # deterministic fallback too
    assert np.allclose(enc.encode(_Req("Mystery", "gizmo")),
                       enc.encode(_Req("Mystery", "gizmo")))


def test_env_installs_precomputed_encoder():
    prev = encoder_base.ENCODER
    try:
        d = FakeDispatcher(tech_count=2)
        d.repair_queue.items.append(FakeRequest(machine_id=1))
        env = KataEnv(
            sim_env=FakeSimEnv(), dispatcher=d,
            config=GymEnvConfig(
                max_episode_steps=5, max_sim_time=100.0,
                ticket_embedding_path=ARTIFACT,
            ),
        )
        env.reset()
        assert isinstance(encoder_base.ENCODER, PrecomputedEncoder)
        assert encoder_base.ENCODER.path == ARTIFACT
    finally:
        encoder_base.ENCODER = prev
