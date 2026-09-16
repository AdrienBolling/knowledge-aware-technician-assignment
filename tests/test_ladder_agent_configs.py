"""The architecture-ladder agent configs build the intended encoders.

``run_configs/agents/set_transformer_v6_ladder_{flat,pool,plainset}.json``
are the permutation-fixed v6 config with only the encoder flags changed.
These tests pin three things:

* each config differs from ``set_transformer_v6_permfix.json`` in the
  encoder flags only (every PPO hyperparameter is the same);
* ``SetTransformerAgent`` built from each config has the rung's encoder
  (the flags in ``tests/test_architecture_ladder.py``);
* a saved checkpoint records the flags, and the benchmark harness
  rebuilds the same architecture from them and restores every tensor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from agents import SetTransformerAgent
from agents.networks.modern_transformer import RoPESelfAttention

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "run_configs" / "agents"

ENCODER_KEYS = {"cross_slot", "numeric_encoding", "use_cross_attention", "flat_hidden"}

RUNGS = {
    "flat": dict(cross_slot="flat", numeric_encoding="plain",
                 use_cross_attention=False, set_positional=False),
    "pool": dict(cross_slot="pool", numeric_encoding="plain",
                 use_cross_attention=False, set_positional=False),
    "plainset": dict(cross_slot="attention", numeric_encoding="plain",
                     use_cross_attention=True, set_positional=False),
}
ENCODER_CLASS = {"flat": "_FlatEncoder", "pool": "_PoolEncoder", "plainset": "_SetEncoder"}

SMALL = dict(n_actions=6, max_techs=6, max_machines=8, env_length=8,
             vocab_size=64, device="cpu")


def _params(rung: str) -> dict:
    data = json.loads((AGENTS / f"set_transformer_v6_ladder_{rung}.json").read_text())
    assert data["agent_type"] == "set_transformer"
    return dict(data["params"])


def _load_harness():
    name = "eval_human_vs_performance"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rung", list(RUNGS))
def test_config_changes_only_the_encoder(rung):
    base = json.loads((AGENTS / "set_transformer_v6_permfix.json").read_text())["params"]
    params = _params(rung)
    changed = {k for k in set(base) | set(params) if base.get(k) != params.get(k)}
    assert changed <= ENCODER_KEYS, changed
    for key, value in RUNGS[rung].items():
        assert params.get(key, True) == value, key


@pytest.mark.parametrize("rung", list(RUNGS))
def test_config_builds_the_rung_encoder(rung):
    agent = SetTransformerAgent(**{**_params(rung), **SMALL})
    enc = agent.net.encoder
    for key, value in RUNGS[rung].items():
        assert getattr(enc, key) == value, key
    assert type(enc.tech_encoder).__name__ == ENCODER_CLASS[rung]
    assert type(enc.machine_encoder).__name__ == ENCODER_CLASS[rung]
    assert enc.plain_numeric is not None and enc._fuser.plain_numeric is not None
    assert (enc.cross_attn is not None) == RUNGS[rung]["use_cross_attention"]
    rope = [m for m in agent.net.modules() if isinstance(m, RoPESelfAttention)]
    if rung == "plainset":
        assert rope and not any(m.use_rope for m in rope)
    else:
        assert not rope, "non-attention rungs must contain no attention"


@pytest.mark.parametrize("rung", list(RUNGS))
def test_checkpoint_round_trip_through_the_harness(rung, tmp_path):
    harness = _load_harness()
    agent = SetTransformerAgent(**{**_params(rung), **SMALL, "use_popart": True,
                                   "normalize_rewards": False})
    ckpt = tmp_path / "set_transformer_best.pt"
    agent.save(ckpt)

    imp = harness.peek_improvements(ckpt)
    for key, value in RUNGS[rung].items():
        assert imp[key] == value, key

    # The harness starts from its generic agent JSON and takes every
    # architecture key from the checkpoint.
    generic = json.loads(harness.AGENT_CONFIG.read_text())["params"]
    params = harness.apply_improvements({**generic, **SMALL}, imp)
    rebuilt = SetTransformerAgent(**params)
    rebuilt.load(ckpt)
    assert harness.check_full_load(rebuilt, ckpt) == []
    for (k, a), (k2, b) in zip(agent.net.state_dict().items(),
                               rebuilt.net.state_dict().items()):
        assert k == k2
        torch.testing.assert_close(a, b)
