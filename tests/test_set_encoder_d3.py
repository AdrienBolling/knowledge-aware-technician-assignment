"""D3 regression tests — role-bound slot fusion + two-view pooling.

Defect D3: the historical ``_SlotFuser`` zeroed the categorical (role)
embedding at continuous positions and mean-pooled within each slot, so
same-kind continuous features (e.g. FATIGUE vs MATCH, both ratio-PLE)
were permutation-indistinguishable: the encoder provably could not tell
a technician with MATCH=0.9/FATIGUE=0.1 from MATCH=0.1/FATIGUE=0.9.

``slot_role_binding=True`` keeps the role embedding at every position
and applies a nonlinear per-position binder before pooling (additive
role info alone cancels in the mean).  ``use_feature_context=True`` adds
the per-feature-across-technicians context view.
"""

from __future__ import annotations

import numpy as np
import torch

from agents.networks.continuous_features import ContKind
from agents.networks.set_transformer import SetTransformerEncoder

VOCAB = 16
PAD = 0
D = 32


def _encoder(**kw) -> SetTransformerEncoder:
    torch.manual_seed(0)
    return SetTransformerEncoder(
        vocab_size=VOCAB,
        d_model=D,
        n_heads=4,
        n_layers=1,
        max_techs=2,
        max_machines=2,
        env_length=4,
        tech_slot_length=4,
        **kw,
    ).eval()


def _obs(swap: bool) -> dict[str, torch.Tensor]:
    """Two techs, each with one categorical + two ratio features.

    ``swap=True`` exchanges the two ratio VALUES within each slot while
    keeping the role tokens in place — the exact permutation the
    historical encoder could not distinguish.
    """
    # positions: [TEMPLATE=x (cat), <RATIO:A>, <RATIO:B>, pad]
    ids = torch.tensor([[[1, 2, 3, PAD]] * 2], dtype=torch.long)
    kinds = torch.tensor(
        [[[ContKind.CATEGORICAL, ContKind.RATIO_PLE, ContKind.RATIO_PLE, 0]] * 2],
        dtype=torch.long,
    )
    a, b = (0.9, 0.1) if not swap else (0.1, 0.9)
    vals = torch.tensor([[[0.0, a, b, 0.0]] * 2], dtype=torch.float32)
    mach_ids = torch.tensor([[[4, PAD], [5, PAD]]], dtype=torch.long)
    zeros2 = torch.zeros(1, 2, 2)
    env_ids = torch.tensor([[6, 7, PAD, PAD]], dtype=torch.long)
    return {
        "tech_token_ids": ids,
        "tech_cont_values": vals,
        "tech_cont_kinds": kinds,
        "tech_mask": torch.ones(1, 2, dtype=torch.bool),
        "machine_token_ids": mach_ids,
        "machine_cont_values": zeros2,
        "machine_cont_kinds": torch.zeros(1, 2, 2, dtype=torch.long),
        "machine_mask": torch.ones(1, 2, dtype=torch.bool),
        "env_token_ids": env_ids,
        "env_cont_values": torch.zeros(1, 4),
        "env_cont_kinds": torch.zeros(1, 4, dtype=torch.long),
    }


def test_legacy_fuser_aliases_same_kind_features():
    """Pin the defect: without role binding, swapping same-kind values
    within a slot leaves the encoder output EXACTLY unchanged."""
    enc = _encoder(slot_role_binding=False)
    with torch.no_grad():
        ctx_a, slots_a, _ = enc(_obs(swap=False))
        ctx_b, slots_b, _ = enc(_obs(swap=True))
    np.testing.assert_allclose(ctx_a.numpy(), ctx_b.numpy(), atol=1e-6)
    np.testing.assert_allclose(slots_a.numpy(), slots_b.numpy(), atol=1e-6)


def test_role_binding_breaks_the_aliasing():
    enc = _encoder(slot_role_binding=True)
    with torch.no_grad():
        ctx_a, slots_a, _ = enc(_obs(swap=False))
        ctx_b, slots_b, _ = enc(_obs(swap=True))
    assert not np.allclose(ctx_a.numpy(), ctx_b.numpy(), atol=1e-6)
    assert not np.allclose(slots_a.numpy(), slots_b.numpy(), atol=1e-6)


def test_feature_context_plumbing():
    enc = _encoder(slot_role_binding=True, use_feature_context=True)
    assert enc.feature_context_proj is not None
    assert enc.context_proj.fc1.in_features == 4 * D
    with torch.no_grad():
        ctx, slots, mask = enc(_obs(swap=False))
    assert ctx.shape == (1, D)
    assert slots.shape == (1, 2, D)
    # Flag off: no module, 3-way concat, identical call signature.
    enc_off = _encoder()
    assert enc_off.feature_context_proj is None
    assert enc_off.context_proj.fc1.in_features == 3 * D
    with torch.no_grad():
        enc_off(_obs(swap=False))


def test_agent_checkpoint_carries_d3_flags(tmp_path):
    from agents.ppo.ppo_set_transformer import SetTransformerAgent

    agent = SetTransformerAgent(
        n_actions=2,
        vocab_size=VOCAB,
        max_techs=2,
        max_machines=2,
        env_length=4,
        d_model=D,
        n_heads=4,
        n_layers=1,
        slot_role_binding=True,
        use_feature_context=True,
        tech_slot_length=4,
        device="cpu",
    )
    path = tmp_path / "ckpt.pt"
    agent.save(path)
    imp = torch.load(path, map_location="cpu", weights_only=False)[
        "improvements"
    ]
    assert imp["slot_role_binding"] is True
    assert imp["use_feature_context"] is True
    assert imp["tech_slot_length"] == 4

    # An agent rebuilt with the flags loads the checkpoint strictly.
    clone = SetTransformerAgent(
        n_actions=2,
        vocab_size=VOCAB,
        max_techs=2,
        max_machines=2,
        env_length=4,
        d_model=D,
        n_heads=4,
        n_layers=1,
        slot_role_binding=True,
        use_feature_context=True,
        tech_slot_length=4,
        device="cpu",
    )
    clone.load(path)
