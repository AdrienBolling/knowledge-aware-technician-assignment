"""The architecture ablation ladder: flat -> shared+pool -> attention -> hybrid.

Each rung must differ from the next in exactly the property it is meant to
isolate, and every rung must keep the parts that are supposed to be held
constant (pointer head, value head, masking).
"""
import pytest
import torch

from agents.networks.set_transformer import (
    SetTransformerActorCritic,
    SetTransformerEncoder,
)

B, S, M, L, LE = 2, 12, 20, 16, 8
N_VALID_T, N_VALID_M = 7, 9

RUNGS = {
    "flat": dict(cross_slot="flat", numeric_encoding="plain", use_cross_attention=False),
    "pool": dict(cross_slot="pool", numeric_encoding="plain", use_cross_attention=False),
    "set_attn": dict(cross_slot="attention", numeric_encoding="plain",
                     set_positional=False),
    "attn_rope": dict(cross_slot="attention", numeric_encoding="plain"),
    "htt": dict(cross_slot="attention", numeric_encoding="hybrid"),
}


def make_obs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "tech_token_ids": torch.randint(1, 50, (B, S, L), generator=g),
        "tech_cont_values": torch.rand(B, S, L, generator=g) * 100.0,
        "tech_cont_kinds": torch.randint(0, 5, (B, S, L), generator=g),
        "machine_token_ids": torch.randint(1, 50, (B, M, L), generator=g),
        "machine_cont_values": torch.rand(B, M, L, generator=g) * 100.0,
        "machine_cont_kinds": torch.randint(0, 5, (B, M, L), generator=g),
        "env_token_ids": torch.randint(1, 50, (B, LE), generator=g),
        "env_cont_values": torch.rand(B, LE, generator=g) * 100.0,
        "env_cont_kinds": torch.randint(0, 5, (B, LE), generator=g),
        "tech_mask": torch.tensor([[i < N_VALID_T for i in range(S)]] * B),
        "machine_mask": torch.tensor([[i < N_VALID_M for i in range(M)]] * B),
    }


def build(rung: str):
    torch.manual_seed(0)
    enc = SetTransformerEncoder(
        vocab_size=64, d_model=32, n_heads=4, n_layers=2, dropout=0.0,
        max_techs=S, max_machines=M, env_length=LE, tech_slot_length=L,
        slot_role_binding=True, use_feature_context=True, **RUNGS[rung]
    )
    net = SetTransformerActorCritic(enc)
    # PointerActionHead initialises its projections with gain 0.01, so at
    # init every rung emits ~1e-6 logits and any allclose comparison passes
    # trivially.  Scale the head so the symmetry tests below actually bite;
    # scaling is linear and symmetric, so it cannot create equivariance.
    with torch.no_grad():
        net.policy_head.q_proj.weight.mul_(100.0)
        net.policy_head.k_proj.weight.mul_(100.0)
    net.eval()
    return net


def permute_techs(obs, perm):
    out = dict(obs)
    for k in ("tech_token_ids", "tech_cont_values", "tech_cont_kinds"):
        out[k] = obs[k][:, perm]
    return out


@pytest.mark.parametrize("rung", list(RUNGS))
def test_shapes_and_masking(rung):
    net = build(rung)
    logits, value, _ = net(make_obs())
    assert logits.shape == (B, S) and value.shape == (B,)
    assert torch.isinf(logits[:, N_VALID_T:]).all(), "padded slots must be -inf"
    assert torch.isfinite(logits[:, :N_VALID_T]).all()
    assert torch.isfinite(value).all()


@pytest.mark.parametrize("rung", ["pool", "set_attn"])
def test_permutation_equivariant_rungs(rung):
    """Rungs without positional encoding must be permutation-equivariant."""
    net = build(rung)
    obs = make_obs()
    perm = torch.tensor([3, 0, 5, 1, 6, 2, 4] + list(range(N_VALID_T, S)))
    base, _, _ = net(obs)
    perm_logits, _, _ = net(permute_techs(obs, perm))
    torch.testing.assert_close(
        perm_logits[:, :N_VALID_T], base[:, perm[:N_VALID_T]], atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("rung", ["flat", "attn_rope", "htt"])
def test_order_aware_rungs_are_not_permutation_equivariant(rung):
    """The flat rung ties meaning to slot index by flattening; the RoPE rungs
    do so through rotary embeddings over the slot axis.  Both are order-aware
    -- which is why the ladder needs a no-positional attention rung to isolate
    "attention" from "positional encoding"."""
    net = build(rung)
    obs = make_obs()
    perm = torch.tensor([3, 0, 5, 1, 6, 2, 4] + list(range(N_VALID_T, S)))
    base, _, _ = net(obs)
    perm_logits, _, _ = net(permute_techs(obs, perm))
    assert not torch.allclose(
        perm_logits[:, :N_VALID_T], base[:, perm[:N_VALID_T]], atol=1e-4
    ), f"{rung} unexpectedly permutation-equivariant"


def test_pool_rung_has_no_attention():
    """Rung 2 must contain no attention module anywhere."""
    net = build("pool")
    names = [type(m).__name__ for m in net.modules()]
    assert not any("Attention" in n or "MultiheadAttention" in n for n in names), names
    assert net.encoder.cross_attn is None


def test_plain_numeric_replaces_the_hybrid_encoders():
    net = build("attn_rope")
    assert net.encoder.plain_numeric is not None
    # the hybrid modules still exist as attributes but must be unused
    assert net.encoder._fuser.plain_numeric is not None
    htt = build("htt")
    assert htt.encoder.plain_numeric is None and htt.encoder._fuser.plain_numeric is None


def test_plain_numeric_is_sensitive_to_unbounded_values():
    """symlog keeps large counts/times from saturating the shared projection."""
    net = build("attn_rope")
    obs = make_obs()
    big = dict(obs)
    big["tech_cont_values"] = obs["tech_cont_values"] * 1e4
    a, _, _ = net(obs)
    b, _, _ = net(big)
    assert torch.isfinite(b[:, :N_VALID_T]).all()
    assert not torch.allclose(a[:, :N_VALID_T], b[:, :N_VALID_T])


def test_attention_rung_rejects_inconsistent_flags():
    with pytest.raises(ValueError, match="no attention at all"):
        SetTransformerEncoder(vocab_size=16, d_model=32, n_heads=4, n_layers=1,
                              cross_slot="pool", use_cross_attention=True)
    with pytest.raises(ValueError, match="numeric_encoding"):
        SetTransformerEncoder(vocab_size=16, d_model=32, n_heads=4, n_layers=1,
                              numeric_encoding="bogus")


@pytest.mark.parametrize("rung", list(RUNGS))
def test_head_and_value_are_shared_across_rungs(rung):
    """Everything after the encoder must be the same code on every rung."""
    net = build(rung)
    assert type(net.policy_head).__name__ == "PointerActionHead"
    assert net.policy_head.q_proj.out_features == 64
    assert isinstance(net.value_head, torch.nn.Sequential)


def test_permissive_action_mask_keeps_busy_technicians_selectable():
    """The reviewer's point: under the default mask, waiting for a busy
    expert is not an available action at all."""
    import numpy as np
    from kata.env import KataEnv

    class _T:
        def __init__(self, busy=False, disr=False, retired=False):
            self.busy, self._in_disruption, self.retired = busy, disr, retired

    class _D:
        def __init__(self, techs): self.techs = techs

    env = KataEnv.__new__(KataEnv)
    env.dispatcher = _D([_T(), _T(busy=True), _T(disr=True), _T(retired=True)])

    class _C:
        mask_unavailable_technicians = True
    env.config = _C()
    strict = env._action_mask()
    assert list(strict) == [1, 0, 0, 0], strict          # only the free novice

    class _P:
        mask_unavailable_technicians = False
    env.config = _P()
    permissive = env._action_mask()
    assert list(permissive) == [1, 1, 1, 0], permissive  # busy/absent selectable
    assert permissive[3] == 0, "retired slots stay impossible"
