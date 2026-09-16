"""Frozen-rate memory (rnn_type="ema").

The memory must (1) decay by exactly exp(-dt / tau) in simulated time,
(2) be silent at initialisation, so a memoryless checkpoint initialises it
without changing the policy, (3) carry gradient into its read and write
projections, and (4) survive the agent's rollout storage, update and
checkpoint round trip.  The env must expose sim_time only when asked.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.networks.set_transformer import (
    EMAMemory,
    SetTransformerActorCritic,
    SetTransformerEncoder,
)
from agents.ppo.ppo_set_transformer import SetTransformerAgent
from kata.EntityFactories import RandomScenarioSampler
from kata.core.config import KATAConfig, get_config
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder

B, S, M, L, LE = 2, 12, 20, 16, 8


def make_obs(seed: int = 0, sim_time: float = 0.0):
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
        "tech_mask": torch.tensor([[i < 7 for i in range(S)]] * B),
        "machine_mask": torch.tensor([[i < 9 for i in range(M)]] * B),
        "sim_time": torch.full((B, 1), float(sim_time)),
    }


def build_net(rnn_type: str):
    torch.manual_seed(0)
    enc = SetTransformerEncoder(
        vocab_size=64, d_model=32, n_heads=4, n_layers=2, dropout=0.0,
        max_techs=S, max_machines=M, env_length=LE, tech_slot_length=L,
        slot_role_binding=True, use_feature_context=True,
        cross_slot="attention", set_positional=False,
    )
    net = SetTransformerActorCritic(enc, rnn_type=rnn_type)
    with torch.no_grad():  # make logits large enough for comparisons to bite
        net.policy_head.q_proj.weight.mul_(100.0)
        net.policy_head.k_proj.weight.mul_(100.0)
    net.eval()
    return net


def identity_memory(half_lives):
    mem = EMAMemory(d_ctx=4, half_lives=half_lives, d_mem=4)
    with torch.no_grad():
        mem.write.weight.copy_(torch.eye(4))
        mem.write.bias.zero_()
    return mem


def test_half_life_is_exact():
    mem = identity_memory([10.0, 1000.0])
    state = mem.initial_state(1, torch.device("cpu"))
    _, state = mem(torch.ones(1, 4), torch.tensor([0.0]), state)
    stored = state[0].reshape(1, 2, 4)
    assert torch.allclose(stored, torch.ones(1, 2, 4)), "first write fills every channel"
    _, state = mem(torch.zeros(1, 4), torch.tensor([10.0]), state)
    stored = state[0].reshape(1, 2, 4)
    assert torch.allclose(stored[0, 0], torch.full((4,), 0.5), atol=1e-6)
    expected_slow = math.exp(-10.0 * math.log(2.0) / 1000.0)
    assert torch.allclose(stored[0, 1], torch.full((4,), expected_slow), atol=1e-6)
    assert state[1].item() == 10.0


def test_time_running_backwards_does_not_amplify():
    mem = identity_memory([10.0])
    state = mem.initial_state(1, torch.device("cpu"))
    _, state = mem(torch.ones(1, 4), torch.tensor([100.0]), state)
    _, state = mem(torch.zeros(1, 4), torch.tensor([50.0]), state)
    assert torch.allclose(state[0], torch.ones(1, 1, 4)), "negative dt must clamp to zero"


def test_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        EMAMemory(d_ctx=4, half_lives=[60.0, 0.0])


def test_silent_at_init_matches_memoryless_policy():
    plain = build_net("none")
    memory = build_net("ema")
    missing = memory.load_state_dict(plain.state_dict(), strict=False)
    assert all(k.startswith("memory.") for k in missing.missing_keys)
    assert not missing.unexpected_keys
    hidden = None
    for step, t in enumerate([0.0, 30.0, 400.0]):
        obs = make_obs(seed=step, sim_time=t)
        lp, vp, _ = plain(obs)
        lm, vm, hidden = memory(obs, hidden)
        finite = torch.isfinite(lp)
        assert torch.allclose(lp[finite], lm[finite], atol=1e-5)
        assert torch.allclose(vp, vm, atol=1e-5)


def test_gradient_reaches_read_and_write():
    net = build_net("ema")
    net.train()
    _, _, hidden = net(make_obs(seed=0, sim_time=0.0))
    with torch.no_grad():
        net.memory.read.weight.normal_(0, 0.1)
    _, value, _ = net(make_obs(seed=1, sim_time=60.0), hidden)
    value.sum().backward()
    assert net.memory.read.weight.grad.abs().sum() > 0
    assert net.memory.write.weight.grad.abs().sum() > 0
    assert net.memory.tau.requires_grad is False


def test_missing_sim_time_is_an_error():
    net = build_net("ema")
    obs = make_obs()
    obs.pop("sim_time")
    with pytest.raises(ValueError, match="expose_sim_time"):
        net(obs)


def _build_set_env(expose_sim_time: bool):
    cfg_dict = json.loads(Path("run_configs/benchmark_suite/baseline.json").read_text())
    gym = cfg_dict["gym"]
    gym.update({
        "observation_representation": "set", "max_techs": 30, "max_machines": 100,
        "set_tech_slot_length": 16, "set_machine_slot_length": 12, "set_env_length": 16,
        "max_sim_time": 1500.0, "max_episode_steps": 40,
        "expose_sim_time": expose_sim_time,
    })
    cfg = KATAConfig(**cfg_dict)
    cached = get_config()
    cached.sim = cfg.sim
    cached.gym = cfg.gym
    sampler = RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=0)
    return KataEnv(
        scenario_factory=lambda: ScenarioBuilder(sampler.sample_config()).build(),
        config=cfg.gym,
    )


def test_env_exposes_sim_time_only_when_asked():
    env = _build_set_env(expose_sim_time=False)
    obs, _ = env.reset(seed=3)
    assert "sim_time" not in obs and "sim_time" not in env.observation_space.spaces

    env = _build_set_env(expose_sim_time=True)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.spaces["sim_time"].shape == (1,)
    t0 = float(obs["sim_time"][0])
    assert t0 == pytest.approx(env._sim_time())
    a = int(np.where(obs["action_mask"])[0][0]) if obs["action_mask"].any() else 0
    obs, *_ = env.step(a)
    assert float(obs["sim_time"][0]) >= t0


@pytest.mark.slow
def test_agent_rollout_update_and_checkpoint_roundtrip(tmp_path):
    env = _build_set_env(expose_sim_time=True)
    obs, _ = env.reset(seed=11)
    for s in range(30):  # populate the vocabulary
        a = int(np.where(obs["action_mask"])[0][0]) if obs["action_mask"].any() else 0
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset(seed=12 + s)
    env._tokenizer.freeze()
    kwargs = dict(
        n_actions=30, vocab_size=env._tokenizer.vocab_size,
        d_model=32, n_heads=4, n_layers=2,
        max_techs=30, max_machines=100, env_length=16,
        rollout_steps=16, minibatch_size=8, n_epochs=2,
        rnn_type="ema", memory_half_lives=[60.0, 1440.0], memory_dim=8,
        device="cpu", seed=0,
    )
    agent = SetTransformerAgent(**kwargs)
    obs, info = env.reset(seed=99)
    for _ in range(20):
        a = agent.select_action(obs)
        nxt, r, term, trunc, info = env.step(a)
        agent.observe_transition(obs, a, r, nxt, term, trunc, info)
        obs = nxt
        if term or trunc:
            break
    stats = agent.update()
    assert np.isfinite(stats["loss"])

    path = tmp_path / "ema.pt"
    agent.save(path)
    imp = torch.load(path, map_location="cpu", weights_only=False)["improvements"]
    assert imp["rnn_type"] == "ema"
    assert imp["memory_half_lives"] == [60.0, 1440.0] and imp["memory_dim"] == 8
    reloaded = SetTransformerAgent(**kwargs)
    reloaded.load(path)
    for k, v in agent.net.state_dict().items():
        assert torch.equal(v, reloaded.net.state_dict()[k]), k
