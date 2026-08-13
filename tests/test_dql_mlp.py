"""Tests for the DQLMLPAgent (Double DQN over the flattened set obs).

The agent is constructed DIRECTLY with tiny dims on the CPU — no
runner registry, no hydra — against the canonical frozen vocab
``run_configs/vocab/set_vocab.json``.  Fake observations are assembled
position-by-position from the flattener's frozen slot layouts, mirroring
``tests/test_set_flattener.py``.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.dqn.dql_mlp import DQLMLPAgent
from agents.networks.continuous_features import ContKind
from agents.networks.mlp_encoder import (
    ENV_SLOT_LAYOUT,
    MACHINE_SLOT_LAYOUT,
    SetObsFlattener,
    TECH_SLOT_LAYOUT,
)

VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "run_configs" / "vocab" / "set_vocab.json"
)
VOCAB = SetObsFlattener.load_vocab(VOCAB_PATH)
UNK_ID = 1

MAX_T = 4
MAX_M = 3

# Token spelling per continuous kind (kata.env._SetEmitter).
_CONT_FMT = {
    ContKind.RATIO_PLE: "<RATIO:{}>",
    ContKind.COUNT_PLE: "<COUNT:{}>",
    ContKind.TIME2VEC: "<TIME:{}>",
    ContKind.FOURIER: "<FOUR:{}>",
}
_TECH_DEFAULTS = {"TEMPLATE": "default", "BUSY": "F", "DISRUPT": "F"}
_MACH_DEFAULTS = {
    "M_TYPE": "CNC",
    "BROKEN": "F",
    "PROC": "F",
    "IS_CURRENT": "F",
    "CUR_COMP": "NONE",
}
_ENV_DEFAULTS = {
    "HAS_T": "T",
    "T_M_TYPE": "CNC",
    "T_C_TYPE": "motor",
    "N1_M_TYPE": "NONE",
    "N1_C_TYPE": "NONE",
    "N2_M_TYPE": "NONE",
    "N2_C_TYPE": "NONE",
}


def _fill_slot(ids_row, vals_row, kinds_row, layout, defaults, overrides=None):
    """Write one slot's (token id, cont value, kind) triples in layout order.

    Mirrors ``KataEnv._set_obs``: the kinds channel is emitted alongside
    the ids, and the flattener's layout-drift guard checks it.
    """
    values = dict(defaults)
    values.update(overrides or {})
    for pos, sp in enumerate(layout):
        if sp.kind == ContKind.CATEGORICAL:
            ids_row[pos] = VOCAB.get(f"{sp.key}={values.get(sp.key)}", UNK_ID)
        else:
            ids_row[pos] = VOCAB[_CONT_FMT[sp.kind].format(sp.key)]
            vals_row[pos] = float(values.get(sp.key, 0.0))
        kinds_row[pos] = int(sp.kind)


def _make_obs(n_techs=MAX_T, n_machines=2, *, mask=None, fatigue=None, lt=16,
              lm=12, le=16):
    """Fake ``set`` observation with the env's shapes, dtypes and mask.

    ``mask`` overrides the ``action_mask`` field (defaults to "every
    real technician is available"); ``fatigue`` sets per-slot FATIGUE so
    two observations can be made distinguishable.
    """
    obs = {
        "tech_token_ids": np.zeros((MAX_T, lt), dtype=np.int64),
        "tech_cont_values": np.zeros((MAX_T, lt), dtype=np.float32),
        "tech_cont_kinds": np.zeros((MAX_T, lt), dtype=np.int8),
        "tech_mask": np.zeros(MAX_T, dtype=np.int8),
        "machine_token_ids": np.zeros((MAX_M, lm), dtype=np.int64),
        "machine_cont_values": np.zeros((MAX_M, lm), dtype=np.float32),
        "machine_cont_kinds": np.zeros((MAX_M, lm), dtype=np.int8),
        "machine_mask": np.zeros(MAX_M, dtype=np.int8),
        "env_token_ids": np.zeros(le, dtype=np.int64),
        "env_cont_values": np.zeros(le, dtype=np.float32),
        "env_cont_kinds": np.zeros(le, dtype=np.int8),
    }
    for i in range(n_techs):
        over = {} if fatigue is None else {"FATIGUE": float(fatigue[i])}
        _fill_slot(
            obs["tech_token_ids"][i], obs["tech_cont_values"][i],
            obs["tech_cont_kinds"][i],
            TECH_SLOT_LAYOUT, _TECH_DEFAULTS, over,
        )
        obs["tech_mask"][i] = 1
    for i in range(n_machines):
        _fill_slot(
            obs["machine_token_ids"][i], obs["machine_cont_values"][i],
            obs["machine_cont_kinds"][i],
            MACHINE_SLOT_LAYOUT, _MACH_DEFAULTS,
        )
        obs["machine_mask"][i] = 1
    _fill_slot(
        obs["env_token_ids"], obs["env_cont_values"], obs["env_cont_kinds"],
        ENV_SLOT_LAYOUT, _ENV_DEFAULTS,
    )
    if mask is None:
        mask = np.zeros(MAX_T, dtype=np.int8)
        mask[:n_techs] = 1
    obs["action_mask"] = np.asarray(mask, dtype=np.int8)
    return obs


def _make_agent(**overrides):
    defaults = dict(
        n_actions=MAX_T,
        vocab=VOCAB,
        max_techs=MAX_T,
        max_machines=MAX_M,
        hidden_sizes=(8, 8),
        batch_size=4,
        min_replay_size=8,
        train_freq=2,
        target_update_freq=3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=100,
        store_dtype="float32",
        seed=0,
        device="cpu",
    )
    defaults.update(overrides)
    return DQLMLPAgent(**defaults)


def _observe(agent, n, *, done_every=None):
    """Feed ``n`` transitions; every ``done_every``-th one terminates."""
    obs = _make_obs()
    for i in range(n):
        done = bool(done_every and (i + 1) % done_every == 0)
        agent.observe_transition(obs, i % MAX_T, 1.0, obs, done, False, {})


class _StubQ(nn.Module):
    """Q-net returning a fixed table, ignoring its input (target math)."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.as_tensor(table, dtype=torch.float32)

    def forward(self, x):
        return self.table[: x.shape[0]]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_input_width_includes_embedded_mask_bits(self):
        agent = _make_agent()
        flat = SetObsFlattener(VOCAB, max_techs=MAX_T, max_machines=MAX_M)
        assert agent.mask_offset == flat.out_dim
        assert agent.obs_dim == flat.out_dim + MAX_T
        first = agent.online_net.trunk.net[0]
        assert first.in_features == agent.obs_dim
        assert agent.online_net.q_head.out_features == MAX_T

    def test_target_starts_synced_and_frozen(self):
        agent = _make_agent()
        for a, b in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.equal(a, b)
        assert not agent.target_net.training

    def test_net_alias_resolves_for_eval_harness(self):
        """The harness resolves ``net`` first, ``online_net`` second."""
        agent = _make_agent()
        net = getattr(agent, "net", None) or getattr(agent, "online_net", None)
        assert net is agent.online_net

    def test_no_lr_scheduler(self):
        """Constant LR by construction — the defect-D1 class cannot bite."""
        agent = _make_agent()
        assert not hasattr(agent, "lr_scheduler")
        assert agent.optimizer.param_groups[0]["lr"] == agent.lr

    def test_n_actions_must_match_max_techs(self):
        try:
            _make_agent(n_actions=MAX_T + 1)
        except ValueError as exc:
            assert "max_techs" in str(exc)
        else:  # pragma: no cover - the guard is the point
            raise AssertionError("expected a ValueError")


# ---------------------------------------------------------------------------
# Gradient / target cadence
# ---------------------------------------------------------------------------


class TestCadence:
    def test_gradient_steps_follow_env_step_cadence(self):
        """N observe_transition calls -> floor((N - min) / train_freq) steps."""
        agent = _make_agent(min_replay_size=8, train_freq=2, batch_size=4)
        n = 30
        _observe(agent, n)
        assert agent.env_steps == n
        expected = (n - agent.min_replay_size) // agent.train_freq
        assert agent.gradient_steps == expected == 11

    def test_no_learning_before_min_replay(self):
        agent = _make_agent(min_replay_size=8, train_freq=2, batch_size=4)
        _observe(agent, 8)
        assert len(agent.replay) == 8
        assert agent.gradient_steps == 0
        _observe(agent, 2)  # first two warm steps -> one gradient step
        assert agent.gradient_steps == 1

    def test_target_syncs_at_gradient_step_multiples(self):
        agent = _make_agent(
            min_replay_size=8, train_freq=2, batch_size=4, target_update_freq=3
        )
        n = 30
        _observe(agent, n)
        assert agent.gradient_steps == 11
        assert agent.target_syncs == 11 // 3 == 3

    def test_target_tracks_online_only_at_syncs(self):
        agent = _make_agent(
            min_replay_size=8, train_freq=1, batch_size=4, target_update_freq=4,
            lr=0.1,
        )
        _observe(agent, 8)
        _observe(agent, 3)  # 3 gradient steps, no sync yet
        assert (agent.gradient_steps, agent.target_syncs) == (3, 0)
        assert not all(
            torch.equal(a, b)
            for a, b in zip(
                agent.online_net.parameters(), agent.target_net.parameters()
            )
        )
        _observe(agent, 1)  # 4th gradient step -> sync
        assert (agent.gradient_steps, agent.target_syncs) == (4, 1)
        for a, b in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.equal(a, b)

    def test_update_returns_cached_metrics_only(self):
        agent = _make_agent(min_replay_size=8, train_freq=2, batch_size=4)
        assert agent.update() == {}
        _observe(agent, 30)
        before = agent.gradient_steps
        metrics = agent.update()
        assert agent.gradient_steps == before  # update() never trains
        assert metrics["grad_steps"] == float(before)
        assert set(metrics) >= {"loss", "mean_q", "epsilon", "buffer_size"}
        assert np.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# Double-DQN target math
# ---------------------------------------------------------------------------


class TestDoubleDQNTarget:
    def test_target_uses_online_argmax_and_target_value(self):
        agent = _make_agent(gamma=0.9)
        # Online prefers action 1 (5.0), target scores 1 at 20 and 2 at 30.
        agent.online_net = _StubQ([[1.0, 5.0, 3.0, 0.0]])
        agent.target_net = _StubQ([[10.0, 20.0, 30.0, 40.0]])
        next_obs = torch.zeros(1, agent.obs_dim)
        mask = torch.tensor([[True, True, True, True]])
        out = agent._double_dqn_target(
            next_obs, mask, torch.tensor([2.0]), torch.tensor([0.0])
        )
        assert out.item() == 2.0 + 0.9 * 20.0

    def test_next_state_mask_constrains_the_argmax(self):
        """Masking the successor changes which action is bootstrapped."""
        agent = _make_agent(gamma=0.9)
        agent.online_net = _StubQ([[1.0, 5.0, 3.0, 0.0]])
        agent.target_net = _StubQ([[10.0, 20.0, 30.0, 40.0]])
        next_obs = torch.zeros(1, agent.obs_dim)
        mask = torch.tensor([[True, False, True, False]])  # 1 unavailable
        out = agent._double_dqn_target(
            next_obs, mask, torch.tensor([2.0]), torch.tensor([0.0])
        )
        # Best VALID online action is 2 (3.0 > 1.0) -> target value 30.
        assert out.item() == 2.0 + 0.9 * 30.0

    def test_terminal_transitions_drop_the_bootstrap(self):
        agent = _make_agent(gamma=0.9)
        agent.online_net = _StubQ([[1.0, 5.0, 3.0, 0.0]] * 2)
        agent.target_net = _StubQ([[10.0, 20.0, 30.0, 40.0]] * 2)
        next_obs = torch.zeros(2, agent.obs_dim)
        # Row 0 terminal with an EMPTY mask (legal at episode end).
        mask = torch.tensor(
            [[False, False, False, False], [True, True, True, True]]
        )
        out = agent._double_dqn_target(
            next_obs, mask, torch.tensor([2.0, 2.0]), torch.tensor([1.0, 0.0])
        )
        assert torch.isfinite(out).all()
        assert out[0].item() == 2.0
        assert out[1].item() == 2.0 + 0.9 * 20.0

    def test_gradient_step_slices_the_stored_next_mask(self):
        """The mask used by the target comes out of the replayed vector."""
        agent = _make_agent(min_replay_size=4, train_freq=1, batch_size=4)
        mask = np.array([1, 0, 1, 0], dtype=np.int8)
        obs = _make_obs(mask=mask)
        for _ in range(8):
            agent.observe_transition(obs, 0, 1.0, obs, False, False, {})
        batch = agent.replay.sample(4)
        stored = batch["next_obs"][:, agent.mask_offset :].numpy()
        assert (stored == mask.astype(np.float32)).all()


# ---------------------------------------------------------------------------
# Acting
# ---------------------------------------------------------------------------


class TestActing:
    def test_masked_greedy_never_picks_an_invalid_slot(self):
        agent = _make_agent()
        rng = np.random.default_rng(0)
        for _ in range(25):
            mask = np.zeros(MAX_T, dtype=np.int8)
            mask[rng.integers(0, MAX_T)] = 1  # exactly one valid slot
            mask[rng.integers(0, MAX_T)] = 1  # (possibly the same one)
            obs = _make_obs(mask=mask)
            assert mask[agent.select_action(obs, deterministic=True)] == 1

    def test_epsilon_exploration_also_respects_the_mask(self):
        agent = _make_agent()  # epsilon starts at 1.0 -> always explores
        random.seed(7)
        mask = np.array([0, 1, 0, 1], dtype=np.int8)
        obs = _make_obs(mask=mask)
        picks = {agent.select_action(obs) for _ in range(50)}
        assert picks == {1, 3}

    def test_deterministic_is_greedy_and_repeatable(self):
        agent = _make_agent()
        obs = _make_obs()
        random.seed(0)
        first = agent.select_action(obs, deterministic=True)
        random.seed(123)  # epsilon must not enter the deterministic path
        assert all(
            agent.select_action(obs, deterministic=True) == first
            for _ in range(5)
        )
        flat = torch.from_numpy(
            agent._flat_vector(obs, agent._flattener.extract_action_mask(obs))
        ).unsqueeze(0)
        with torch.no_grad():
            assert int(agent.online_net(flat).argmax(-1).item()) == first

    def test_epsilon_annealing_endpoints(self):
        agent = _make_agent(
            epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100
        )
        assert agent.epsilon == 1.0
        _observe(agent, 50)
        assert agent.epsilon == 0.55  # linear midpoint
        _observe(agent, 50)
        assert agent.epsilon == 0.1
        _observe(agent, 25)  # clamped past the end of the schedule
        assert agent.epsilon == 0.1


# ---------------------------------------------------------------------------
# RNG isolation — the simulator owns the global ``random`` stream
# ---------------------------------------------------------------------------


class TestRNGIsolation:
    """The agent's draws must not touch the process-global stream.

    ``kata.machine.Machine._breakdown_driver`` draws its failure times
    from the global ``random`` module and the runner re-seeds it per
    episode (runner.py:1421-1428) — an agent consuming it would make
    the WORLD a function of its own epsilon schedule and gradient
    cadence.
    """

    def test_exploration_never_consumes_the_global_stream(self):
        agent = _make_agent(epsilon_start=1.0, epsilon_end=1.0)
        obs = _make_obs()
        random.seed(11)
        before = random.getstate()
        for _ in range(20):
            agent.select_action(obs)
        assert random.getstate() == before

    def test_replay_sampling_never_consumes_the_global_stream(self):
        agent = _make_agent(min_replay_size=8, train_freq=2, batch_size=4)
        _observe(agent, 8)  # fill to the warmup, no gradient step yet
        random.seed(11)
        before = random.getstate()
        _observe(agent, 20)  # gradient steps -> replay.sample() calls
        assert agent.gradient_steps > 0
        assert random.getstate() == before

    def test_epsilon_decisions_survive_interleaved_global_draws(self):
        """Identically-seeded agents explore identically whatever the
        simulator draws in between."""
        obs = _make_obs()
        a = _make_agent(seed=3, epsilon_start=1.0, epsilon_end=1.0)
        random.seed(11)
        picks_a = [a.select_action(obs) for _ in range(30)]

        b = _make_agent(seed=3, epsilon_start=1.0, epsilon_end=1.0)
        random.seed(11)
        picks_b = []
        for _ in range(30):
            random.random()  # a machine failure draw landing in between
            random.randrange(97)
            picks_b.append(b.select_action(obs))

        assert picks_a == picks_b
        assert len(set(picks_a)) > 1  # the draws really are random

    def test_exploration_is_independent_of_the_gradient_cadence(self):
        """Changing ``train_freq`` must not shift the epsilon sequence."""
        obs = _make_obs()
        seqs = []
        for train_freq in (2, 7):
            agent = _make_agent(
                seed=3, epsilon_start=1.0, epsilon_end=1.0,
                min_replay_size=8, batch_size=4, train_freq=train_freq,
            )
            seq = []
            for _ in range(24):
                seq.append(agent.select_action(obs))
                agent.observe_transition(obs, 0, 1.0, obs, False, False, {})
            assert agent.gradient_steps > 0
            seqs.append(seq)
        assert seqs[0] == seqs[1]

    def test_exploration_and_replay_streams_are_distinct(self):
        agent = _make_agent(seed=3)
        assert (
            agent._explore_rng.getstate() != agent._replay_rng.getstate()
        )
        assert agent.replay._rng is agent._replay_rng


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_round_trip(self):
        agent = _make_agent(min_replay_size=8, train_freq=2, batch_size=4)
        agent.attach_vocab(VOCAB)
        _observe(agent, 30)
        obs = _make_obs()
        action = agent.select_action(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dql.pt"
            agent.save(path)
            fresh = _make_agent(
                min_replay_size=8, train_freq=2, batch_size=4, seed=99
            )
            # Different init -> the round trip has something to prove.
            assert not all(
                torch.equal(a, b)
                for a, b in zip(
                    agent.online_net.parameters(), fresh.online_net.parameters()
                )
            )
            fresh.load(path)

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert DQLMLPAgent.peek_vocab(path) == VOCAB

        for a, b in zip(
            agent.online_net.parameters(), fresh.online_net.parameters()
        ):
            assert torch.equal(a, b)
        for a, b in zip(
            agent.target_net.parameters(), fresh.target_net.parameters()
        ):
            assert torch.equal(a, b)
        assert fresh.env_steps == agent.env_steps
        assert fresh.gradient_steps == agent.gradient_steps
        assert fresh.target_syncs == agent.target_syncs
        assert fresh.epsilon == agent.epsilon
        assert fresh._vocab == VOCAB
        # Optimizer moments travelled (Adam step counts restored).
        assert fresh.optimizer.state_dict()["state"].keys() == (
            agent.optimizer.state_dict()["state"].keys()
        )
        assert fresh.select_action(obs, deterministic=True) == action
        assert ckpt["improvements"]["algo"] == "double_dqn"
        assert ckpt["improvements"]["prioritized_replay"] is False
        assert ckpt["improvements"]["hidden_sizes"] == [8, 8]
        assert ckpt["improvements"]["max_techs"] == MAX_T

    def test_rng_states_round_trip(self):
        """A resume continues the exploration sequence, not restarts it.

        Global seeding no longer pins the agent's draws, so the private
        streams have to ride in the checkpoint.
        """
        agent = _make_agent(
            seed=3, epsilon_start=1.0, epsilon_end=1.0,
            min_replay_size=8, train_freq=2, batch_size=4,
        )
        obs = _make_obs()
        _observe(agent, 20)
        for _ in range(5):
            agent.select_action(obs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dql.pt"
            agent.save(path)
            fresh = _make_agent(
                seed=99, epsilon_start=1.0, epsilon_end=1.0,
                min_replay_size=8, train_freq=2, batch_size=4,
            )
            assert fresh._explore_rng.getstate() != agent._explore_rng.getstate()
            fresh.load(path)

        assert fresh._explore_rng.getstate() == agent._explore_rng.getstate()
        assert fresh._replay_rng.getstate() == agent._replay_rng.getstate()
        # Both continue from the SAME saved state, hence the same picks.
        assert (
            [fresh.select_action(obs) for _ in range(10)]
            == [agent.select_action(obs) for _ in range(10)]
        )

    def test_load_tolerates_a_checkpoint_without_rng_states(self):
        """Pre-fix checkpoints keep the constructor-seeded streams."""
        agent = _make_agent(seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dql.pt"
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt.pop("explore_rng_state")
            ckpt.pop("replay_rng_state")
            torch.save(ckpt, path)
            fresh = _make_agent(seed=3)
            expected = fresh._explore_rng.getstate()
            fresh.load(path)
        assert fresh._explore_rng.getstate() == expected

    def test_incompatible_vocab_is_refused(self):
        """Vocab drift re-indexes keys — the one-hot bins would flip."""
        agent = _make_agent()
        drifted = dict(VOCAB)
        drifted["BUSY=T"], drifted["BUSY=F"] = drifted["BUSY=F"], drifted["BUSY=T"]
        try:
            agent.attach_vocab(drifted)
        except RuntimeError as exc:
            assert "BUSY=" in str(exc)
        else:  # pragma: no cover - the guard is the point
            raise AssertionError("expected a RuntimeError")


# ---------------------------------------------------------------------------
# Scale transfer
# ---------------------------------------------------------------------------


class TestScaleTransfer:
    def test_flat_width_is_independent_of_the_real_fleet_size(self):
        agent = _make_agent()
        small = agent._flat_vector(
            _make_obs(n_techs=2, n_machines=1),
            np.array([1, 1, 0, 0], dtype=bool),
        )
        big = agent._flat_vector(
            _make_obs(n_techs=4, n_machines=3),
            np.array([1, 1, 1, 1], dtype=bool),
        )
        assert small.shape == big.shape == (agent.obs_dim,)
        assert not np.array_equal(small, big)

    def test_checkpoint_acts_at_a_different_fleet_size(self):
        """Same caps, different real fleet: weights stay loadable/usable."""
        agent = _make_agent()
        agent.attach_vocab(VOCAB)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dql.pt"
            agent.save(path)
            fresh = _make_agent(seed=5)
            fresh.load(path)
        for n in (1, 2, 4):
            mask = np.zeros(MAX_T, dtype=np.int8)
            mask[:n] = 1
            obs = _make_obs(n_techs=n, n_machines=1, mask=mask)
            a_ref = agent.select_action(obs, deterministic=True)
            a_new = fresh.select_action(obs, deterministic=True)
            assert a_new == a_ref
            assert mask[a_new] == 1
