"""Tests for the KataEnv gymnasium wrapper."""

from conftest import FakeDispatcher, FakeMachine, FakeRequest, FakeSimEnv

from kata.core.config import GymEnvConfig
from kata.env import KataEnv


def test_reset_advances_to_next_breakdown_ticket():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    sim_env.schedule(
        at=7.0,
        callback=lambda: dispatcher.repair_queue.items.append(
            FakeRequest(machine_id=11, created_at=sim_env.now)
        ),
    )
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
    )

    obs, info = env.reset()

    assert obs["has_open_ticket"][0] == 1
    assert obs["ticket_machine_id"][0] == 11.0
    assert info["sim_time"] == 7.0


def test_step_assigns_technician_for_current_ticket():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=5, created_at=0.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
    )
    env.reset()

    obs, reward, terminated, truncated, _ = env.step(1)

    assert dispatcher.assignments == [(1, 5)]
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert obs["has_open_ticket"][0] == 0


def test_invalid_action_is_penalized_by_default():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=3, created_at=0.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(invalid_action_penalty=-3.0),
    )
    env.reset()

    _, reward, terminated, _, _ = env.step(99)
    assert dispatcher.assignments == []

    _, _, _, _, _ = env.step(1)

    assert reward == -3.0
    assert terminated is False
    assert dispatcher.assignments == [(1, 3)]


def test_token_observation_ticket_only_is_fixed_shape():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=9, created_at=2.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="ticket_only",
            token_observation_length=32,
        ),
    )

    obs, _ = env.reset()

    assert "tokens" in obs
    assert len(obs["tokens"]) == 32
    # Key-value format: TICKET_MACHINE_TYPE followed by machine type value
    assert "TICKET_MACHINE_TYPE" in obs["tokens"]
    # ticket_only mode should NOT include MACHINE_TYPE key (that's broken_machine level)
    assert "MACHINE_TYPE" not in obs["tokens"]


def test_token_observation_factory_level_with_technician_tokens():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.techs[0].fatigue = 0.3
    dispatcher.techs[0].knowledge = 0.4
    dispatcher.techs[1].fatigue = 0.7
    dispatcher.techs[1].knowledge = 0.9
    machine = FakeMachine(machine_id=4)
    machine.input_buffer.items.extend([object(), object()])
    machine.output_buffer.items.append(object())
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=4, created_at=1.0))
    dispatcher.repair_queue.items[-1].machine = machine
    dispatcher.machines = [machine]
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="factory_level",
            include_technician_fatigue_tokens=True,
            include_technician_knowledge_tokens=True,
            token_observation_length=64,
            token_pad_value="<PAD>",
        ),
    )

    obs, _ = env.reset()
    tokens = obs["tokens"]

    assert len(tokens) == 64
    # Key-value pairs: key token followed by bucketed value token
    assert "FACTORY_MACHINES" in tokens
    assert "MACHINE_INPUT_BUF" in tokens
    # Decile buckets: fatigue 0.3 -> R_30_40, knowledge 0.4 -> R_40_50,
    # fatigue 0.7 -> R_70_80, knowledge 0.9 -> R_90_100.
    assert "FATIGUE" in tokens
    assert "KNOWLEDGE" in tokens
    assert "R_30_40" in tokens
    assert "R_70_80" in tokens
    assert tokens[-1] == "<PAD>"


def test_token_observation_tech_aware_includes_per_tech_ticket_signals():
    """tech_aware mode adds per-tech MATCH/ETA/LAST_AGE and queue peek."""
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    machine = FakeMachine(machine_id=42)
    req = FakeRequest(machine_id=42, created_at=0.0)
    req.machine = machine
    dispatcher.repair_queue.items.append(req)
    # Two more queued tickets so the NEXT1/NEXT2 slots have real data
    n1 = FakeRequest(machine_id=43, created_at=0.0); n1.machine = FakeMachine(machine_id=43)
    n2 = FakeRequest(machine_id=44, created_at=0.0); n2.machine = FakeMachine(machine_id=44)
    dispatcher.repair_queue.items.extend([n1, n2])
    dispatcher.machines = [machine]

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="tech_aware",
            include_technician_fatigue_tokens=True,
            include_technician_knowledge_tokens=True,
            token_observation_length=128,
            max_episode_steps=10,
            max_sim_time=100.0,
        ),
    )
    obs, _ = env.reset()
    tokens = obs["tokens"]

    # tech_aware is a superset of factory_level
    for key in ("FACTORY_MACHINES", "MACHINE_BROKEN"):
        assert key in tokens
    # Ticket-aware additions
    assert "TICKET_COMPONENT_TYPE" in tokens
    for key in (
        "NEXT1_MACHINE_TYPE", "NEXT1_COMPONENT_TYPE", "NEXT1_AGE",
        "NEXT2_MACHINE_TYPE", "NEXT2_COMPONENT_TYPE", "NEXT2_AGE",
    ):
        assert key in tokens
    # Per-tech ticket-specific signals
    for key in ("MATCH", "ETA", "LAST_AGE"):
        assert key in tokens
    assert tokens[0] == "OBS_MODE" and tokens[1] == "tech_aware"


def test_tech_aware_last_age_starts_as_none_then_updates_after_assignment():
    sim_env = FakeSimEnv()
    sim_env.now = 0.0
    dispatcher = FakeDispatcher(tech_count=2)
    for i in range(4):
        dispatcher.repair_queue.items.append(
            FakeRequest(machine_id=i + 1, created_at=0.0)
        )
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="tech_aware",
            token_observation_length=128,
        ),
    )
    obs, _ = env.reset()
    # Both techs unused → both report LAST_AGE = T_NONE
    tokens = obs["tokens"]
    last_age_idxs = [i for i, t in enumerate(tokens) if t == "LAST_AGE"]
    for i in last_age_idxs:
        assert tokens[i + 1] == "T_NONE"

    # Assign tech 0, then move sim time forward and re-observe
    env.step(0)
    sim_env.now = 75.0  # bump time
    tokens = env._token_obs()["tokens"]
    last_age_idxs = [i for i, t in enumerate(tokens) if t == "LAST_AGE"]
    # tech_0 should now have a non-NONE age bucket
    age_tech_0 = tokens[last_age_idxs[0] + 1]
    assert age_tech_0 != "T_NONE"


def test_token_observation_broken_machine_contains_machine_tokens_only():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=1)
    machine = FakeMachine(machine_id=6)
    machine.input_buffer.items.append(object())
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=6, created_at=0.0))
    dispatcher.repair_queue.items[-1].machine = machine
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="broken_machine",
            token_observation_length=20,
        ),
    )

    obs, _ = env.reset()
    tokens = obs["tokens"]

    assert "MACHINE_INPUT_BUF" in tokens
    assert "MACHINE_BROKEN" in tokens
    # broken_machine mode should NOT have factory-level keys
    assert "FACTORY_MACHINES" not in tokens


def test_reward_is_composed_from_enabled_sub_rewards():
    sim_env = FakeSimEnv()
    sim_env.now = 10.0
    dispatcher = FakeDispatcher(tech_count=1)
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=2.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            assignment_reward=3.0,
            ticket_wait_time_penalty=0.5,
            reward={
                "assignment": {"enabled": True, "coefficient": 2.0},
                "wait_time": {"enabled": False, "coefficient": 1.0},
                "queue_size": {"enabled": False, "coefficient": 1.0},
                "busy_technician": {"enabled": False, "coefficient": 1.0},
            },
        ),
    )
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == 6.0
    assert info["reward_breakdown"]["assignment"] == 6.0
    assert info["reward_breakdown"]["wait_time"] == 0.0


def test_info_exposes_assignment_counts_per_tech():
    """info['assignment_counts'] reflects the per-tech assignment history."""
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=3)
    # Enough work to keep the episode running through every step we do
    for i in range(8):
        dispatcher.repair_queue.items.append(
            FakeRequest(machine_id=i + 1, created_at=0.0)
        )
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
    )
    obs, info = env.reset()
    counts0 = info["assignment_counts"]
    # Three uniquely-named FakeTechs with default names tech_0/tech_1/tech_2
    assert set(counts0.keys()) == {"tech_0", "tech_1", "tech_2"}
    assert all(v == 0 for v in counts0.values())

    env.step(0)
    env.step(0)
    env.step(2)
    _, _, _, _, info = env.step(0)
    assert info["assignment_counts"]["tech_0"] == 3
    assert info["assignment_counts"]["tech_1"] == 0
    assert info["assignment_counts"]["tech_2"] == 1


def test_machine_stats_tracks_downtime_and_breakdowns():
    """Per-machine stats reflect transitions in broken / processing state."""
    sim_env = FakeSimEnv()
    sim_env.now = 0.0
    dispatcher = FakeDispatcher(tech_count=1)
    m1 = FakeMachine(machine_id=1)
    m1.name = "alpha"
    m1.broken = False
    m1.is_processing = False
    m2 = FakeMachine(machine_id=2)
    m2.name = "beta"
    m2.broken = False
    m2.is_processing = False
    dispatcher.machines = [m1, m2]
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=1000.0),
    )
    env.reset()

    # Step 1: m1 just broke, m2 is producing
    sim_env.now = 10.0
    m1.broken = True
    m2.is_processing = True
    env._update_machine_state_tracking()

    sim_env.now = 30.0
    env._update_machine_state_tracking()

    # m1 finishes its repair at t=40, m2 finishes producing at t=50
    sim_env.now = 40.0
    m1.broken = False
    env._update_machine_state_tracking()
    sim_env.now = 50.0
    m2.is_processing = False
    env._update_machine_state_tracking()

    stats = env._per_machine_episode_stats()
    # alpha (m1): broken from t=10..40 → 30 units of maintenance, 1 breakdown
    assert abs(stats["alpha"]["maintenance_time"] - 30.0) < 1e-6
    assert stats["alpha"]["breakdowns"] == 1.0
    # beta (m2): producing from t=10..50 → 40 units of production
    assert abs(stats["beta"]["production_time"] - 40.0) < 1e-6
    assert stats["beta"]["breakdowns"] == 0.0


def test_machine_stats_does_not_double_count_broken_machine_as_producing():
    """is_processing stays True during a repair wait — we must exclude that."""
    sim_env = FakeSimEnv()
    sim_env.now = 0.0
    dispatcher = FakeDispatcher(tech_count=1)
    m = FakeMachine(machine_id=7)
    m.name = "gamma"
    # Simulator state: in the middle of a job AND broken
    m.is_processing = True
    m.broken = True
    dispatcher.machines = [m]
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=7, created_at=0.0))

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=1000.0),
    )
    env.reset()
    # Register the initial broken state at t=0 (which would normally
    # happen on the first step() call after the env starts).
    env._update_machine_state_tracking()
    sim_env.now = 25.0
    env._update_machine_state_tracking()
    stats = env._per_machine_episode_stats()
    # All 25 units count as maintenance, 0 as production
    assert abs(stats["gamma"]["maintenance_time"] - 25.0) < 1e-6
    assert stats["gamma"]["production_time"] == 0.0


def test_action_mask_marks_busy_techs_invalid():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=3)
    dispatcher.techs[0].busy = True
    dispatcher.techs[2].busy = True
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
    )
    obs, _ = env.reset()
    mask = obs["action_mask"]
    # Only tech_1 is free → mask should be [0, 1, 0]
    assert list(mask) == [0, 1, 0]


def test_action_mask_all_busy_falls_back_to_all_one():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    for t in dispatcher.techs:
        t.busy = True
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
    )
    obs, _ = env.reset()
    # All-busy edge case: fall back to all-1 so the agent can still pick
    assert list(obs["action_mask"]) == [1, 1]


def test_action_mask_disabled_when_flag_off():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            max_episode_steps=10, max_sim_time=100.0, expose_action_mask=False
        ),
    )
    obs, _ = env.reset()
    assert "action_mask" not in obs
    assert "action_mask" not in env.observation_space.spaces


def test_duplicate_technician_names_raise_at_init():
    """Two technicians sharing a name must fail fast."""
    import pytest

    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=2)
    dispatcher.techs[0].name = "alice"
    dispatcher.techs[1].name = "alice"
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=1, created_at=0.0))

    with pytest.raises(ValueError, match="unique"):
        KataEnv(
            sim_env=sim_env,
            dispatcher=dispatcher,
            config=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
        )


def test_reward_busy_technician_component_applies_when_enabled():
    sim_env = FakeSimEnv()
    dispatcher = FakeDispatcher(tech_count=1)
    dispatcher.techs[0].busy = True
    dispatcher.repair_queue.items.append(FakeRequest(machine_id=2, created_at=0.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            reward={
                "assignment": {"enabled": False, "coefficient": 1.0},
                "wait_time": {"enabled": False, "coefficient": 1.0},
                "queue_size": {"enabled": False, "coefficient": 1.0},
                "busy_technician": {"enabled": True, "coefficient": 2.0},
            },
        ),
    )
    env.reset()

    _, reward, _, _, info = env.step(0)

    assert reward == -2.0
    assert info["reward_breakdown"]["busy_technician"] == -2.0
