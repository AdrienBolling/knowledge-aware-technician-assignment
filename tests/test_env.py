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
    # Fatigue 0.3 -> R_MEDLOW bucket; knowledge 0.9 -> R_HIGH bucket
    assert "FATIGUE" in tokens
    assert "KNOWLEDGE" in tokens
    assert tokens[-1] == "<PAD>"


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
