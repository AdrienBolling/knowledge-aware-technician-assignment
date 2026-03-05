import math

from kata.core.config import GymEnvConfig
from kata.env import KataEnv


class _FakeMachine:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.broken = True
        self.is_processing = False
        self.total_processed = 2
        self.input_buffer = _FakeQueue()
        self.output_buffer = _FakeQueue()


class _FakeRequest:
    def __init__(self, machine_id: int, created_at: float):
        self.machine = _FakeMachine(machine_id)
        self.created_at = created_at


class _FakeTech:
    def __init__(self, tech_id: int):
        self.id = tech_id
        self.busy = False
        self.fatigue = 0.0
        self.knowledge = 0.0


class _FakeQueue:
    def __init__(self):
        self.items = []


class _FakeDispatcher:
    def __init__(self, tech_count: int = 2):
        self.techs = [_FakeTech(i) for i in range(tech_count)]
        self.repair_queue = _FakeQueue()
        self.assignments = []

    def start_repair(self, tech_id, request):
        self.assignments.append((tech_id, request.machine.machine_id))
        self.techs[tech_id].busy = True


class _FakeSimEnv:
    def __init__(self):
        self.now = 0.0
        self._events = []

    def schedule(self, at: float, callback):
        self._events.append((float(at), callback))
        self._events.sort(key=lambda x: x[0])

    def peek(self):
        if not self._events:
            return math.inf
        return self._events[0][0]

    def step(self):
        when, callback = self._events.pop(0)
        self.now = when
        callback()


def test_reset_advances_to_next_breakdown_ticket():
    sim_env = _FakeSimEnv()
    dispatcher = _FakeDispatcher(tech_count=2)
    sim_env.schedule(
        at=7.0,
        callback=lambda: dispatcher.repair_queue.items.append(
            _FakeRequest(machine_id=11, created_at=sim_env.now)
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
    sim_env = _FakeSimEnv()
    dispatcher = _FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(_FakeRequest(machine_id=5, created_at=0.0))
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
    sim_env = _FakeSimEnv()
    dispatcher = _FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(_FakeRequest(machine_id=3, created_at=0.0))
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
    sim_env = _FakeSimEnv()
    dispatcher = _FakeDispatcher(tech_count=2)
    dispatcher.repair_queue.items.append(_FakeRequest(machine_id=9, created_at=2.0))
    env = KataEnv(
        sim_env=sim_env,
        dispatcher=dispatcher,
        config=GymEnvConfig(
            observation_representation="tokens",
            observation_mode="ticket_only",
            token_observation_length=8,
        ),
    )

    obs, _ = env.reset()

    assert "tokens" in obs
    assert len(obs["tokens"]) == 8
    assert any(token.startswith("TICKET_MACHINE_ID:9") for token in obs["tokens"])
    assert all(token.startswith("MACHINE_") is False for token in obs["tokens"])


def test_token_observation_factory_level_with_technician_tokens():
    sim_env = _FakeSimEnv()
    dispatcher = _FakeDispatcher(tech_count=2)
    dispatcher.techs[0].fatigue = 0.3
    dispatcher.techs[0].knowledge = 0.4
    dispatcher.techs[1].fatigue = 0.7
    dispatcher.techs[1].knowledge = 0.9
    machine = _FakeMachine(machine_id=4)
    machine.input_buffer.items.extend([object(), object()])
    machine.output_buffer.items.append(object())
    dispatcher.repair_queue.items.append(_FakeRequest(machine_id=4, created_at=1.0))
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
            token_observation_length=32,
            token_pad_value="<PAD>",
        ),
    )

    obs, _ = env.reset()
    tokens = obs["tokens"]

    assert len(tokens) == 32
    assert any(token.startswith("FACTORY_MACHINE_COUNT:1") for token in tokens)
    assert any(token.startswith("MACHINE_INPUT_BUFFER:2") for token in tokens)
    assert any(token.startswith("TECH_0_FATIGUE:0.300") for token in tokens)
    assert any(token.startswith("TECH_1_KNOWLEDGE:0.900") for token in tokens)
    assert tokens[-1] == "<PAD>"


def test_reward_is_composed_from_enabled_sub_rewards():
    sim_env = _FakeSimEnv()
    sim_env.now = 10.0
    dispatcher = _FakeDispatcher(tech_count=1)
    dispatcher.repair_queue.items.append(_FakeRequest(machine_id=1, created_at=2.0))
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
