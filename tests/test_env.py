import math

from kata.core.config import GymEnvConfig
from kata.env import KataEnv


class _FakeMachine:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id


class _FakeRequest:
    def __init__(self, machine_id: int, created_at: float):
        self.machine = _FakeMachine(machine_id)
        self.created_at = created_at


class _FakeTech:
    def __init__(self, tech_id: int):
        self.id = tech_id
        self.busy = False
        self.fatigue = 0.0


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

    assert reward == -3.0
    assert terminated is False
    assert dispatcher.assignments == []
