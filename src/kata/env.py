from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from kata import get_config
from kata.core.config import GymEnvConfig


class KataEnv(gym.Env):
    """Event-driven Gym environment for technician assignment.

    The environment asks for an action only when a new repair request appears.
    The action is the technician index to assign to the current request.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        sim_env: Any | None = None,
        dispatcher: Any | None = None,
        config: GymEnvConfig | None = None,
        *,
        scenario_factory: Callable[[], tuple[Any, Any]] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config: GymEnvConfig = config or get_config().gym
        self._scenario_factory = scenario_factory
        self._initial_env = sim_env
        self._initial_dispatcher = dispatcher
        self.render_mode = render_mode

        self.sim_env: Any | None = None
        self.dispatcher: Any | None = None
        self.current_request: Any | None = None
        self.episode_step = 0
        self._last_reward_breakdown: dict[str, float] = {}

        self._bootstrap_scenario()

    def _bootstrap_scenario(self) -> None:
        if self._scenario_factory is not None:
            self.sim_env, self.dispatcher = self._scenario_factory()
        else:
            self.sim_env, self.dispatcher = self._initial_env, self._initial_dispatcher

        if self.dispatcher is None:
            message = (
                "KataEnv requires a dispatcher instance or a scenario_factory "
                "that returns (sim_env, dispatcher)."
            )
            raise ValueError(message)

        n_techs = len(self.dispatcher.techs)
        self.action_space = gym.spaces.Discrete(n_techs)
        if self.config.observation_representation == "tokens":
            token_spaces = tuple(
                gym.spaces.Text(max_length=self.config.token_max_length)
                for _ in range(self.config.token_observation_length)
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "tokens": gym.spaces.Tuple(token_spaces),
                }
            )
            return

        observation_space: dict[str, gym.Space] = {
            "sim_time": gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                dtype=np.float32,
            ),
            "has_open_ticket": gym.spaces.MultiBinary(1),
            "ticket_created_at": gym.spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                dtype=np.float32,
            ),
            "ticket_machine_id": gym.spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                dtype=np.float32,
            ),
            "technician_busy": gym.spaces.MultiBinary(n_techs),
        }
        if self.config.include_fatigue_in_observation:
            observation_space["technician_fatigue"] = gym.spaces.Box(
                low=np.zeros(n_techs, dtype=np.float32),
                high=np.ones(n_techs, dtype=np.float32),
                dtype=np.float32,
            )
        if self.config.include_queue_size_in_observation:
            observation_space["pending_queue_size"] = gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([np.finfo(np.float32).max], dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(observation_space)

    def _queue(self) -> Any:
        return self.dispatcher.repair_queue

    def _queue_size(self) -> int:
        queue = self._queue()
        if hasattr(queue, "items"):
            return len(queue.items)
        return len(queue)

    def _queue_pop(self) -> Any | None:
        queue = self._queue()
        if hasattr(queue, "items"):
            if not queue.items:
                return None
            return queue.items.pop(0)
        if not queue:
            return None
        return queue.pop(0)

    def _sim_time(self) -> float:
        return float(getattr(self.sim_env, "now", 0.0))

    def _machine_id_from_request(self, request: Any | None) -> float:
        if request is None:
            return -1.0
        machine = getattr(request, "machine", None)
        if machine is None:
            return -1.0
        for attr in ("machine_id", "id"):
            if hasattr(machine, attr):
                return float(getattr(machine, attr))
        return -1.0

    def _request_created_at(self, request: Any | None) -> float:
        if request is None:
            return -1.0
        return float(getattr(request, "created_at", -1.0))

    def _advance_until_next_ticket(self) -> None:
        if self.current_request is not None:
            return

        while self.current_request is None and not self._is_done():
            request = self._queue_pop()
            if request is not None:
                self.current_request = request
                return

            if self.sim_env is None:
                return

            if not hasattr(self.sim_env, "peek") or not hasattr(self.sim_env, "step"):
                return

            next_event_time = self.sim_env.peek()
            try:
                next_event_time_float = float(next_event_time)
            except (TypeError, ValueError):
                return
            if np.isinf(next_event_time_float):
                return
            if next_event_time_float > float(self.config.max_sim_time):
                return
            self.sim_env.step()

    def _is_done(self) -> bool:
        if self.episode_step >= self.config.max_episode_steps:
            return True
        if self._sim_time() >= self.config.max_sim_time:
            return True
        if self.sim_env is None or not hasattr(self.sim_env, "peek"):
            return False
        if not np.isinf(self.sim_env.peek()):
            return False
        return self.current_request is None and self._queue_size() == 0

    def _machine_buffer_size(self, machine: Any, attr: str) -> int:
        buffer = getattr(machine, attr, None)
        if buffer is None:
            return 0
        if hasattr(buffer, "items"):
            return len(buffer.items)
        if hasattr(buffer, "__len__"):
            return len(buffer)
        return 0

    def _factory_machines(self) -> list[Any]:
        for source in (self.dispatcher, self.sim_env):
            machines = getattr(source, "machines", None)
            if machines is None:
                continue
            if isinstance(machines, dict):
                return list(machines.values())
            return list(machines)
        if self.current_request is not None and getattr(self.current_request, "machine", None) is not None:
            return [self.current_request.machine]
        return []

    def _technician_knowledge_value(self, tech: Any) -> float:
        if hasattr(tech, "knowledge"):
            return float(getattr(tech, "knowledge"))
        knowledge_grid = getattr(tech, "knowledge_grid", None)
        if knowledge_grid is not None:
            for method_name in ("mean_knowledge", "mean", "to_array"):
                method = getattr(knowledge_grid, method_name, None)
                if callable(method):
                    value = method()
                    if np.isscalar(value):
                        return float(value)
                    return float(np.asarray(value, dtype=np.float32).mean())
        return 0.0

    def _token_obs(self) -> dict[str, tuple[str, ...]]:
        ticket = self.current_request
        tokens = [
            f"OBS_MODE:{self.config.observation_mode}",
            f"SIM_TIME:{self._sim_time():.3f}",
            f"HAS_OPEN_TICKET:{1 if ticket is not None else 0}",
            f"TICKET_CREATED_AT:{self._request_created_at(ticket):.3f}",
            f"TICKET_MACHINE_ID:{self._machine_id_from_request(ticket):.0f}",
        ]

        machine = getattr(ticket, "machine", None)
        if self.config.observation_mode in {"broken_machine", "factory_level"}:
            tokens.extend(
                [
                    f"MACHINE_ID:{self._machine_id_from_request(ticket):.0f}",
                    f"MACHINE_BROKEN:{1 if bool(getattr(machine, 'broken', False)) else 0}",
                    f"MACHINE_PROCESSING:{1 if bool(getattr(machine, 'is_processing', False)) else 0}",
                    f"MACHINE_TOTAL_PROCESSED:{int(getattr(machine, 'total_processed', 0))}",
                    f"MACHINE_INPUT_BUFFER:{self._machine_buffer_size(machine, 'input_buffer')}",
                    f"MACHINE_OUTPUT_BUFFER:{self._machine_buffer_size(machine, 'output_buffer')}",
                ]
            )

        if self.config.observation_mode == "factory_level":
            machines = self._factory_machines()
            broken_count = sum(1 for m in machines if bool(getattr(m, "broken", False)))
            processing_count = sum(1 for m in machines if bool(getattr(m, "is_processing", False)))
            total_processed = sum(int(getattr(m, "total_processed", 0)) for m in machines)
            tokens.extend(
                [
                    f"FACTORY_MACHINE_COUNT:{len(machines)}",
                    f"FACTORY_BROKEN_COUNT:{broken_count}",
                    f"FACTORY_PROCESSING_COUNT:{processing_count}",
                    f"FACTORY_TOTAL_PROCESSED:{total_processed}",
                    f"FACTORY_QUEUE_SIZE:{self._queue_size()}",
                ]
            )

        if self.config.include_technician_fatigue_tokens:
            tokens.extend(
                [
                    f"TECH_{idx}_FATIGUE:{float(getattr(tech, 'fatigue', 0.0)):.3f}"
                    for idx, tech in enumerate(self.dispatcher.techs)
                ]
            )

        if self.config.include_technician_knowledge_tokens:
            tokens.extend(
                [
                    f"TECH_{idx}_KNOWLEDGE:{self._technician_knowledge_value(tech):.3f}"
                    for idx, tech in enumerate(self.dispatcher.techs)
                ]
            )

        target_length = self.config.token_observation_length
        if len(tokens) < target_length:
            tokens.extend([self.config.token_pad_value] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]
        return {"tokens": tuple(tokens)}

    def _structured_obs(self) -> dict[str, np.ndarray]:
        ticket = self.current_request
        busy = np.asarray(
            [1 if getattr(tech, "busy", False) else 0 for tech in self.dispatcher.techs],
            dtype=np.int8,
        )
        observation: dict[str, np.ndarray] = {
            "sim_time": np.asarray([self._sim_time()], dtype=np.float32),
            "has_open_ticket": np.asarray([1 if ticket is not None else 0], dtype=np.int8),
            "ticket_created_at": np.asarray([self._request_created_at(ticket)], dtype=np.float32),
            "ticket_machine_id": np.asarray(
                [self._machine_id_from_request(ticket)], dtype=np.float32
            ),
            "technician_busy": busy,
        }
        if self.config.include_fatigue_in_observation:
            observation["technician_fatigue"] = np.asarray(
                [float(getattr(tech, "fatigue", 0.0)) for tech in self.dispatcher.techs],
                dtype=np.float32,
            )
        if self.config.include_queue_size_in_observation:
            observation["pending_queue_size"] = np.asarray(
                [float(self._queue_size())], dtype=np.float32
            )
        return observation

    def _obs(self) -> dict[str, Any]:
        if self.config.observation_representation == "tokens":
            return self._token_obs()
        return self._structured_obs()

    def _info(self) -> dict[str, Any]:
        return {
            "episode_step": self.episode_step,
            "sim_time": self._sim_time(),
            "has_open_ticket": self.current_request is not None,
            "pending_queue_size": self._queue_size(),
            "reward_breakdown": dict(self._last_reward_breakdown),
        }

    def _reward_for_assignment(self, request: Any, tech_id: int) -> float:
        created_at = float(getattr(request, "created_at", self._sim_time()))
        waiting = max(0.0, self._sim_time() - created_at)
        assignment_raw = float(self.config.assignment_reward)
        wait_time_raw = -float(self.config.ticket_wait_time_penalty) * waiting
        tech = self.dispatcher.techs[tech_id]
        busy_raw = -1.0 if bool(getattr(tech, "busy", False)) else 0.0
        queue_raw = -float(self._queue_size())

        breakdown = {
            "assignment": (
                self.config.reward.assignment.coefficient * assignment_raw
                if self.config.reward.assignment.enabled
                else 0.0
            ),
            "wait_time": (
                self.config.reward.wait_time.coefficient * wait_time_raw
                if self.config.reward.wait_time.enabled
                else 0.0
            ),
            "queue_size": (
                self.config.reward.queue_size.coefficient * queue_raw
                if self.config.reward.queue_size.enabled
                else 0.0
            ),
            "busy_technician": (
                self.config.reward.busy_technician.coefficient * busy_raw
                if self.config.reward.busy_technician.enabled
                else 0.0
            ),
        }
        self._last_reward_breakdown = breakdown
        return float(sum(breakdown.values()))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        _ = options

        if self._scenario_factory is not None:
            self._bootstrap_scenario()

        self.episode_step = 0
        self.current_request = None
        self._last_reward_breakdown = {}
        self._advance_until_next_ticket()
        return self._obs(), self._info()

    def step(self, action: int):
        if self.current_request is None:
            self._advance_until_next_ticket()
            if self.current_request is None:
                return self._obs(), 0.0, self._is_done(), False, self._info()

        action = int(action)
        invalid_action = action < 0 or action >= len(self.dispatcher.techs)
        if invalid_action:
            if self.config.invalid_action_mode == "raise":
                message = f"Invalid technician index: {action}"
                raise ValueError(message)
            reward = float(self.config.invalid_action_penalty)
            self._last_reward_breakdown = {"invalid_action": reward}
            terminated = self.config.invalid_action_mode == "terminate"
            return self._obs(), reward, terminated, False, self._info()

        request = self.current_request
        self.current_request = None
        self.dispatcher.start_repair(action, request)
        self.episode_step += 1

        reward = self._reward_for_assignment(request, action)
        self._advance_until_next_ticket()
        terminated = self._is_done()
        return self._obs(), reward, terminated, False, self._info()

    def render(self):
        if self.render_mode != "human":
            return None
        print(
            f"[KataEnv] t={self._sim_time():.2f} | step={self.episode_step} "
            f"| open_ticket={self.current_request is not None} "
            f"| queue={self._queue_size()}"
        )
        return None
