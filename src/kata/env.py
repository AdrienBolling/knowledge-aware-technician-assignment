from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from kata import get_config
from kata.core.config import GymEnvConfig
from kata.core.tokenizer import StateTokenizer
from kata.metrics import EPISODE_METRICS, STEP_METRICS

# ---------------------------------------------------------------------------
# Value bucketing helpers – keep vocabulary bounded for Transformer input
# ---------------------------------------------------------------------------


def _bucket_time(t: float) -> str:
    """Bucket a simulation time value into a categorical token."""
    if t < 0:
        return "T_NONE"
    if t < 50:
        return "T_0_50"
    if t < 200:
        return "T_50_200"
    if t < 500:
        return "T_200_500"
    if t < 1000:
        return "T_500_1K"
    if t < 5000:
        return "T_1K_5K"
    return "T_5K+"


def _bucket_ratio(v: float) -> str:
    """Bucket a [0, 1] ratio (fatigue, knowledge) into quintiles."""
    if v <= 0.0:
        return "R_0"
    if v < 0.2:
        return "R_LOW"
    if v < 0.4:
        return "R_MEDLOW"
    if v < 0.6:
        return "R_MED"
    if v < 0.8:
        return "R_MEDHIGH"
    return "R_HIGH"


def _bucket_count(n: int) -> str:
    """Bucket a small non-negative count into a categorical token."""
    if n <= 0:
        return "C_0"
    if n == 1:
        return "C_1"
    if n <= 3:
        return "C_2_3"
    if n <= 5:
        return "C_4_5"
    if n <= 10:
        return "C_6_10"
    if n <= 20:
        return "C_11_20"
    return "C_20+"


def _bool_token(v: bool) -> str:
    return "TRUE" if v else "FALSE"


class KataEnv(gym.Env):
    """Event-driven Gym environment for technician assignment.

    The environment asks for an action only when a new repair request appears.
    The action is the technician index to assign to the current request.
    """

    metadata = {"render_modes": ["human", "cli", "dict", "visual"], "render_fps": 4}

    def __init__(
        self,
        sim_env: Any | None = None,
        dispatcher: Any | None = None,
        config: GymEnvConfig | None = None,
        *,
        scenario_factory: Callable[[], tuple[Any, Any]] | None = None,
        render_mode: str | None = None,
        tokenizer: StateTokenizer | None = None,
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

        # Tokenizer & MCA state (populated by warmup or passed in)
        self._tokenizer: StateTokenizer | None = tokenizer
        self._mca_encoder: Any | None = None
        self._warmup_done = False

        # Metrics state (reset each episode)
        self._last_step_metrics: dict[str, float] = {}
        self._breakdown_counter: int = 0
        self._repair_counter: int = 0

        # State tracking for delta-based rewards
        self._prev_finished_products: int = 0
        self._machine_down_since: dict[int, float] = {}  # machine_id -> time it broke
        self._total_downtime: float = 0.0  # accumulated machine-time lost

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
        if self.config.observation_representation == "token_ids":
            seq_len = self.config.tokenizer_seq_length
            self.observation_space = gym.spaces.Dict(
                {
                    "token_ids": gym.spaces.Box(
                        low=0,
                        high=np.iinfo(np.int64).max,
                        shape=(seq_len,),
                        dtype=np.int64,
                    ),
                }
            )
            return
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
                self._breakdown_counter += 1
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
            try:
                self.sim_env.step()
            except Exception:
                # Unhandled SimPy event failure (e.g. breakdown interrupt
                # propagating from a machine process). Safe to skip.
                return

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
        if (
            self.current_request is not None
            and getattr(self.current_request, "machine", None) is not None
        ):
            return [self.current_request.machine]
        return []

    def _technician_knowledge_value(self, tech: Any) -> float:
        if hasattr(tech, "knowledge"):
            return float(tech.knowledge)
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
        """Build a key-value token sequence.

        Each observation field is emitted as a **key token** followed by one or
        more **value tokens**.  Continuous values are bucketed into categorical
        tokens so that the vocabulary stays bounded.  Categorical/boolean values
        are emitted directly.

        Token design principles:
        - Keys are fixed strings (``SIM_TIME``, ``MACHINE_BROKEN``, …)
        - Time values -> ``T_0_50``, ``T_50_200``, … (7 buckets)
        - Ratios [0, 1] -> ``R_0``, ``R_LOW``, … (6 buckets)
        - Counts -> ``C_0``, ``C_1``, ``C_2_3``, … (7 buckets)
        - Booleans -> ``TRUE`` / ``FALSE``
        - Machine types -> categorical string (``CNC``, ``Assembly``, …)
        """
        ticket = self.current_request
        machine = getattr(ticket, "machine", None)
        machine_type = str(getattr(machine, "mtype", "NONE"))

        tokens: list[str] = [
            # -- header / context --
            "OBS_MODE",
            self.config.observation_mode,
            # -- simulation time --
            "SIM_TIME",
            _bucket_time(self._sim_time()),
            # -- current ticket --
            "HAS_TICKET",
            _bool_token(ticket is not None),
            "TICKET_AGE",
            _bucket_time(
                self._sim_time() - self._request_created_at(ticket)
                if ticket is not None
                else -1
            ),
            "TICKET_MACHINE_TYPE",
            machine_type,
        ]

        if self.config.observation_mode in {"broken_machine", "factory_level"}:
            tokens.extend(
                [
                    "MACHINE_TYPE",
                    machine_type,
                    "MACHINE_BROKEN",
                    _bool_token(bool(getattr(machine, "broken", False))),
                    "MACHINE_PROCESSING",
                    _bool_token(bool(getattr(machine, "is_processing", False))),
                    "MACHINE_TOTAL_PROCESSED",
                    _bucket_count(int(getattr(machine, "total_processed", 0))),
                    "MACHINE_INPUT_BUF",
                    _bucket_count(self._machine_buffer_size(machine, "input_buffer")),
                    "MACHINE_OUTPUT_BUF",
                    _bucket_count(self._machine_buffer_size(machine, "output_buffer")),
                ]
            )

        if self.config.observation_mode == "factory_level":
            machines = self._factory_machines()
            broken_count = sum(1 for m in machines if bool(getattr(m, "broken", False)))
            processing_count = sum(
                1 for m in machines if bool(getattr(m, "is_processing", False))
            )
            total_processed = sum(
                int(getattr(m, "total_processed", 0)) for m in machines
            )
            tokens.extend(
                [
                    "FACTORY_MACHINES",
                    _bucket_count(len(machines)),
                    "FACTORY_BROKEN",
                    _bucket_count(broken_count),
                    "FACTORY_PROCESSING",
                    _bucket_count(processing_count),
                    "FACTORY_PRODUCED",
                    _bucket_count(total_processed),
                    "FACTORY_QUEUE",
                    _bucket_count(self._queue_size()),
                ]
            )

        # -- per-technician tokens --
        for idx, tech in enumerate(self.dispatcher.techs):
            tech_prefix = f"TECH_{idx}"
            busy = bool(getattr(tech, "busy", False))
            tokens.extend(
                [
                    tech_prefix,
                    "BUSY",
                    _bool_token(busy),
                ]
            )
            if self.config.include_technician_fatigue_tokens:
                fatigue = float(getattr(tech, "fatigue", 0.0))
                tokens.extend([tech_prefix, "FATIGUE", _bucket_ratio(fatigue)])
            if self.config.include_technician_knowledge_tokens:
                knowledge = self._technician_knowledge_value(tech)
                tokens.extend([tech_prefix, "KNOWLEDGE", _bucket_ratio(knowledge)])

        target_length = self.config.token_observation_length
        if len(tokens) < target_length:
            tokens.extend([self.config.token_pad_value] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]
        return {"tokens": tuple(tokens)}

    def _structured_obs(self) -> dict[str, np.ndarray]:
        ticket = self.current_request
        busy = np.asarray(
            [
                1 if getattr(tech, "busy", False) else 0
                for tech in self.dispatcher.techs
            ],
            dtype=np.int8,
        )
        observation: dict[str, np.ndarray] = {
            "sim_time": np.asarray([self._sim_time()], dtype=np.float32),
            "has_open_ticket": np.asarray(
                [1 if ticket is not None else 0], dtype=np.int8
            ),
            "ticket_created_at": np.asarray(
                [self._request_created_at(ticket)], dtype=np.float32
            ),
            "ticket_machine_id": np.asarray(
                [self._machine_id_from_request(ticket)], dtype=np.float32
            ),
            "technician_busy": busy,
        }
        if self.config.include_fatigue_in_observation:
            observation["technician_fatigue"] = np.asarray(
                [
                    float(getattr(tech, "fatigue", 0.0))
                    for tech in self.dispatcher.techs
                ],
                dtype=np.float32,
            )
        if self.config.include_queue_size_in_observation:
            observation["pending_queue_size"] = np.asarray(
                [float(self._queue_size())], dtype=np.float32
            )
        return observation

    def _token_id_obs(self) -> dict[str, np.ndarray]:
        """Return token observations encoded as integer IDs."""
        raw = self._token_obs()
        tokens = raw["tokens"]
        if self._tokenizer is None:
            self._tokenizer = StateTokenizer(
                seq_length=self.config.tokenizer_seq_length
            )
        return {"token_ids": self._tokenizer.encode(list(tokens))}

    def _obs(self) -> dict[str, Any]:
        if self.config.observation_representation == "token_ids":
            return self._token_id_obs()
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
            "metrics": dict(self._last_step_metrics),
        }

    def _reward_component(self, name: str, raw: float) -> float:
        """Apply coefficient and enabled flag for a named reward component."""
        comp = getattr(self.config.reward, name, None)
        if comp is None or not comp.enabled:
            return 0.0
        return comp.coefficient * raw

    def _reward_for_assignment(self, request: Any, tech_id: int) -> float:
        created_at = float(getattr(request, "created_at", self._sim_time()))
        waiting = max(0.0, self._sim_time() - created_at)
        tech = self.dispatcher.techs[tech_id]

        # --- original components ---
        assignment_raw = float(self.config.assignment_reward)
        wait_time_raw = -float(self.config.ticket_wait_time_penalty) * waiting
        busy_raw = -1.0 if bool(getattr(tech, "busy", False)) else 0.0
        queue_raw = -float(self._queue_size())

        breakdown: dict[str, float] = {
            "assignment": self._reward_component("assignment", assignment_raw),
            "wait_time": self._reward_component("wait_time", wait_time_raw),
            "queue_size": self._reward_component("queue_size", queue_raw),
            "busy_technician": self._reward_component("busy_technician", busy_raw),
        }

        # --- fatigue_cost: penalise high-fatigue assignments ---
        if self.config.reward.fatigue_cost.enabled:
            fatigue = float(getattr(tech, "fatigue", 0.0))
            breakdown["fatigue_cost"] = self._reward_component("fatigue_cost", -fatigue)

        # --- knowledge_match: reward expertise-repair alignment ---
        if self.config.reward.knowledge_match.enabled:
            km = self._knowledge_match_raw(tech, request)
            breakdown["knowledge_match"] = self._reward_component("knowledge_match", km)

        # --- workload_balance: penalise fatigue disparity across fleet ---
        if self.config.reward.workload_balance.enabled:
            fatigues = np.array(
                [float(getattr(t, "fatigue", 0.0)) for t in self.dispatcher.techs],
                dtype=np.float32,
            )
            breakdown["workload_balance"] = self._reward_component(
                "workload_balance", -float(fatigues.std())
            )

        # --- estimated_repair_time: penalise slow expected repairs ---
        if self.config.reward.estimated_repair_time.enabled:
            ert = self._estimated_repair_time_raw(tech, request)
            breakdown["estimated_repair_time"] = self._reward_component(
                "estimated_repair_time", ert
            )

        # --- machine_criticality: prioritise high-value machines ---
        if self.config.reward.machine_criticality.enabled:
            mc = self._machine_criticality_raw(request)
            breakdown["machine_criticality"] = self._reward_component(
                "machine_criticality", mc
            )

        # --- fleet_availability: system-level health ---
        if self.config.reward.fleet_availability.enabled:
            breakdown["fleet_availability"] = self._reward_component(
                "fleet_availability", self._fleet_availability_raw()
            )

        # --- throughput_delta: production flow ---
        if self.config.reward.throughput_delta.enabled:
            breakdown["throughput_delta"] = self._reward_component(
                "throughput_delta", self._throughput_delta_raw()
            )

        # --- repair_backlog_age: overall queue health ---
        if self.config.reward.repair_backlog_age.enabled:
            breakdown["repair_backlog_age"] = self._reward_component(
                "repair_backlog_age", self._repair_backlog_age_raw()
            )

        # --- technician_utilization: workforce efficiency ---
        if self.config.reward.technician_utilization.enabled:
            breakdown["technician_utilization"] = self._reward_component(
                "technician_utilization", self._technician_utilization_raw()
            )

        # --- downtime_cost: accumulated repair debt ---
        if self.config.reward.downtime_cost.enabled:
            breakdown["downtime_cost"] = self._reward_component(
                "downtime_cost", self._downtime_cost_raw()
            )

        self._last_reward_breakdown = breakdown
        return float(sum(breakdown.values()))

    # ------------------------------------------------------------------
    # Raw reward helpers for new components
    # ------------------------------------------------------------------

    def _knowledge_match_raw(self, tech: Any, request: Any) -> float:
        """Return a [0, 1] score for how well the tech's knowledge fits.

        The knowledge multiplier is in (0, 1]: 1 means no knowledge
        (base repair time), values near 0 mean high expertise (fast repair).
        We invert so that high knowledge -> high reward.
        """
        get_km = getattr(tech, "get_knowledge_multiplier", None)
        if get_km is None or not callable(get_km):
            return 0.0
        # multiplier in (0, 1]: lower = more knowledgeable
        multiplier = float(get_km(request))
        return max(0.0, min(1.0, 1.0 - multiplier))

    def _estimated_repair_time_raw(self, tech: Any, request: Any) -> float:
        """Return negative log-ratio of estimated vs base repair time.

        Using log makes the penalty sub-linear: doubling the repair time
        always adds the same penalty regardless of absolute magnitude,
        which produces more stable gradients for the agent.
        """
        base = (
            float(request.get_repair_time())
            if hasattr(request, "get_repair_time")
            else 10.0
        )
        compute = getattr(tech, "compute_repair_time", None)
        if compute is not None and callable(compute):
            estimated = float(compute(int(base), request))
        else:
            estimated = base
        # Ratio < 1 when tech is knowledgeable (faster), > 1 when fatigued.
        # Log < 0 means faster than base (good), > 0 means slower (bad).
        ratio = max(estimated, 1.0) / max(base, 1.0)
        return -math.log(max(ratio, 1e-6))

    def _machine_criticality_raw(self, request: Any) -> float:
        """Return a [0, 1] criticality score for the broken machine.

        Combines two signals:
        - Productivity share: fraction of total factory output this
          machine is responsible for (losing a high-throughput machine
          is costlier).
        - Buffer backlog: items waiting in the input buffer that are
          blocked while the machine is down.
        """
        machine = getattr(request, "machine", None)
        if machine is None:
            return 0.0

        # Productivity share across all factory machines
        all_machines = self._factory_machines()
        total_processed = sum(
            int(getattr(m, "total_processed", 0)) for m in all_machines
        )
        machine_processed = int(getattr(machine, "total_processed", 0))
        prod_share = machine_processed / max(total_processed, 1)

        # Normalised input buffer backlog
        backlog = self._machine_buffer_size(machine, "input_buffer")
        backlog_norm = min(backlog / 5.0, 1.0)  # saturates at 5 items

        # Weighted combination (both in [0, 1])
        return 0.6 * prod_share + 0.4 * backlog_norm

    def _fleet_availability_raw(self) -> float:
        """Fraction of machines currently operational — in [0, 1].

        Captures system-level health: a cascade of breakdowns drives
        this towards 0, while fast repairs keep it near 1.
        """
        machines = self._factory_machines()
        if not machines:
            return 1.0
        operational = sum(1 for m in machines if not bool(getattr(m, "broken", False)))
        return operational / len(machines)

    def _throughput_delta_raw(self) -> float:
        """Change in finished products since last assignment step — [0, 1].

        Rewards policies that keep the production line flowing.  Clipped
        to [0, 1] for scale stability (a single step rarely produces more
        than a handful of products).
        """
        sinks = getattr(self.dispatcher, "sinks", [])
        current = sum(int(getattr(s, "completed", 0)) for s in sinks)
        delta = current - self._prev_finished_products
        self._prev_finished_products = current
        return min(max(delta, 0), 1)

    def _repair_backlog_age_raw(self) -> float:
        """Negative mean age of ALL pending repair requests — (-1, 0].

        Unlike ``wait_time`` (which only considers the current ticket),
        this captures the health of the entire queue.  Saturated via
        ``-tanh(mean_age / scale)`` to prevent explosion.
        """
        queue = self._queue()
        items = getattr(queue, "items", queue) if queue is not None else []
        if not items:
            return 0.0
        now = self._sim_time()
        ages = [max(0.0, now - float(getattr(r, "created_at", now))) for r in items]
        mean_age = sum(ages) / len(ages)
        # Saturate: scale=200 means age~200 ≈ -0.76, age~500 ≈ -0.96
        return -math.tanh(mean_age / 200.0)

    def _technician_utilization_raw(self) -> float:
        """Reward for productive technician utilization — [-1, 1].

        Peaks at an optimal utilization ratio (0.6–0.7, matching
        real-world "wrench time" benchmarks) and penalizes both
        under-utilization (idle workforce) and over-utilization
        (burnout risk).  Uses a Gaussian centred at 0.65.
        """
        techs = self.dispatcher.techs
        if not techs:
            return 0.0
        busy_count = sum(1 for t in techs if bool(getattr(t, "busy", False)))
        utilization = busy_count / len(techs)
        # Gaussian reward centred at optimal utilization (0.65)
        optimal = 0.65
        sigma = 0.3
        return (
            math.exp(-((utilization - optimal) ** 2) / (2 * sigma * sigma)) * 2.0 - 1.0
        )

    def _downtime_cost_raw(self) -> float:
        """Negative fraction of machine-time lost to breakdowns — [-1, 0].

        Tracks accumulated downtime across all machines since episode
        start, normalised by total available machine-time.  Captures
        the systemic cost of slow repairs.
        """
        machines = self._factory_machines()
        now = self._sim_time()

        # Update downtime tracking for currently broken machines
        for m in machines:
            mid = getattr(m, "machine_id", id(m))
            if bool(getattr(m, "broken", False)):
                if mid not in self._machine_down_since:
                    self._machine_down_since[mid] = now
            elif mid in self._machine_down_since:
                self._total_downtime += now - self._machine_down_since.pop(mid)

        # Include still-broken machines in the total
        active_downtime = sum(now - t0 for t0 in self._machine_down_since.values())
        total_down = self._total_downtime + active_downtime

        # Normalise by total available machine-time
        total_available = max(now * len(machines), 1.0)
        return -min(total_down / total_available, 1.0)

    def _run_warmup(self) -> None:
        """Run MCA warmup to fit encoder and build tokenizer vocabulary."""
        if self._warmup_done or not self.config.use_mca_encoder:
            return
        if self._scenario_factory is None:
            return

        from kata.env_warmup import WarmupEnv

        # Force token observation mode for warmup so vocabulary is populated
        warmup = WarmupEnv(
            scenario_factory=self._scenario_factory,
            config=self.config.model_copy(
                update={
                    "observation_representation": "tokens",
                    "use_mca_encoder": False,
                }
            ),
            n_warmup_steps=self.config.warmup_steps,
            mca_grid_shape=self.config.mca_grid_shape,
            mca_n_components=self.config.mca_n_components,
            tokenizer_seq_length=self.config.tokenizer_seq_length,
        )
        self._mca_encoder, self._tokenizer = warmup.run()

        # Install MCA encoder as the global encoder if fitted
        if self._mca_encoder is not None and self._mca_encoder.fitted:
            from kata.entities.encoder import base as encoder_base

            encoder_base.ENCODER = self._mca_encoder

        self._warmup_done = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        _ = options

        if self._scenario_factory is not None:
            self._bootstrap_scenario()

        # Run MCA warmup on first reset
        self._run_warmup()

        self.episode_step = 0
        self.current_request = None
        self._last_reward_breakdown = {}
        self._last_step_metrics = {}
        self._breakdown_counter = 0
        self._repair_counter = 0
        self._prev_finished_products = 0
        self._machine_down_since = {}
        self._total_downtime = 0.0
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
        tech = self.dispatcher.techs[action]
        tech_id = tech.id
        self.dispatcher.start_repair(tech_id, request)
        self.episode_step += 1
        self._repair_counter += 1

        # Step metrics
        self._last_step_metrics = {
            m.name: m.compute(tech, request, self) for m in STEP_METRICS
        }

        reward = self._reward_for_assignment(request, action)
        self._advance_until_next_ticket()
        terminated = self._is_done()

        # On episode end, append episode-level metrics
        if terminated:
            self._last_step_metrics.update(
                {m.name: m.compute(self) for m in EPISODE_METRICS}
            )

        return self._obs(), reward, terminated, False, self._info()

    def render(self):
        from kata.render_utils import render_cli, render_dict, render_visual

        mode = self.render_mode
        if mode in ("human", "cli"):
            render_cli(self)
            return None
        if mode == "dict":
            return render_dict(self)
        if mode == "visual":
            return render_visual(self)
        return None
