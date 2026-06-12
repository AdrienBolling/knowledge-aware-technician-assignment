from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import simpy
import simpy.exceptions

from kata import get_config
from kata.core.config import GymEnvConfig
from kata.core.reward_normalizer import RewardNormalizer
from kata.core.tokenizer import StateTokenizer
from kata.metrics import EPISODE_METRICS, STEP_METRICS

# ---------------------------------------------------------------------------
# Value bucketing helpers – keep vocabulary bounded for Transformer input
# ---------------------------------------------------------------------------


_TIME_BUCKETS: tuple[str, ...] = (
    "T_NONE",
    "T_0_50",
    "T_50_200",
    "T_200_500",
    "T_500_1K",
    "T_1K_5K",
    "T_5K_10K",
    "T_10K_50K",
    "T_50K_100K",
    "T_100K+",
)

_RATIO_BUCKETS: tuple[str, ...] = (
    "R_0",
    "R_0_10",
    "R_10_20",
    "R_20_30",
    "R_30_40",
    "R_40_50",
    "R_50_60",
    "R_60_70",
    "R_70_80",
    "R_80_90",
    "R_90_100",
)

_COUNT_BUCKETS: tuple[str, ...] = (
    "C_0",
    "C_1",
    "C_2_3",
    "C_4_5",
    "C_6_10",
    "C_11_20",
    "C_21_50",
    "C_51_100",
    "C_100+",
)


def _bucket_time(t: float) -> str:
    """Bucket a simulation time value into a categorical token.

    Extended from the original 7-bucket scheme to keep distinguishing
    information past 5K sim time — long-horizon runs (max_sim_time of
    50K–200K) used to collapse everything past 5K into a single token.
    """
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
    if t < 10_000:
        return "T_5K_10K"
    if t < 50_000:
        return "T_10K_50K"
    if t < 100_000:
        return "T_50K_100K"
    return "T_100K+"


def _bucket_ratio(v: float) -> str:
    """Bucket a [0, 1] ratio (fatigue, knowledge, match) into deciles.

    Replaces the original 6-bucket scheme that collapsed e.g. 0.61 and
    0.79 into the same ``R_MEDHIGH`` token — too coarse for fatigue
    and knowledge-match signals that drive most of the policy gradient.
    """
    if v <= 0.0:
        return "R_0"
    if v < 0.1:
        return "R_0_10"
    if v < 0.2:
        return "R_10_20"
    if v < 0.3:
        return "R_20_30"
    if v < 0.4:
        return "R_30_40"
    if v < 0.5:
        return "R_40_50"
    if v < 0.6:
        return "R_50_60"
    if v < 0.7:
        return "R_60_70"
    if v < 0.8:
        return "R_70_80"
    if v < 0.9:
        return "R_80_90"
    return "R_90_100"


def _bucket_count(n: int) -> str:
    """Bucket a small non-negative count into a categorical token.

    Extended past 20 so larger factories (>20 machines) stop saturating
    the ``FACTORY_MACHINES`` / ``FACTORY_PRODUCED`` tokens.
    """
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
    if n <= 50:
        return "C_21_50"
    if n <= 100:
        return "C_51_100"
    return "C_100+"


def _bool_token(v: bool) -> str:
    return "TRUE" if v else "FALSE"


# ---------------------------------------------------------------------------
# Emitter pattern — single source of truth for the (key, value) stream
# ---------------------------------------------------------------------------


class _Emitter:
    """Strategy interface for flattening the obs schema into a stream.

    The env's ``_build_token_stream`` drives an ``_Emitter`` instance
    through one ``emit`` call per piece of state.  Two concrete
    flatteners exist: ``_StringEmitter`` produces the legacy bucket
    tokens, ``_HybridEmitter`` produces ``(<NUM>, raw_value, kind)``
    triples consumed by :class:`HybridTokenEncoder`.

    Method semantics:

    * ``bare(s)`` — append a raw categorical token with no key/value pair
      (used for the ``TECH_{i}`` prefix between repeated triples).
    * ``cat(key, value)`` — categorical key-value pair.
    * ``bool(key, value)`` — boolean key-value pair.
    * ``ratio(key, value)`` — scalar in [0, 1].
    * ``count(key, value)`` — non-negative integer / float count.
    * ``time(key, value)`` — recent event time / age (Time2Vec target).
    * ``hazard(key, value)`` — long-horizon time (Fourier target).
    """

    def bare(self, token: str) -> None:
        raise NotImplementedError

    def cat(self, key: str, value: str) -> None:
        raise NotImplementedError

    def bool(self, key: str, value: bool) -> None:
        raise NotImplementedError

    def ratio(self, key: str, value: float) -> None:
        raise NotImplementedError

    def count(self, key: str, value: int) -> None:
        raise NotImplementedError

    def time(self, key: str, value: float) -> None:
        raise NotImplementedError

    def hazard(self, key: str, value: float) -> None:
        raise NotImplementedError


class _StringEmitter(_Emitter):
    """Legacy bucket-string flattener.  Output: ``self.tokens: list[str]``."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def bare(self, token: str) -> None:
        self.tokens.append(token)

    def cat(self, key: str, value: str) -> None:
        self.tokens.append(key)
        self.tokens.append(value)

    def bool(self, key: str, value: bool) -> None:
        self.tokens.append(key)
        self.tokens.append(_bool_token(value))

    def ratio(self, key: str, value: float) -> None:
        self.tokens.append(key)
        self.tokens.append(_bucket_ratio(float(value)))

    def count(self, key: str, value: int) -> None:
        self.tokens.append(key)
        self.tokens.append(_bucket_count(int(value)))

    def time(self, key: str, value: float) -> None:
        self.tokens.append(key)
        self.tokens.append(_bucket_time(float(value)))

    def hazard(self, key: str, value: float) -> None:
        # Hazard times use the same bucket vocabulary as recent times
        # in legacy mode — only hybrid mode routes them differently.
        self.tokens.append(key)
        self.tokens.append(_bucket_time(float(value)))


class _HybridEmitter(_Emitter):
    """Hybrid flattener producing aligned categorical + continuous channels."""

    def __init__(self) -> None:
        from agents.networks.continuous_features import ContKind

        self._CAT = ContKind.CATEGORICAL
        self._RATIO = ContKind.RATIO_PLE
        self._COUNT = ContKind.COUNT_PLE
        self._TIME = ContKind.TIME2VEC
        self._HAZARD = ContKind.FOURIER
        self.tokens: list[str] = []
        self.cont_values: list[float] = []
        self.cont_kinds: list[int] = []

    def _emit(self, token: str, value: float, kind: int) -> None:
        self.tokens.append(token)
        self.cont_values.append(float(value))
        self.cont_kinds.append(int(kind))

    def bare(self, token: str) -> None:
        self._emit(token, 0.0, self._CAT)

    def cat(self, key: str, value: str) -> None:
        self._emit(key, 0.0, self._CAT)
        self._emit(value, 0.0, self._CAT)

    def bool(self, key: str, value: bool) -> None:
        self._emit(key, 0.0, self._CAT)
        self._emit(_bool_token(value), 0.0, self._CAT)

    def ratio(self, key: str, value: float) -> None:
        self._emit(key, 0.0, self._CAT)
        self._emit("<NUM>", float(value), self._RATIO)

    def count(self, key: str, value: int) -> None:
        self._emit(key, 0.0, self._CAT)
        self._emit("<NUM>", float(value), self._COUNT)

    def time(self, key: str, value: float) -> None:
        self._emit(key, 0.0, self._CAT)
        # Recent-time emissions use a negative sentinel ``-1`` for "no
        # such event yet"; map those to ``0`` so PLE/Time2Vec see a
        # well-defined input.  The categorical key alone disambiguates
        # "missing" from "very recent" via the surrounding structure.
        v = max(0.0, float(value))
        self._emit("<NUM>", v, self._TIME)

    def hazard(self, key: str, value: float) -> None:
        self._emit(key, 0.0, self._CAT)
        v = max(0.0, float(value))
        self._emit("<NUM>", v, self._HAZARD)


class _SetEmitter(_Emitter):
    """Per-slot hybrid emitter for the ``set`` observation mode.

    Writes (token, cont_value, cont_kind) triples into the *currently
    open slot* — a Python list owned by the caller, swapped in via
    :meth:`open_slot`.  This lets the env's stream-construction code
    fan out into many independent slot streams (one per technician,
    one per machine, plus an env stream) using the same emit-API as
    :class:`_HybridEmitter`.

    The slot's owner is responsible for fixed-length padding /
    truncation after emission completes.
    """

    def __init__(self) -> None:
        from agents.networks.continuous_features import ContKind

        self._CAT = ContKind.CATEGORICAL
        self._RATIO = ContKind.RATIO_PLE
        self._COUNT = ContKind.COUNT_PLE
        self._TIME = ContKind.TIME2VEC
        self._HAZARD = ContKind.FOURIER
        # tuple of three parallel lists: (tokens, cont_values, cont_kinds)
        self._slot: tuple[list[str], list[float], list[int]] | None = None

    def open_slot(
        self, slot: tuple[list[str], list[float], list[int]]
    ) -> None:
        self._slot = slot

    def close_slot(self) -> None:
        self._slot = None

    def _emit(self, token: str, value: float, kind: int) -> None:
        if self._slot is None:
            msg = "_SetEmitter called without an open slot"
            raise RuntimeError(msg)
        toks, vals, kinds = self._slot
        toks.append(token)
        vals.append(float(value))
        kinds.append(int(kind))

    def bare(self, token: str) -> None:
        self._emit(token, 0.0, self._CAT)

    def cat(self, key: str, value: str) -> None:
        # In the set mode each "slot position" is a single triple, so
        # we collapse (key, value) into a single categorical token of
        # the form "KEY=VALUE".  This avoids inflating the slot length
        # for pairs that always co-occur.
        self._emit(f"{key}={value}", 0.0, self._CAT)

    def bool(self, key: str, value: bool) -> None:
        self._emit(f"{key}={_bool_token(value)}", 0.0, self._CAT)

    def ratio(self, key: str, value: float) -> None:
        # The categorical key marks the *semantic role* of the
        # following NUM slot; the NUM carries the raw scalar that
        # PLE/Time2Vec/Fourier will route on.  Here we collapse to a
        # single position by hashing the key into the token id (each
        # role gets its own learnable embedding) and storing the value.
        self._emit(f"<RATIO:{key}>", float(value), self._RATIO)

    def count(self, key: str, value: int) -> None:
        self._emit(f"<COUNT:{key}>", float(value), self._COUNT)

    def time(self, key: str, value: float) -> None:
        v = max(0.0, float(value))
        self._emit(f"<TIME:{key}>", v, self._TIME)

    def hazard(self, key: str, value: float) -> None:
        v = max(0.0, float(value))
        self._emit(f"<FOUR:{key}>", v, self._HAZARD)


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

        # Per-component reward normaliser — persists across episodes
        # within this env instance.  Active only when
        # ``GymRewardConfig.normalize_components`` is set; otherwise
        # ``normalize()`` calls are skipped.
        self._reward_normalizer: RewardNormalizer = RewardNormalizer(
            epsilon=float(
                getattr(
                    self.config.reward, "normalize_components_eps", 1e-4
                )
            )
        )
        # Fleet knowledge volume sampled at the previous decision step,
        # used to compute the ``knowledge_increment`` per-step reward.
        self._prev_fleet_knowledge: float = 0.0

        # Tokenizer & MCA state (populated by warmup or passed in)
        self._tokenizer: StateTokenizer | None = tokenizer
        self._mca_encoder: Any | None = None
        self._warmup_done = False

        # Metrics state (reset each episode)
        self._last_step_metrics: dict[str, float] = {}
        self._breakdown_counter: int = 0
        # ``_repair_counter`` historically meant "tickets assigned" and is
        # still incremented at assignment time; ``_completed_repair_counter``
        # is incremented when the dispatcher finishes a repair so we can
        # distinguish in-flight from completed work.
        self._repair_counter: int = 0
        self._completed_repair_counter: int = 0
        # Sum of estimated repair times across all assignments — used by
        # the MTTR metric so it reflects mean repair *duration*, not the
        # mean time between repairs.
        self._total_repair_time: float = 0.0

        # State tracking for delta-based rewards.  These are updated on
        # every step (regardless of which reward components are enabled)
        # so that episode-level metrics like fleet availability stay
        # accurate even when the corresponding reward is off.
        self._prev_finished_products: int = 0
        self._machine_down_since: dict[int, float] = {}  # machine_id -> time it broke
        self._total_downtime: float = 0.0  # accumulated machine-time lost

        # Per-machine episode stats — populated by
        # ``_update_machine_state_tracking`` and surfaced through
        # ``info["machine_stats"]`` so the runner can render one
        # histogram per machine at episode end.
        self._machine_total_downtime: dict[int, float] = {}
        self._machine_total_processing_time: dict[int, float] = {}
        self._machine_processing_since: dict[int, float] = {}
        self._machine_breakdown_counts: dict[int, int] = {}
        self._machine_labels: dict[int, str] = {}

        # Per-technician assignment counts within the current episode.
        # Drives the ``selection_diversity`` reward and is reset on
        # ``reset()``.  Indexed by the *action* (= position in
        # ``self.dispatcher.techs``), so it stays valid even if
        # technicians are re-instantiated by the scenario builder.
        self._tech_assignment_counts: list[int] = []
        # Simulation time at which each technician was last assigned a
        # repair (``-1.0`` ≡ "never assigned this episode").  Used by
        # the ``tech_aware`` observation mode to surface a per-tech
        # idle-time signal to the policy.
        self._tech_last_assignment_time: list[float] = []

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

        # Technician names must be unique across the fleet — they are
        # used as labels in fleet-wide plots and metrics, and silent
        # collisions would merge two technicians' series in W&B.  Fail
        # fast here with a clear remediation hint.
        self._check_technician_name_uniqueness(self.dispatcher.techs)

        # Listen for repair-completion events so MTTR / completed-repair
        # metrics stay accurate even when the corresponding rewards are
        # disabled.
        if hasattr(self.dispatcher, "on_repair_completed"):
            self.dispatcher.on_repair_completed = self._on_repair_completed

        n_techs = len(self.dispatcher.techs)
        # Re-shape the per-tech counters to match the new fleet (they
        # are fully reset at every ``reset()`` regardless).
        self._tech_assignment_counts = [0] * n_techs
        self._tech_last_assignment_time = [-1.0] * n_techs
        self.action_space = gym.spaces.Discrete(n_techs)
        if self.config.observation_representation == "set":
            max_t = int(self.config.max_techs)
            max_m = int(self.config.max_machines)
            L_t = int(self.config.set_tech_slot_length)
            L_m = int(self.config.set_machine_slot_length)
            L_e = int(self.config.set_env_length)
            space_dict: dict[str, gym.Space] = {
                "tech_token_ids": gym.spaces.Box(
                    low=0, high=np.iinfo(np.int64).max,
                    shape=(max_t, L_t), dtype=np.int64,
                ),
                "tech_cont_values": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_t, L_t), dtype=np.float32,
                ),
                "tech_cont_kinds": gym.spaces.Box(
                    low=0, high=127, shape=(max_t, L_t), dtype=np.int8,
                ),
                "tech_mask": gym.spaces.MultiBinary(max_t),
                "machine_token_ids": gym.spaces.Box(
                    low=0, high=np.iinfo(np.int64).max,
                    shape=(max_m, L_m), dtype=np.int64,
                ),
                "machine_cont_values": gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(max_m, L_m), dtype=np.float32,
                ),
                "machine_cont_kinds": gym.spaces.Box(
                    low=0, high=127, shape=(max_m, L_m), dtype=np.int8,
                ),
                "machine_mask": gym.spaces.MultiBinary(max_m),
                "env_token_ids": gym.spaces.Box(
                    low=0, high=np.iinfo(np.int64).max,
                    shape=(L_e,), dtype=np.int64,
                ),
                "env_cont_values": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(L_e,), dtype=np.float32,
                ),
                "env_cont_kinds": gym.spaces.Box(
                    low=0, high=127, shape=(L_e,), dtype=np.int8,
                ),
            }
            if self.config.expose_action_mask:
                space_dict["action_mask"] = gym.spaces.MultiBinary(max_t)
            self.observation_space = gym.spaces.Dict(space_dict)
            return
        if self.config.observation_representation == "hybrid":
            seq_len = self.config.tokenizer_seq_length
            space_dict: dict[str, gym.Space] = {
                "token_ids": gym.spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(seq_len,),
                    dtype=np.int64,
                ),
                "cont_values": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(seq_len,),
                    dtype=np.float32,
                ),
                "cont_kinds": gym.spaces.Box(
                    low=0,
                    high=127,
                    shape=(seq_len,),
                    dtype=np.int8,
                ),
            }
            if self.config.expose_action_mask:
                space_dict["action_mask"] = gym.spaces.MultiBinary(n_techs)
            self.observation_space = gym.spaces.Dict(space_dict)
            return
        if self.config.observation_representation == "token_ids":
            seq_len = self.config.tokenizer_seq_length
            space_dict = {
                "token_ids": gym.spaces.Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(seq_len,),
                    dtype=np.int64,
                ),
            }
            if self.config.expose_action_mask:
                space_dict["action_mask"] = gym.spaces.MultiBinary(n_techs)
            self.observation_space = gym.spaces.Dict(space_dict)
            return
        if self.config.observation_representation == "tokens":
            token_spaces = tuple(
                gym.spaces.Text(max_length=self.config.token_max_length)
                for _ in range(self.config.token_observation_length)
            )
            space_dict = {"tokens": gym.spaces.Tuple(token_spaces)}
            if self.config.expose_action_mask:
                space_dict["action_mask"] = gym.spaces.MultiBinary(n_techs)
            self.observation_space = gym.spaces.Dict(space_dict)
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
        if self.config.expose_action_mask:
            observation_space["action_mask"] = gym.spaces.MultiBinary(n_techs)
        self.observation_space = gym.spaces.Dict(observation_space)

    @staticmethod
    def _check_technician_name_uniqueness(techs: list[Any]) -> None:
        """Raise ``ValueError`` if any two technicians share a name.

        Many fleet-level plots and metrics (``tech_knowledge``,
        ``tech_fatigue`` …) key on ``tech.name``.  If two technicians
        share a name their series would silently overwrite each other,
        so the env refuses to start with duplicates.  Fix the offending
        config by overriding ``name`` per technician (e.g. by passing
        a unique ``name`` field next to ``"template": "junior"``).
        """
        from collections import Counter

        names = [str(getattr(t, "name", "")) for t in techs]
        duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
        if duplicates:
            msg = (
                "Technician names must be unique across the fleet "
                f"(duplicates: {duplicates}). "
                "Override the 'name' field per technician in the env "
                "config (e.g. add \"name\": \"junior_1\" alongside the "
                "\"template\": \"junior\" entry)."
            )
            raise ValueError(msg)

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
            except (simpy.exceptions.Interrupt, simpy.exceptions.SimPyException):
                # Expected SimPy control flow: a process was preempted
                # (e.g. by a disruption) or the scheduler raised an
                # internal/exhaustion condition.  Either way, this
                # advance returns cleanly; the episode-end check
                # downstream decides whether the episode is over.
                # Note: we deliberately do *not* catch arbitrary
                # ``Exception`` --- genuine bugs in machine processes,
                # technicians, or breakdown models should surface
                # instead of being silently swallowed.
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

    def _build_token_stream(self, emit: "_Emitter") -> None:
        """Drive an :class:`_Emitter` through the obs schema.

        This is the single source of truth for the order and content of
        the (key, value) stream that becomes either bucket-string tokens
        (legacy ``tokens`` / ``token_ids`` modes) or
        ``(token_id, cont_value, cont_kind)`` triples (new ``hybrid`` mode).
        See ``_StringEmitter`` and ``_HybridEmitter`` for the two
        flatteners.
        """
        ticket = self.current_request
        machine = getattr(ticket, "machine", None)
        machine_type = str(getattr(machine, "mtype", "NONE"))

        # -- header / context --
        emit.cat("OBS_MODE", self.config.observation_mode)
        emit.hazard("SIM_TIME", self._sim_time())
        emit.bool("HAS_TICKET", ticket is not None)
        ticket_age = (
            self._sim_time() - self._request_created_at(ticket)
            if ticket is not None
            else -1.0
        )
        emit.time("TICKET_AGE", ticket_age)
        emit.cat("TICKET_MACHINE_TYPE", machine_type)

        if self.config.observation_mode in {"broken_machine", "factory_level", "tech_aware"}:
            emit.cat("MACHINE_TYPE", machine_type)
            emit.bool("MACHINE_BROKEN", bool(getattr(machine, "broken", False)))
            emit.bool("MACHINE_PROCESSING", bool(getattr(machine, "is_processing", False)))
            emit.count("MACHINE_TOTAL_PROCESSED", int(getattr(machine, "total_processed", 0)))
            emit.count(
                "MACHINE_INPUT_BUF",
                self._machine_buffer_size(machine, "input_buffer"),
            )
            emit.count(
                "MACHINE_OUTPUT_BUF",
                self._machine_buffer_size(machine, "output_buffer"),
            )

        if self.config.observation_mode in {"factory_level", "tech_aware"}:
            machines = self._factory_machines()
            broken_count = sum(1 for m in machines if bool(getattr(m, "broken", False)))
            processing_count = sum(
                1 for m in machines if bool(getattr(m, "is_processing", False))
            )
            total_processed = sum(
                int(getattr(m, "total_processed", 0)) for m in machines
            )
            emit.count("FACTORY_MACHINES", len(machines))
            emit.count("FACTORY_BROKEN", broken_count)
            emit.count("FACTORY_PROCESSING", processing_count)
            emit.count("FACTORY_PRODUCED", total_processed)
            emit.count("FACTORY_QUEUE", self._queue_size())

            if self.config.include_broken_by_type_tokens:
                broken_by_type: dict[str, int] = {}
                for m in machines:
                    if bool(getattr(m, "broken", False)):
                        mt = str(getattr(m, "mtype", "NONE"))
                        broken_by_type[mt] = broken_by_type.get(mt, 0) + 1
                for mt in sorted(broken_by_type):
                    emit.count(f"BROKEN_{mt}", broken_by_type[mt])

        # -- tech_aware: ticket-specific extras --
        tech_aware = self.config.observation_mode == "tech_aware"
        if tech_aware:
            comp_type = "NONE"
            if ticket is not None and hasattr(ticket, "get_failed_component_info"):
                info = ticket.get_failed_component_info()
                if info:
                    comp_type = str(info.get("component_type", "NONE"))
            emit.cat("TICKET_COMPONENT_TYPE", comp_type)

            queue = self._queue()
            items = getattr(queue, "items", queue) if queue is not None else []
            lookahead = int(self.config.next_ticket_lookahead)
            for slot in range(lookahead):
                prefix = f"NEXT{slot + 1}"
                if slot < len(items):
                    req = items[slot]
                    nmt = str(getattr(getattr(req, "machine", None), "mtype", "NONE"))
                    nct = "NONE"
                    if hasattr(req, "get_failed_component_info"):
                        info = req.get_failed_component_info()
                        if info:
                            nct = str(info.get("component_type", "NONE"))
                    nage = self._sim_time() - self._request_created_at(req)
                else:
                    nmt, nct, nage = "NONE", "NONE", -1.0
                emit.cat(f"{prefix}_MACHINE_TYPE", nmt)
                emit.cat(f"{prefix}_COMPONENT_TYPE", nct)
                emit.time(f"{prefix}_AGE", nage)

            if self.config.include_queue_composition_tokens:
                qc_by_type: dict[str, int] = {}
                for req in items:
                    ct = "NONE"
                    if hasattr(req, "get_failed_component_info"):
                        info = req.get_failed_component_info()
                        if info:
                            ct = str(info.get("component_type", "NONE"))
                    qc_by_type[ct] = qc_by_type.get(ct, 0) + 1
                for ct in sorted(qc_by_type):
                    emit.count(f"QC_{ct}", qc_by_type[ct])

        # -- per-technician tokens --
        for idx, tech in enumerate(self.dispatcher.techs):
            tech_prefix = f"TECH_{idx}"
            emit.bare(tech_prefix)
            emit.bool("BUSY", bool(getattr(tech, "busy", False)))
            if self.config.include_technician_fatigue_tokens:
                emit.bare(tech_prefix)
                emit.ratio("FATIGUE", float(getattr(tech, "fatigue", 0.0)))
            if self.config.include_technician_knowledge_tokens:
                emit.bare(tech_prefix)
                emit.ratio("KNOWLEDGE", self._technician_knowledge_value(tech))
            if self.config.include_technician_assignment_count_tokens:
                ac = (
                    int(self._tech_assignment_counts[idx])
                    if idx < len(self._tech_assignment_counts)
                    else 0
                )
                emit.bare(tech_prefix)
                emit.count("ASSIGN_COUNT", ac)

            if tech_aware:
                match_val = 0.0
                if ticket is not None and hasattr(tech, "get_knowledge_multiplier"):
                    try:
                        mult = float(tech.get_knowledge_multiplier(ticket))
                        match_val = max(0.0, min(1.0, 1.0 - mult))
                    except Exception:
                        match_val = 0.0
                emit.bare(tech_prefix)
                emit.ratio("MATCH", match_val)

                eta = -1.0
                if ticket is not None and hasattr(tech, "compute_repair_time"):
                    base = (
                        float(ticket.get_repair_time())
                        if hasattr(ticket, "get_repair_time")
                        else 10.0
                    )
                    try:
                        eta = float(tech.compute_repair_time(base, ticket))
                    except Exception:
                        eta = base
                emit.bare(tech_prefix)
                emit.time("ETA", eta)

                last = (
                    self._tech_last_assignment_time[idx]
                    if idx < len(self._tech_last_assignment_time)
                    else -1.0
                )
                last_age = -1.0 if last < 0 else max(0.0, self._sim_time() - last)
                emit.bare(tech_prefix)
                emit.time("LAST_AGE", last_age)

    def _token_obs(self) -> dict[str, tuple[str, ...]]:
        """Build a key-value token sequence (legacy bucket-string mode)."""
        emitter = _StringEmitter()
        self._build_token_stream(emitter)
        tokens = emitter.tokens
        target_length = self.config.token_observation_length
        if len(tokens) < target_length:
            tokens.extend([self.config.token_pad_value] * (target_length - len(tokens)))
        else:
            tokens = tokens[:target_length]
        return {"tokens": tuple(tokens)}

    def _hybrid_obs(self) -> dict[str, np.ndarray]:
        """Build a hybrid observation: categorical token-ids + parallel continuous channels.

        Returns three aligned ``(S,)`` arrays (after padding / truncation):

        * ``token_ids`` — categorical IDs.  Continuous-value positions
          carry the ``<NUM>`` placeholder id.
        * ``cont_values`` — raw scalar values at continuous positions, 0
          at categorical positions.
        * ``cont_kinds`` — :class:`ContKind` code per position (0 =
          categorical, 1 = ratio_PLE, 2 = count_PLE, 3 = time2vec,
          4 = fourier).

        Consumed by :class:`agents.networks.hybrid_encoder.HybridTokenEncoder`.
        """
        from kata.core.tokenizer import PAD_ID
        from agents.networks.continuous_features import ContKind

        emitter = _HybridEmitter()
        self._build_token_stream(emitter)

        tokens = emitter.tokens
        cont_values = emitter.cont_values
        cont_kinds = emitter.cont_kinds
        target_length = self.config.tokenizer_seq_length

        # Pad / truncate all three streams in lockstep so positions stay aligned.
        if len(tokens) < target_length:
            pad = target_length - len(tokens)
            tokens = tokens + [self.config.token_pad_value] * pad
            cont_values = cont_values + [0.0] * pad
            cont_kinds = cont_kinds + [ContKind.CATEGORICAL] * pad
        else:
            tokens = tokens[:target_length]
            cont_values = cont_values[:target_length]
            cont_kinds = cont_kinds[:target_length]

        if self._tokenizer is None:
            self._tokenizer = StateTokenizer(seq_length=target_length)
        token_ids = np.array(
            [self._tokenizer.token_to_id(t) for t in tokens],
            dtype=np.int64,
        )
        # Pad positions get PAD_ID through token_to_id but defensively
        # enforce the contract — and zero out continuous channels at pads.
        pad_positions = token_ids == PAD_ID
        cont_values_arr = np.asarray(cont_values, dtype=np.float32)
        cont_kinds_arr = np.asarray(cont_kinds, dtype=np.int8)
        cont_values_arr[pad_positions] = 0.0
        cont_kinds_arr[pad_positions] = ContKind.CATEGORICAL
        return {
            "token_ids": token_ids,
            "cont_values": cont_values_arr,
            "cont_kinds": cont_kinds_arr,
        }

    def _set_obs(self) -> dict[str, np.ndarray]:
        """Return the three-stream ``set`` observation.

        Output keys
        -----------
        ``tech_token_ids``     ``(max_techs, L_TECH)``        int64
        ``tech_cont_values``   ``(max_techs, L_TECH)``        float32
        ``tech_cont_kinds``    ``(max_techs, L_TECH)``        int8
        ``tech_mask``          ``(max_techs,)``               int8 (1 == valid slot)
        ``machine_token_ids``  ``(max_machines, L_MACH)``     int64
        ``machine_cont_values`` ``(max_machines, L_MACH)``    float32
        ``machine_cont_kinds`` ``(max_machines, L_MACH)``     int8
        ``machine_mask``       ``(max_machines,)``            int8
        ``env_token_ids``      ``(L_ENV,)``                   int64
        ``env_cont_values``    ``(L_ENV,)``                   float32
        ``env_cont_kinds``     ``(L_ENV,)``                   int8

        Each (tech / machine / env) slot is a fixed-length sub-sequence
        of hybrid triples emitted by :class:`_SetEmitter`.  Real fleets
        smaller than ``max_techs`` / ``max_machines`` are zero-padded
        with the mask flagging which slots are real.  Real fleets that
        exceed the caps are truncated and a ``RuntimeWarning`` is
        raised — pick caps larger than any expected scenario.
        """
        from agents.networks.continuous_features import ContKind
        from kata.core.tokenizer import PAD_ID

        max_t = int(self.config.max_techs)
        max_m = int(self.config.max_machines)
        L_t = int(self.config.set_tech_slot_length)
        L_m = int(self.config.set_machine_slot_length)
        L_e = int(self.config.set_env_length)

        emitter = _SetEmitter()
        ticket = self.current_request
        ticket_machine = getattr(ticket, "machine", None)
        ticket_machine_id = (
            self._machine_id_from_machine(ticket_machine)
            if ticket_machine is not None
            else None
        )
        # Component type of the currently-broken machine — repeated on
        # the machine slot so the cross-attention can match tech
        # expertise to broken-machine context locally.
        cur_comp_type = "NONE"
        if ticket is not None and hasattr(ticket, "get_failed_component_info"):
            info = ticket.get_failed_component_info()
            if info:
                cur_comp_type = str(info.get("component_type", "NONE"))

        # Peek at the next two queued requests for tech-side
        # MATCH lookahead and env-side NEXT_K tokens.
        queue = self._queue()
        q_items = getattr(queue, "items", queue) if queue is not None else []
        next_requests: list[Any] = list(q_items)[:2] if q_items else []

        # ----- TECHNICIAN STREAM ------------------------------------------
        tech_tokens: list[list[str]] = []
        tech_vals: list[list[float]] = []
        tech_kinds: list[list[int]] = []
        techs = self.dispatcher.techs if self.dispatcher else []
        if len(techs) > max_t:
            import warnings as _w
            _w.warn(
                f"set obs: {len(techs)} techs exceeds max_techs={max_t}; "
                "extra slots truncated.",
                RuntimeWarning,
            )
            techs = list(techs)[:max_t]

        for idx, tech in enumerate(techs):
            slot = ([], [], [])
            emitter.open_slot(slot)
            # --- profile / template ----------------------------------
            emitter.cat("TEMPLATE", self._technician_template(tech))
            # --- state flags -----------------------------------------
            emitter.bool("BUSY", bool(getattr(tech, "busy", False)))
            emitter.bool("DISRUPT", bool(getattr(tech, "_in_disruption", False)))
            emitter.ratio("FATIGUE", float(getattr(tech, "fatigue", 0.0)))
            ac = (
                int(self._tech_assignment_counts[idx])
                if idx < len(self._tech_assignment_counts)
                else 0
            )
            emitter.count("ASSIGNS", ac)
            # --- knowledge-space features (six scalars) --------------
            kf = self._tech_knowledge_features(tech)
            emitter.count("KNOW_VOL", kf["volume"])
            emitter.count("KNOW_MAX", kf["max_k"])
            emitter.ratio("KNOW_SPEC", kf["spec_idx"])
            emitter.count("KNOW_ENT", kf["entropy"])
            # --- per-ticket expertise (current + 2 queued) ----------
            emitter.ratio("MATCH", self._tech_match(tech, ticket))
            emitter.ratio(
                "MATCH_N1",
                self._tech_match(
                    tech, next_requests[0] if len(next_requests) > 0 else None
                ),
            )
            emitter.ratio(
                "MATCH_N2",
                self._tech_match(
                    tech, next_requests[1] if len(next_requests) > 1 else None
                ),
            )
            # --- timing ----------------------------------------------
            eta = -1.0
            if ticket is not None and hasattr(tech, "compute_repair_time"):
                base = (
                    float(ticket.get_repair_time())
                    if hasattr(ticket, "get_repair_time")
                    else 10.0
                )
                try:
                    eta = float(tech.compute_repair_time(base, ticket))
                except Exception:
                    eta = base
            emitter.time("ETA", eta)
            last = (
                self._tech_last_assignment_time[idx]
                if idx < len(self._tech_last_assignment_time)
                else -1.0
            )
            last_age = -1.0 if last < 0 else max(0.0, self._sim_time() - last)
            emitter.time("LAST_AGE", last_age)
            emitter.close_slot()
            tech_tokens.append(slot[0])
            tech_vals.append(slot[1])
            tech_kinds.append(slot[2])

        # ----- MACHINE STREAM ---------------------------------------------
        machine_tokens: list[list[str]] = []
        machine_vals: list[list[float]] = []
        machine_kinds: list[list[int]] = []
        machines = self._factory_machines()
        if len(machines) > max_m:
            import warnings as _w
            _w.warn(
                f"set obs: {len(machines)} machines exceeds max_machines="
                f"{max_m}; extra slots truncated.",
                RuntimeWarning,
            )
            machines = machines[:max_m]
        sim_time_now = self._sim_time()
        for m in machines:
            slot = ([], [], [])
            emitter.open_slot(slot)
            mt = str(getattr(m, "mtype", "NONE"))
            emitter.cat("M_TYPE", mt)
            emitter.bool("BROKEN", bool(getattr(m, "broken", False)))
            emitter.bool("PROC", bool(getattr(m, "is_processing", False)))
            mid = self._machine_id_from_machine(m)
            # IS_CURRENT_TICKET: explicit signal that this is the
            # broken machine the agent is being asked about NOW.
            is_current = (
                ticket_machine_id is not None and mid == ticket_machine_id
            )
            emitter.bool("IS_CURRENT", is_current)
            # CUR_COMP: the failed component type of the current
            # ticket, repeated on the broken machine slot so the
            # cross-attention can match local-to-local.  For non-
            # current machines we emit "NONE" — the model learns to
            # gate on IS_CURRENT.
            emitter.cat(
                "CUR_COMP", cur_comp_type if is_current else "NONE"
            )
            emitter.count("PROC_TOT", int(getattr(m, "total_processed", 0)))
            emitter.count("IN_BUF", self._machine_buffer_size(m, "input_buffer"))
            emitter.count("OUT_BUF", self._machine_buffer_size(m, "output_buffer"))
            bd_count = int(self._machine_breakdown_counts.get(mid, 0))
            emitter.count("BD_COUNT", bd_count)
            emitter.time(
                "DOWNTIME", float(self._machine_total_downtime.get(mid, 0.0))
            )
            # MEAN_TBF: rough mean-time-between-failures proxy.  Long-
            # horizon recent time signal — route through Time2Vec.  A
            # machine that has never broken returns sim_time_now (so
            # "very long" → "highly reliable").
            mean_tbf = (
                sim_time_now / max(1.0, float(bd_count))
                if sim_time_now > 0
                else 0.0
            )
            emitter.time("MEAN_TBF", mean_tbf)
            emitter.close_slot()
            machine_tokens.append(slot[0])
            machine_vals.append(slot[1])
            machine_kinds.append(slot[2])

        # ----- ENV STREAM -------------------------------------------------
        env_slot = ([], [], [])
        emitter.open_slot(env_slot)
        emitter.bool("HAS_T", ticket is not None)
        emitter.cat(
            "T_M_TYPE", str(getattr(ticket_machine, "mtype", "NONE"))
        )
        emitter.cat("T_C_TYPE", cur_comp_type)
        emitter.hazard("SIM_T", self._sim_time())
        t_age = (
            self._sim_time() - self._request_created_at(ticket)
            if ticket is not None
            else -1.0
        )
        emitter.time("T_AGE", t_age)
        emitter.count("Q_SIZE", self._queue_size())
        all_machines = self._factory_machines()
        broken_n = sum(1 for m in all_machines if bool(getattr(m, "broken", False)))
        proc_n = sum(
            1 for m in all_machines if bool(getattr(m, "is_processing", False))
        )
        emitter.count("BROKEN_N", broken_n)
        emitter.count("PROC_N", proc_n)
        # --- lookahead: next two queued tickets ------------------------
        for slot_idx in range(2):
            prefix = f"N{slot_idx + 1}"
            if slot_idx < len(next_requests):
                nreq = next_requests[slot_idx]
                nm = str(getattr(getattr(nreq, "machine", None), "mtype", "NONE"))
                nc = "NONE"
                if hasattr(nreq, "get_failed_component_info"):
                    info = nreq.get_failed_component_info()
                    if info:
                        nc = str(info.get("component_type", "NONE"))
                nage = self._sim_time() - self._request_created_at(nreq)
            else:
                nm, nc, nage = "NONE", "NONE", -1.0
            emitter.cat(f"{prefix}_M_TYPE", nm)
            emitter.cat(f"{prefix}_C_TYPE", nc)
            emitter.time(f"{prefix}_AGE", nage)
        emitter.close_slot()

        # ----- TOKENIZE + PAD ---------------------------------------------
        if self._tokenizer is None:
            self._tokenizer = StateTokenizer(
                seq_length=self.config.tokenizer_seq_length
            )

        def _pack_slot(
            toks: list[str],
            vals: list[float],
            kinds: list[int],
            L: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            ids = np.full(L, PAD_ID, dtype=np.int64)
            cv = np.zeros(L, dtype=np.float32)
            ck = np.full(L, int(ContKind.CATEGORICAL), dtype=np.int8)
            n = min(len(toks), L)
            for i in range(n):
                ids[i] = self._tokenizer.token_to_id(toks[i])
                cv[i] = float(vals[i])
                ck[i] = int(kinds[i])
            # Pad positions: zero out continuous channel (already 0) and
            # force kind CATEGORICAL so they route through the pad-id
            # embedding rather than a continuous encoder.
            return ids, cv, ck

        tech_ids = np.full((max_t, L_t), PAD_ID, dtype=np.int64)
        tech_cv = np.zeros((max_t, L_t), dtype=np.float32)
        tech_ck = np.full((max_t, L_t), int(ContKind.CATEGORICAL), dtype=np.int8)
        tech_mask = np.zeros(max_t, dtype=np.int8)
        for i in range(len(techs)):
            ids, cv, ck = _pack_slot(
                tech_tokens[i], tech_vals[i], tech_kinds[i], L_t
            )
            tech_ids[i] = ids
            tech_cv[i] = cv
            tech_ck[i] = ck
            tech_mask[i] = 1

        mach_ids = np.full((max_m, L_m), PAD_ID, dtype=np.int64)
        mach_cv = np.zeros((max_m, L_m), dtype=np.float32)
        mach_ck = np.full((max_m, L_m), int(ContKind.CATEGORICAL), dtype=np.int8)
        mach_mask = np.zeros(max_m, dtype=np.int8)
        for i in range(len(machines)):
            ids, cv, ck = _pack_slot(
                machine_tokens[i], machine_vals[i], machine_kinds[i], L_m
            )
            mach_ids[i] = ids
            mach_cv[i] = cv
            mach_ck[i] = ck
            mach_mask[i] = 1

        env_ids, env_cv, env_ck = _pack_slot(
            env_slot[0], env_slot[1], env_slot[2], L_e
        )

        return {
            "tech_token_ids": tech_ids,
            "tech_cont_values": tech_cv,
            "tech_cont_kinds": tech_ck,
            "tech_mask": tech_mask,
            "machine_token_ids": mach_ids,
            "machine_cont_values": mach_cv,
            "machine_cont_kinds": mach_ck,
            "machine_mask": mach_mask,
            "env_token_ids": env_ids,
            "env_cont_values": env_cv,
            "env_cont_kinds": env_ck,
        }

    def _machine_id_from_machine(self, machine: Any) -> int:
        """Return a stable machine identifier, defaulting to ``id()``."""
        mid = getattr(machine, "id", None)
        if mid is None:
            mid = id(machine)
        try:
            return int(mid)
        except (TypeError, ValueError):
            return int(hash(mid))

    # ------------------------------------------------------------------
    # Helpers for the ``set`` observation mode
    # ------------------------------------------------------------------

    _TECH_NAME_SUFFIX_RE = __import__("re").compile(r"^(.*?)_\d+$")

    def _technician_template(self, tech: Any) -> str:
        """Best-effort extraction of the template name from ``tech.name``.

        Scenario builders name technicians ``<template>_<index>``
        (e.g. ``junior_0``, ``motor_specialist_3``).  This helper
        strips the trailing ``_<digits>`` so the cross-attention sees
        the bare template token (``junior``, ``motor_specialist``).
        Falls back to the raw name when the pattern doesn't match.
        """
        name = str(getattr(tech, "name", ""))
        m = self._TECH_NAME_SUFFIX_RE.match(name)
        return m.group(1) if m else (name or "unknown")

    def _tech_knowledge_features(self, tech: Any) -> dict[str, float]:
        """Aggregate knowledge-grid scalars used as per-tech features.

        Returns four scalars derived from the technician's knowledge
        grid: total ``volume``, peak ``max_k``, ``spec_idx`` (in [0, 1])
        and ``entropy``.  Missing methods return ``0.0`` so test fakes
        without a full ``ongoing.KnowledgeGrid`` keep working.
        """
        grid = getattr(tech, "knowledge_grid", None)

        def _safe(method_name: str, default: float = 0.0) -> float:
            if grid is None:
                return default
            fn = getattr(grid, method_name, None)
            if not callable(fn):
                return default
            try:
                v = float(fn())
            except Exception:
                return default
            return v if math.isfinite(v) else default

        return {
            "volume": _safe("knowledge_volume"),
            "max_k": _safe("get_max_knowledge"),
            # specialisation_index is in [0, 1] so cap defensively.
            "spec_idx": max(0.0, min(1.0, _safe("specialisation_index"))),
            "entropy": _safe("knowledge_entropy"),
        }

    def _tech_match(self, tech: Any, request: Any) -> float:
        """Return the knowledge match score ``1 - m_k`` for ``request``.

        Score in ``[0, 1]``: 1 = full expertise, 0 = no expertise (or
        no ticket / fake tech).  Identical to the ``RepairQuality``
        step metric so the reward and the observation see the same
        signal.
        """
        if request is None:
            return 0.0
        get_km = getattr(tech, "get_knowledge_multiplier", None)
        if not callable(get_km):
            return 0.0
        try:
            mult = float(get_km(request))
        except Exception:
            return 0.0
        return max(0.0, min(1.0, 1.0 - mult))

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

    def _action_mask(self) -> np.ndarray:
        """Return a ``(n_techs,)`` ``int8`` mask, 1 = valid (available).

        A technician is *available* iff they are neither ``busy``
        (currently performing a repair) nor ``_in_disruption``
        (currently absorbed by an injury / vacation / exhaustion
        hold).  Treating disrupted technicians as available would mis-
        lead the policy into queuing assignments behind a long
        absence; including them in the mask makes the action surface
        honest.

        Falls back to all-1 when every technician is unavailable so
        that the action space is never empty --- the agent must still
        pick someone to enqueue, even if the choice is forced.
        """
        techs = self.dispatcher.techs if self.dispatcher else []
        mask = np.asarray(
            [
                0
                if (
                    bool(getattr(t, "busy", False))
                    or bool(getattr(t, "_in_disruption", False))
                )
                else 1
                for t in techs
            ],
            dtype=np.int8,
        )
        if mask.size == 0 or int(mask.sum()) == 0:
            mask = np.ones(len(techs), dtype=np.int8)
        return mask

    def _obs(self) -> dict[str, Any]:
        if self.config.observation_representation == "set":
            payload = self._set_obs()
        elif self.config.observation_representation == "hybrid":
            payload = self._hybrid_obs()
        elif self.config.observation_representation == "token_ids":
            payload = self._token_id_obs()
        elif self.config.observation_representation == "tokens":
            payload = self._token_obs()
        else:
            payload = self._structured_obs()
        if self.config.expose_action_mask:
            mask = self._action_mask()
            if self.config.observation_representation == "set":
                # Pad / truncate the action mask to ``max_techs`` so the
                # agent's pointer head can consume a fixed-size logit
                # vector regardless of fleet size.
                max_t = int(self.config.max_techs)
                padded = np.zeros(max_t, dtype=np.int8)
                padded[: min(len(mask), max_t)] = mask[:max_t]
                mask = padded
            payload["action_mask"] = mask
        return payload

    def _info(self) -> dict[str, Any]:
        techs = getattr(self.dispatcher, "techs", []) if self.dispatcher else []
        # Per-technician assignment counts so far in the episode, keyed
        # by ``tech.name`` (uniqueness is enforced at env init).  The
        # runner uses this to plot an end-of-episode histogram.
        assignment_counts: dict[str, int] = {}
        for i, t in enumerate(techs):
            label = str(getattr(t, "name", f"tech_{getattr(t, 'id', i)}"))
            count = (
                int(self._tech_assignment_counts[i])
                if i < len(self._tech_assignment_counts)
                else 0
            )
            assignment_counts[label] = count

        return {
            "episode_step": self.episode_step,
            "sim_time": self._sim_time(),
            "has_open_ticket": self.current_request is not None,
            "pending_queue_size": self._queue_size(),
            "reward_breakdown": dict(self._last_reward_breakdown),
            "metrics": dict(self._last_step_metrics),
            "assignment_counts": assignment_counts,
            "machine_stats": self._per_machine_episode_stats(),
        }

    def _reward_component(self, name: str, raw: float) -> float:
        """Apply coefficient and enabled flag for a named reward component.

        When ``GymRewardConfig.normalize_components`` is True the raw
        value is first divided by its running standard deviation —
        the per-component :class:`RewardNormalizer` accumulates stats
        across episodes and across calls.  This keeps heterogeneously
        scaled signals (e.g. a ``[-1, 0]`` busy penalty vs. a
        ``[0, 100]`` knowledge-increment delta) comparable so the
        coefficients act as *relative* weights.
        """
        comp = getattr(self.config.reward, name, None)
        if comp is None or not comp.enabled:
            return 0.0
        value = float(raw)
        if getattr(self.config.reward, "normalize_components", False):
            value = self._reward_normalizer.normalize(name, value)
        return comp.coefficient * value

    def freeze_reward_normalizer(self) -> None:
        """Stop updating per-component reward stats — call at eval time."""
        self._reward_normalizer.freeze()

    def unfreeze_reward_normalizer(self) -> None:
        """Resume updating per-component reward stats."""
        self._reward_normalizer.unfreeze()

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

        # --- fleet_knowledge: reward fleet-wide expertise growth ---
        if self.config.reward.fleet_knowledge.enabled:
            fk = self._fleet_knowledge_raw()
            breakdown["fleet_knowledge"] = self._reward_component("fleet_knowledge", fk)

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

        # --- selection_diversity: spread assignments across the fleet ---
        if self.config.reward.selection_diversity.enabled:
            breakdown["selection_diversity"] = self._reward_component(
                "selection_diversity", self._selection_diversity_raw(tech_id)
            )

        # --- repair_quality: knowledge-matched assignment ---
        if self.config.reward.repair_quality.enabled:
            get_km = getattr(tech, "get_knowledge_multiplier", None)
            if callable(get_km):
                multiplier = float(get_km(request))
                quality = max(0.0, min(1.0, 1.0 - multiplier))
            else:
                quality = 0.0
            breakdown["repair_quality"] = self._reward_component(
                "repair_quality", quality
            )

        # --- knowledge_increment: dense fleet-knowledge growth ---
        # Computed as the delta in mean per-tech knowledge volume
        # since the previous decision step.  Between two consecutive
        # decisions the simulator may have completed several queued
        # repairs, so this signal captures the *aggregate* knowledge
        # gain in that window (not strictly attributable to the
        # current action — see the config docstring).  Floored at 0
        # so the agent is never penalised for a stale (or pre-loaded)
        # snapshot drifting downward.
        if self.config.reward.knowledge_increment.enabled:
            current_volume = self._fleet_mean_knowledge_volume()
            increment = max(0.0, current_volume - self._prev_fleet_knowledge)
            self._prev_fleet_knowledge = current_volume
            breakdown["knowledge_increment"] = self._reward_component(
                "knowledge_increment", increment
            )

        self._last_reward_breakdown = breakdown
        return float(sum(breakdown.values()))

    # ------------------------------------------------------------------
    # Raw reward helpers for new components
    # ------------------------------------------------------------------

    def _fleet_knowledge_raw(self) -> float:
        """Return a bounded reward for the fleet's accumulated knowledge.

        The raw signal is::

            tanh(mean_per_tech_volume / fleet_knowledge_scale)

        where ``mean_per_tech_volume`` is the average of each
        technician's knowledge-grid volume, computed via:

        1. ``grid.knowledge_volume()`` if available (newer
           ``ongoing.KnowledgeGrid`` API),
        2. else ``grid.get_max_knowledge()`` as a safe fallback,
        3. else ``tech.knowledge`` (e.g. lightweight test fakes),
        4. else ``0.0``.

        ``tanh`` keeps the result in ``[0, 1]`` regardless of how big
        the knowledge grid grows, so the component cannot overshadow
        the rest of the reward stack — its scale is fully controlled
        by ``RewardComponentConfig.coefficient``.
        """
        mean_volume = self._fleet_mean_knowledge_volume()
        scale = float(getattr(self.config, "fleet_knowledge_scale", 10.0))
        if scale <= 0.0:
            return 0.0
        return float(math.tanh(mean_volume / scale))

    def _fleet_mean_knowledge_volume(self) -> float:
        """Return the fleet's current mean per-technician knowledge volume.

        Robust to several ``KnowledgeGrid`` API shapes and test fakes
        (see ``_fleet_knowledge_raw`` for the fallback chain).  Returns
        ``0.0`` if the fleet is empty.
        """
        techs = getattr(self.dispatcher, "techs", None) or []
        if not techs:
            return 0.0
        total = 0.0
        for tech in techs:
            grid = getattr(tech, "knowledge_grid", None)
            if grid is not None and hasattr(grid, "knowledge_volume"):
                value = float(grid.knowledge_volume())
            elif grid is not None and hasattr(grid, "get_max_knowledge"):
                value = float(grid.get_max_knowledge())
            else:
                value = float(getattr(tech, "knowledge", 0.0))
            if not math.isfinite(value):
                value = 0.0
            total += max(0.0, value)
        return total / len(techs)

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
            estimated = float(compute(base, request))
        else:
            estimated = base
        # Ratio < 1 when tech is knowledgeable (faster), > 1 when fatigued.
        # Log < 0 means faster than base (good), > 0 means slower (bad).
        ratio = estimated / max(base, 1e-6)
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

    def _update_machine_state_tracking(self) -> None:
        """Refresh downtime / production / breakdown tracking.

        Called on every step so that episode-level metrics
        (``fleet_availability_rate``, ``mttr``, the per-machine
        histograms…) remain accurate whether or not the matching
        reward components are enabled.
        """
        machines = self._factory_machines()
        now = self._sim_time()

        for m in machines:
            mid = getattr(m, "machine_id", id(m))
            # Capture a stable label the first time we see this machine
            if mid not in self._machine_labels:
                label = getattr(m, "name", None) or f"{getattr(m, 'mtype', 'machine')}_{mid}"
                self._machine_labels[mid] = str(label)

            broken_now = bool(getattr(m, "broken", False))
            # "Productive" processing = the machine claims it's
            # processing AND it isn't broken.  The simulator leaves
            # ``is_processing=True`` while a machine waits for a
            # repair, so without this guard the two intervals would
            # double-count and overlap.
            processing_now = (
                bool(getattr(m, "is_processing", False)) and not broken_now
            )

            # -- Downtime --
            if broken_now:
                if mid not in self._machine_down_since:
                    # Edge: just transitioned to broken → bump breakdown count.
                    self._machine_down_since[mid] = now
                    self._machine_breakdown_counts[mid] = (
                        self._machine_breakdown_counts.get(mid, 0) + 1
                    )
            elif mid in self._machine_down_since:
                elapsed = now - self._machine_down_since.pop(mid)
                self._total_downtime += elapsed
                self._machine_total_downtime[mid] = (
                    self._machine_total_downtime.get(mid, 0.0) + elapsed
                )

            # -- Production time (per-machine) --
            if processing_now:
                if mid not in self._machine_processing_since:
                    self._machine_processing_since[mid] = now
            elif mid in self._machine_processing_since:
                elapsed = now - self._machine_processing_since.pop(mid)
                self._machine_total_processing_time[mid] = (
                    self._machine_total_processing_time.get(mid, 0.0) + elapsed
                )

    def _per_machine_episode_stats(self) -> dict[str, dict[str, float]]:
        """Return ``{label: {maintenance, production, breakdowns}}``.

        Pending-broken / pending-processing intervals are flushed using
        the current sim time so the snapshot is accurate even mid-episode.
        """
        now = self._sim_time()
        stats: dict[str, dict[str, float]] = {}
        all_mids = (
            set(self._machine_labels.keys())
            | set(self._machine_total_downtime.keys())
            | set(self._machine_total_processing_time.keys())
            | set(self._machine_breakdown_counts.keys())
        )
        for mid in all_mids:
            label = self._machine_labels.get(mid, f"machine_{mid}")
            maintenance = self._machine_total_downtime.get(mid, 0.0)
            if mid in self._machine_down_since:
                maintenance += max(0.0, now - self._machine_down_since[mid])
            production = self._machine_total_processing_time.get(mid, 0.0)
            if mid in self._machine_processing_since:
                production += max(0.0, now - self._machine_processing_since[mid])
            stats[label] = {
                "maintenance_time": float(maintenance),
                "production_time": float(production),
                "breakdowns": float(self._machine_breakdown_counts.get(mid, 0)),
            }
        return stats

    def _on_repair_completed(self, request: Any, repair_duration: float) -> None:
        """Dispatcher callback invoked when a repair finishes."""
        _ = request
        self._completed_repair_counter += 1
        self._total_repair_time += float(repair_duration)

    def _selection_diversity_raw(self, action: int) -> float:
        """Reward for spreading assignments across the fleet.

        Returns a value in ``[0, 1]`` based on per-technician assignment
        counts collected so far in the episode (BEFORE this step is
        recorded):

        * 1.0 when the chosen technician is tied for the *least* used
          (or when no assignments have happened yet).
        * 0.0 when the chosen technician is tied for the *most* used.
        * Linearly interpolated in between using the
          ``(max - chosen) / (max - min)`` ratio over the per-tech
          counts.
        """
        counts = self._tech_assignment_counts
        if not counts or action < 0 or action >= len(counts):
            return 0.0
        if sum(counts) == 0:
            # First assignment of the episode — diversity is undefined,
            # treat any pick as fully diverse.
            return 1.0
        chosen = counts[action]
        mn, mx = min(counts), max(counts)
        if mx == mn:
            # All technicians used the same amount so far — no
            # imbalance, every choice is equally diverse.
            return 1.0
        return float((mx - chosen) / (mx - mn))

    def _downtime_cost_raw(self) -> float:
        """Negative fraction of machine-time lost to breakdowns — [-1, 0].

        Tracks accumulated downtime across all machines since episode
        start, normalised by total available machine-time.  Captures
        the systemic cost of slow repairs.
        """
        self._update_machine_state_tracking()

        machines = self._factory_machines()
        now = self._sim_time()

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

        # Seed the disruption RNG on each technician deterministically
        # from the env-level seed.  Done *after* the scenario rebuild
        # so the freshly-spawned technicians are the ones that get
        # seeded.  Falls back to non-deterministic disruption timing
        # when ``seed`` is None (Gymnasium contract).
        if seed is not None and hasattr(self.dispatcher, "seed_disruptions"):
            self.dispatcher.seed_disruptions(int(seed))

        # Run MCA warmup on first reset
        self._run_warmup()

        self.episode_step = 0
        self.current_request = None
        self._last_reward_breakdown = {}
        self._last_step_metrics = {}
        self._breakdown_counter = 0
        self._repair_counter = 0
        self._completed_repair_counter = 0
        self._total_repair_time = 0.0
        self._prev_finished_products = 0
        self._machine_down_since = {}
        self._total_downtime = 0.0
        self._machine_total_downtime = {}
        self._machine_total_processing_time = {}
        self._machine_processing_since = {}
        self._machine_breakdown_counts = {}
        self._machine_labels = {}
        self._tech_assignment_counts = [0] * len(self.dispatcher.techs)
        self._tech_last_assignment_time = [-1.0] * len(self.dispatcher.techs)
        self._initial_mean_knowledge_volume = self._fleet_mean_knowledge_volume()
        # Seed the per-step knowledge_increment baseline at the same
        # value so the first decision's increment is computed against
        # the genuine episode-start knowledge.  The normaliser's
        # running stats are *not* reset — they accumulate across
        # episodes within this env instance.
        self._prev_fleet_knowledge = self._initial_mean_knowledge_volume
        self._advance_until_next_ticket()
        return self._obs(), self._info()

    def step(self, action: int):
        """Apply the agent's technician choice to the current ticket.

        ``action`` is the integer index of the chosen technician.  The
        only validation performed here is the range check
        ``0 <= action < n_techs``; in particular this method does
        **not** consult the action mask.  Picking a technician who is
        currently busy (in a repair) or absorbed by a disruption is
        permitted --- the dispatcher's ``_repair_job`` queues the
        request behind whatever the technician is doing, and it will
        run when they next become available.  Agents that want to
        avoid this behaviour should read ``obs['action_mask']`` and
        sample only from positions where the mask is 1.
        """
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

        # Step metrics — a metric may return either a scalar or a dict
        # mapping sub-keys to values; dicts are flattened into individual
        # series with names like ``<metric>/<subkey>``.
        self._last_step_metrics = {}
        for m in STEP_METRICS:
            value = m.compute(tech, request, self)
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    self._last_step_metrics[f"{m.name}/{subkey}"] = float(subval)
            else:
                self._last_step_metrics[m.name] = float(value)

        reward = self._reward_for_assignment(request, action)
        # Record the assignment for the diversity counter AFTER computing
        # the reward, so the reward sees pre-assignment counts and the
        # *next* step sees this assignment in the history.
        if 0 <= action < len(self._tech_assignment_counts):
            self._tech_assignment_counts[action] += 1
            self._tech_last_assignment_time[action] = self._sim_time()
        self._advance_until_next_ticket()
        # Update bookkeeping every step regardless of reward config so
        # episode-level metrics stay accurate when their matching reward
        # components are disabled.
        self._update_machine_state_tracking()
        terminated = self._is_done()

        # On episode end, append episode-level metrics.  Run a final
        # tracking update first so any machines still down at episode
        # boundary are accounted for.
        if terminated:
            self._update_machine_state_tracking()
            self._last_step_metrics.update(
                {m.name: m.compute(self) for m in EPISODE_METRICS}
            )

            # Terminal reward — fires once, proportional to finished
            # products.  Tracked in the reward breakdown so the
            # decomposition plots remain accurate.
            if self.config.reward.terminal_finished_products.enabled:
                sinks = getattr(self.dispatcher, "sinks", []) or []
                n_finished = sum(int(getattr(s, "completed", 0)) for s in sinks)
                bonus = self._reward_component(
                    "terminal_finished_products", float(n_finished)
                )
                self._last_reward_breakdown["terminal_finished_products"] = bonus
                reward += bonus

            # Terminal reward — fires once, proportional to the fleet's
            # knowledge *growth* during the episode (final minus initial
            # mean per-technician volume).  The raw is scaled by
            # ``fleet_knowledge_scale`` for unit-consistency with the
            # per-step ``fleet_knowledge`` component but is *not*
            # saturated — over long episodes the growth genuinely
            # accumulates, and clipping it would kill the gradient.
            # Pre-loaded profile grids are subtracted out by design, so
            # the agent is credited only for what *it* added.
            if self.config.reward.terminal_fleet_knowledge.enabled:
                final_vol = self._fleet_mean_knowledge_volume()
                initial_vol = float(
                    getattr(self, "_initial_mean_knowledge_volume", 0.0)
                )
                growth = max(0.0, final_vol - initial_vol)
                scale = float(getattr(self.config, "fleet_knowledge_scale", 10.0))
                raw = growth / scale if scale > 0.0 else 0.0
                bonus = self._reward_component("terminal_fleet_knowledge", raw)
                self._last_reward_breakdown["terminal_fleet_knowledge"] = bonus
                reward += bonus

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
