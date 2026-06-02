"""Centralised settings for KATA.

``KATAConfig`` is the single source of truth for all run-time options.
It is a ``pydantic-settings`` model that can be populated from:
  1. A JSON file whose path is given by the ``KATA_CONF_PATH`` environment
     variable (defaults to ``run_configs/config.json``).
  2. Environment variables (prefixed with ``KATA_``).
  3. Hard-coded defaults defined below.

Entity-level configuration models (``TechnicianConfig``, ``MachineConfig``,
``ComponentConfig``, …) live alongside their respective entity files under
``src/kata/entities/…/config.py`` and are **re-exported from this module**
for backward compatibility.

Usage
-----
>>> from kata.core.config import get_config
>>> cfg = get_config()
>>> cfg.sim.technicians.travel_time
10
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from kata.entities.buffers.config import BufferConfig  # noqa: F401
from kata.entities.components.config import ComponentConfig  # noqa: F401
from kata.entities.machine_feeder.config import MachineFeederConfig  # noqa: F401
from kata.entities.machines.config import MachineConfig
from kata.entities.production_line.config import ProductionLineConfig  # noqa: F401
from kata.entities.routers.config import RouterConfig  # noqa: F401
from kata.entities.sinks.config import SinkConfig  # noqa: F401
from kata.entities.sources.config import SourceConfig  # noqa: F401
from kata.entities.tech_dispatcher.config import TechDispatcherConfig  # noqa: F401

# ---------------------------------------------------------------------------
# Re-export entity-level configs so existing imports keep working.
# ---------------------------------------------------------------------------
from kata.entities.technicians.config import TechnicianConfig
from kata.EntityFactories.config import SyntheticTicketFactoryConfig
from kata.features.breakdown.config import (  # noqa: F401
    SimpleBreakdownConfig,
    WeibullBreakdownConfig,
)

# ---------------------------------------------------------------------------
# Simulation-environment sub-configs (global/cross-entity concerns)
# ---------------------------------------------------------------------------


class DisruptionTypeConfig(BaseModel):
    """One named disruption type with its own trigger mechanism.

    Three trigger families are supported:

    * ``"random"`` --- a Poisson process with mean rate ``rate``.  Models
      stochastic, memory-less events like injuries: inter-arrival times
      are sampled from ``Exponential(1/rate)``.
    * ``"fatigue"`` --- the technician is polled every
      ``poll_interval`` sim time units, with per-poll firing probability
      ``fatigue_coefficient * fatigue * poll_interval``.  Models
      exhaustion-driven absences that scale with how worn out the
      worker is.
    * ``"periodic"`` --- fires every ``interval`` sim time units, with
      optional ``jitter`` uniform offset.  Models scheduled events
      such as vacations.

    Duration of the absence is drawn from ``Normal(duration_mu,
    duration_sig)`` regardless of trigger.

    ``preemptive`` controls whether this disruption type can preempt
    an ongoing repair: typical values are ``True`` for injury /
    exhaustion (you simply can't keep working), ``False`` for vacation
    (you would naturally postpone leaving rather than abandon a job).
    """

    trigger: Literal["random", "fatigue", "periodic"] = Field(
        description="Mechanism that decides when this disruption fires."
    )
    duration_mu: float = Field(
        gt=0.0,
        description="Mean duration of an instance, in simulation time units.",
    )
    duration_sig: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of the instance duration.",
    )
    preemptive: bool = Field(
        default=True,
        description=(
            "When True, this disruption preempts an ongoing repair "
            "(the partially-completed ticket is re-queued).  When "
            "False, the disruption waits for the current repair to "
            "finish before claiming the technician."
        ),
    )

    # Trigger-specific parameters.  Only the field matching ``trigger``
    # is consulted at runtime; the validator below enforces presence.
    rate: float | None = Field(
        default=None,
        description=(
            "Expected events per simulation time unit for "
            "``trigger='random'`` (Poisson rate)."
        ),
    )
    fatigue_coefficient: float | None = Field(
        default=None,
        description=(
            "Hazard multiplier for ``trigger='fatigue'``: the per-poll "
            "probability is ``fatigue_coefficient * fatigue * "
            "poll_interval``."
        ),
    )
    poll_interval: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Sim time between fatigue polls when ``trigger='fatigue'``."
        ),
    )
    interval: float | None = Field(
        default=None,
        description=(
            "Sim time between firings when ``trigger='periodic'``."
        ),
    )
    jitter: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Uniform random offset added to each periodic firing time "
            "(in sim time units).  Default 0 = strictly regular schedule."
        ),
    )

    @model_validator(mode="after")
    def _check_trigger_params(self) -> "DisruptionTypeConfig":
        if self.trigger == "random":
            if self.rate is None or self.rate <= 0.0:
                msg = "trigger='random' requires a positive ``rate``"
                raise ValueError(msg)
        elif self.trigger == "fatigue":
            if self.fatigue_coefficient is None or self.fatigue_coefficient <= 0.0:
                msg = "trigger='fatigue' requires a positive ``fatigue_coefficient``"
                raise ValueError(msg)
        elif self.trigger == "periodic":
            if self.interval is None or self.interval <= 0.0:
                msg = "trigger='periodic' requires a positive ``interval``"
                raise ValueError(msg)
        return self


def _default_disruption_pool() -> dict[str, DisruptionTypeConfig]:
    return {
        "injury": DisruptionTypeConfig(
            trigger="random",
            rate=1e-4,           # ~1 event per 10 000 sim time units
            duration_mu=240.0,
            duration_sig=60.0,
            preemptive=True,
        ),
        "exhaustion": DisruptionTypeConfig(
            trigger="fatigue",
            fatigue_coefficient=1e-3,  # at fatigue=1 and poll=60s → p≈6% per poll
            poll_interval=60.0,
            duration_mu=120.0,
            duration_sig=30.0,
            preemptive=True,
        ),
        "vacation": DisruptionTypeConfig(
            trigger="periodic",
            interval=8000.0,
            jitter=400.0,
            duration_mu=480.0,
            duration_sig=120.0,
            preemptive=False,
        ),
    }


class DisruptionConfig(BaseModel):
    """Top-level disruption settings.

    Each entry in ``dis_dict`` is a :class:`DisruptionTypeConfig` and
    runs its own independent process on every technician for the
    duration of the episode --- so a technician can experience an
    arbitrary number of disruptions, of mixed types, over a long run.
    """

    dis_dict: dict[str, DisruptionTypeConfig] = Field(
        default_factory=_default_disruption_pool,
        description=(
            "Mapping of disruption-type name -> per-type configuration "
            "(trigger mechanism, duration distribution, preemption flag)."
        ),
    )

    # Legacy / convenience alias retained so callers that previously
    # read ``interrupt_on_disrupt`` still see *something*; the source of
    # truth is now the per-type ``preemptive`` flag.  Reads as ``True``
    # iff any configured type is preemptive.
    @property
    def interrupt_on_disrupt(self) -> bool:
        return any(cfg.preemptive for cfg in self.dis_dict.values())


class RepairConfig(BaseModel):
    """Global switches and parameters that control how repair time is computed.

    The knowledge multiplier follows a saturating-exponential law:

        m_k = min_repair_fraction
              + (1 - min_repair_fraction) * exp(-knowledge_sensitivity * k)

    where ``k`` is the technician's knowledge for the current request:

      * ``k = 0``      → ``m_k = 1`` (no speed-up, full base repair time)
      * ``k → +inf``   → ``m_k = min_repair_fraction`` (asymptotic floor)
      * ``knowledge_sensitivity`` tunes how quickly knowledge translates
        into a speed-up — higher values give a steeper drop near ``k=0``.
    """

    knowledge_enabled: bool = Field(
        default=True,
        description="Apply knowledge multiplier to base repair time when True.",
    )
    fatigue_enabled: bool = Field(
        default=True,
        description="Apply fatigue multiplier to base repair time when True.",
    )
    min_repair_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Lower bound on the knowledge multiplier — even an arbitrarily "
            "experienced technician will still spend at least "
            "``min_repair_fraction * base_repair_time`` on a repair."
        ),
    )
    knowledge_sensitivity: float = Field(
        default=0.002,
        gt=0.0,
        description=(
            "Decay rate ``alpha`` of the saturating-exponential knowledge "
            "response.  Larger values produce a steeper speed-up at low "
            "knowledge; smaller values give a more gradual descent.  "
            "With the default knowledge-grid settings (learning_rate=0.1, "
            "propagation_sigma=1.0) ``alpha=0.002`` saturates the "
            "multiplier at the floor after ~60–70 similar repairs."
        ),
    )
    failure_wise_knowledge_parameters: bool = Field(
        default=False,
        description=(
            "When False (the default), every failure uses the global "
            "``min_repair_fraction`` and ``knowledge_sensitivity`` "
            "values.  When True, the simulator first checks the "
            "failed component for a per-component override (set via "
            "``ComponentConfig.min_repair_fraction`` / "
            "``ComponentConfig.knowledge_sensitivity``) and falls back "
            "to the global values when the component has none.  Use "
            "this to model that some failures are barely accelerated "
            "by experience while others can be heavily sped up by a "
            "trained technician."
        ),
    )


class GlobalTechniciansConfig(BaseModel):
    """Global, cross-technician parameters used by the simulation environment."""

    travel_time: int = Field(
        default=10,
        gt=0,
        description="Travel time (in simulation time units) between any two machines.",
    )
    fatigue_model: str = Field(
        default="exponential",
        description=(
            "Fatigue-multiplier model.  Both shapes are slowdowns "
            "(multiplier >= 1, growing with fatigue):"
            " 'linear'      -> 1 + fatigue            (range [1, 2]);"
            " 'exponential' -> exp(alpha * fatigue)   (range [1, exp(alpha)])."
        ),
    )
    fatigue_alpha: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Alpha parameter for the exponential fatigue model.  At "
            "fatigue=1 a tech needs ``exp(alpha)`` times the base repair "
            "time — alpha=0.5 -> ~1.65x; alpha=1.0 -> ~2.72x."
        ),
    )


class SimEnvConfig(BaseModel):
    """Top-level configuration for the SimPy simulation environment."""

    disruptions: DisruptionConfig = Field(default_factory=DisruptionConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    technicians: GlobalTechniciansConfig = Field(
        default_factory=GlobalTechniciansConfig
    )


class RewardComponentConfig(BaseModel):
    """Configuration for one reward component."""

    enabled: bool = Field(
        default=True,
        description="Whether this reward component contributes to total reward.",
    )
    coefficient: float = Field(
        default=1.0,
        description="Linear coefficient applied to this reward component.",
    )


def _disabled_reward_component() -> RewardComponentConfig:
    return RewardComponentConfig(enabled=False, coefficient=1.0)


class GymRewardConfig(BaseModel):
    """Composable reward settings for the Gym environment.

    Each component produces a raw scalar at assignment time.  The final
    reward is ``sum(coefficient * raw  for each enabled component)``.
    """

    assignment: RewardComponentConfig = Field(
        default_factory=RewardComponentConfig,
        description="Constant assignment reward component.",
    )
    wait_time: RewardComponentConfig = Field(
        default_factory=RewardComponentConfig,
        description="Penalty/reward component based on ticket waiting time.",
    )
    queue_size: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description="Component based on pending queue size at dispatch time.",
    )
    busy_technician: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description="Component that penalizes assigning already-busy technicians.",
    )
    # -- new reward components ------------------------------------------------
    fatigue_cost: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Penalizes high-fatigue assignments.  Raw value is the "
            "negative fatigue level of the assigned technician in [0, 1]."
        ),
    )
    fleet_knowledge: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Rewards growing the fleet-wide knowledge volume.  Raw value "
            "is ``tanh(mean_per_tech_volume / fleet_knowledge_scale)`` so "
            "it stays in [0, 1] regardless of how large the underlying "
            "knowledge grid grows.  Replaces the old ``knowledge_match`` "
            "(which tied the signal to the *currently broken* machine "
            "rather than the fleet's overall expertise)."
        ),
    )
    workload_balance: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Penalizes unbalanced workload across the fleet.  Raw value "
            "is the negative standard deviation of current technician "
            "fatigue levels (measured before the new assignment lands; "
            "fatigue is updated by the simulator at repair completion, "
            "not at dispatch)."
        ),
    )
    estimated_repair_time: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Penalizes long expected repair times.  Raw value is the "
            "negative log of the estimated repair time (accounting for "
            "knowledge and fatigue), normalized by the base repair time."
        ),
    )
    machine_criticality: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Rewards urgently fixing critical machines.  Raw value is "
            "a bonus proportional to the machine's productivity (total "
            "processed) and its input buffer backlog."
        ),
    )
    # -- manufacturing-KPI reward components ---------------------------------
    fleet_availability: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Rewards high fleet availability.  Raw value is the fraction "
            "of machines currently operational (not broken), in [0, 1]."
        ),
    )
    throughput_delta: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Rewards production throughput.  Raw value is the change in "
            "finished products since the previous assignment step, "
            "clipped to [0, 1] for stability."
        ),
    )
    repair_backlog_age: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Penalizes an aging repair backlog.  Raw value is the "
            "negative mean waiting time of ALL queued requests "
            "(not just the current ticket), saturated via tanh."
        ),
    )
    technician_utilization: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Rewards productive use of the technician workforce.  "
            "Raw value peaks at an optimal utilization ratio and "
            "penalizes both under- and over-utilization, in [-1, 1].  "
            "Utilization is measured BEFORE the new assignment is "
            "reflected (the freshly-assigned tech only flips ``busy`` "
            "after acquiring the SimPy resource)."
        ),
    )
    selection_diversity: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Encourages spreading assignments across the fleet.  Raw "
            "value is in ``[0, 1]``: 1 when the chosen technician is "
            "tied for *least* used so far in the episode, 0 when they "
            "are tied for *most* used, with linear interpolation in "
            "between (computed from the per-episode assignment counts "
            "BEFORE this step is recorded).  At episode start, when no "
            "assignments have happened yet, the raw value is 1."
        ),
    )
    terminal_finished_products: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Terminal bonus paid once on the episode's last step, "
            "proportional to the total number of products that reached "
            "a sink during the episode.  Use this to make "
            "*finished products* the dominant training signal without "
            "flooding every step with dense throughput rewards.  Raw "
            "value is the integer product count; final contribution is "
            "``coefficient × n_finished``."
        ),
    )
    terminal_fleet_knowledge: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Terminal bonus paid once on the episode's last step, "
            "proportional to the fleet's knowledge *growth* during the "
            "episode (final minus initial mean per-tech volume), scaled "
            "by ``fleet_knowledge_scale``.  Pre-loaded profile grids "
            "are subtracted out by design, so the agent is credited "
            "only for knowledge it accumulated.  Raw value is "
            "``growth / fleet_knowledge_scale``; final contribution is "
            "``coefficient × raw``."
        ),
    )
    repair_quality: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Per-assignment reward for picking a technician whose "
            "accumulated expertise matches the failed component family.  "
            "Raw value is ``1 - m_k`` in [0, 1], where ``m_k`` is the "
            "knowledge multiplier the chosen technician would apply to "
            "this repair (1 = no expertise, near 0 = full expertise).  "
            "This is the reward counterpart of the ``repair_quality`` "
            "step metric and is the load-bearing signal for "
            "knowledge-matched assignment."
        ),
    )
    downtime_cost: RewardComponentConfig = Field(
        default_factory=_disabled_reward_component,
        description=(
            "Penalizes accumulated machine downtime.  Raw value is "
            "the negative fraction of total machine-time lost to "
            "breakdowns since episode start, in [-1, 0]."
        ),
    )


class GymEnvConfig(BaseModel):
    """Configuration for the Gymnasium wrapper around the simulation."""

    max_episode_steps: int = Field(
        default=10_000,
        gt=0,
        description="Maximum number of environment steps per episode.",
    )
    max_sim_time: float = Field(
        default=10_000.0,
        gt=0.0,
        description="Maximum simulation time in one episode.",
    )
    invalid_action_mode: Literal["penalize", "terminate", "raise"] = Field(
        default="penalize",
        description=(
            "Behavior when an invalid technician action is provided: "
            "'penalize' applies a penalty and continues, "
            "'terminate' ends the episode, 'raise' raises ValueError."
        ),
    )
    invalid_action_penalty: float = Field(
        default=-1.0,
        description="Reward added when an invalid action is taken.",
    )
    assignment_reward: float = Field(
        default=0.0,
        description="Raw value of the assignment reward component.",
    )
    ticket_wait_time_penalty: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Raw penalty multiplier for waiting time before assignment. "
            "The resulting raw component is `-ticket_wait_time_penalty * wait_time`."
        ),
    )
    fleet_knowledge_scale: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Saturation scale for the ``fleet_knowledge`` reward component: "
            "raw = tanh(mean_per_tech_knowledge_volume / fleet_knowledge_scale).  "
            "Smaller values make the reward saturate earlier (good when the "
            "knowledge grid stays small); larger values keep the curve "
            "informative for longer-horizon training runs."
        ),
    )
    reward: GymRewardConfig = Field(
        default_factory=GymRewardConfig,
        description="Composable reward settings with configurable sub-components.",
    )
    observation_representation: Literal[
        "structured", "tokens", "token_ids", "hybrid", "set"
    ] = Field(
        default="structured",
        description=(
            "Observation payload format. "
            "'structured' keeps numeric fields, "
            "'tokens' returns fixed-size textual token tuples, "
            "'token_ids' returns integer ID sequences for Transformer input, "
            "'hybrid' returns the same integer sequence (with a ``<NUM>`` "
            "placeholder at numerical-value positions) PLUS parallel "
            "``cont_values`` / ``cont_kinds`` channels that carry the raw "
            "scalars and route them to PLE / Time2Vec / Fourier encoders, "
            "'set' emits THREE grouped streams (technicians, machines, env) "
            "where each per-entity slot is a fixed-length hybrid triple "
            "sub-sequence padded out to ``max_techs`` / ``max_machines`` "
            "slots, with companion masks identifying valid slots."
        ),
    )
    max_techs: int = Field(
        default=30,
        gt=0,
        description=(
            "Hard cap on the technician slot count exposed by the 'set' "
            "observation mode.  Real fleets smaller than this are padded "
            "(with ``tech_mask`` indicating valid slots); larger fleets "
            "fail at env init.  Set to the largest fleet the agent will "
            "encounter across scenarios so the network can be reused."
        ),
    )
    max_machines: int = Field(
        default=100,
        gt=0,
        description=(
            "Hard cap on the machine slot count exposed by the 'set' "
            "observation mode.  Padding/mask semantics mirror ``max_techs``."
        ),
    )
    set_tech_slot_length: int = Field(
        default=16,
        gt=0,
        description=(
            "Number of (token_id, cont_value, cont_kind) triples emitted "
            "per technician slot in 'set' observation mode.  Within-slot "
            "padding fills shorter feature streams.  Default 16 leaves "
            "headroom over the 14 features currently emitted (template "
            "+ 6 knowledge-derived features + 7 state features)."
        ),
    )
    set_machine_slot_length: int = Field(
        default=12,
        gt=0,
        description=(
            "Per-machine slot length in 'set' observation mode.  Default "
            "12 holds the 11 features emitted: machine type, broken / "
            "processing flags, is-current-ticket flag, current ticket "
            "component type, total processed, in/out buffer sizes, "
            "breakdown count, downtime, mean time between failures."
        ),
    )
    set_env_length: int = Field(
        default=16,
        gt=0,
        description=(
            "Length of the global env-token stream in 'set' mode.  "
            "Default 16 holds the 14 features emitted: ticket info "
            "(has-ticket, machine type, component type), simulation "
            "time, ticket age, queue size, broken count, processing "
            "count, plus 2 × 3 = 6 lookahead tokens for the next two "
            "queued tickets (machine type / component type / age)."
        ),
    )
    observation_mode: Literal[
        "ticket_only",
        "broken_machine",
        "factory_level",
        "tech_aware",
    ] = Field(
        default="ticket_only",
        description=(
            "Level of context to include in token observations. "
            "``tech_aware`` is a superset of ``factory_level`` that also "
            "exposes per-technician *ticket-specific* signals — expected "
            "repair time, knowledge match for this component type, age "
            "since last assignment — plus the failed component type and "
            "a peek at the next 2 queued tickets."
        ),
    )
    token_observation_length: int = Field(
        default=64,
        gt=0,
        description="Fixed number of textual tokens emitted in token observations.",
    )
    token_max_length: int = Field(
        default=64,
        gt=0,
        description="Maximum character length for each textual token.",
    )
    token_pad_value: str = Field(
        default="<PAD>",
        description="Token used to pad token observations to fixed length.",
    )
    include_technician_fatigue_tokens: bool = Field(
        default=False,
        description="Include fleet fatigue tokens when using token observations.",
    )
    include_technician_knowledge_tokens: bool = Field(
        default=False,
        description="Include fleet knowledge tokens when using token observations.",
    )
    next_ticket_lookahead: int = Field(
        default=4,
        ge=0,
        le=12,
        description=(
            "Number of queued tickets after the current one whose "
            "(machine_type, component_type, age) is exposed in "
            "``tech_aware`` token observations as ``NEXT{i}_*`` triples.  "
            "Tradeoff: each slot costs 6 sequence positions and ~3 vocab "
            "entries; deeper lookahead gives the policy more scheduling "
            "context but eats into ``token_observation_length``."
        ),
    )
    include_queue_composition_tokens: bool = Field(
        default=False,
        description=(
            "Add ``QC_<component_type> C_<count>`` tokens summarising the "
            "*entire* pending queue by failed component type.  Independent "
            "of ``next_ticket_lookahead`` (which only describes individual "
            "tickets in order) and useful when the agent should reason "
            "about queue *composition* rather than just the next few items."
        ),
    )
    include_broken_by_type_tokens: bool = Field(
        default=False,
        description=(
            "In ``factory_level`` / ``tech_aware`` observation modes, add "
            "``BROKEN_<machine_type> C_<count>`` tokens for every machine "
            "type with at least one broken machine.  Without this flag the "
            "fleet's broken machines collapse into a single bucketed count "
            "(``FACTORY_BROKEN``) — fine for small fleets, lossy at scale."
        ),
    )
    include_technician_assignment_count_tokens: bool = Field(
        default=False,
        description=(
            "Add ``TECH_{i} ASSIGN_COUNT C_*`` tokens describing how many "
            "tickets each technician has been assigned this episode.  Same "
            "signal the ``selection_diversity`` reward uses internally — "
            "exposing it lets the policy learn diversification directly."
        ),
    )
    # -- MCA / tokenizer warmup settings -----------------------------------
    use_mca_encoder: bool = Field(
        default=False,
        description=(
            "When True, run an MCA warmup phase on first reset to fit an "
            "encoder and pre-populate the tokenizer vocabulary."
        ),
    )
    warmup_steps: int = Field(
        default=200,
        gt=0,
        description="Number of heuristic steps during MCA warmup.",
    )
    mca_grid_shape: tuple[int, ...] = Field(
        default=(10, 10),
        description="Grid shape for the MCA encoder's output coordinates.",
    )
    mca_n_components: int = Field(
        default=2,
        gt=0,
        description="Number of MCA components to keep.",
    )
    tokenizer_seq_length: int = Field(
        default=64,
        gt=0,
        description="Fixed output length of the StateTokenizer.",
    )

    include_fatigue_in_observation: bool = Field(
        default=True,
        description="Include technicians' fatigue values in observation vectors.",
    )
    include_queue_size_in_observation: bool = Field(
        default=True,
        description="Include pending-repair queue size in observations.",
    )
    expose_action_mask: bool = Field(
        default=True,
        description=(
            "When True, the observation includes an ``action_mask`` field "
            "— a ``MultiBinary(n_techs)`` array where 1 means the "
            "technician is currently NOT busy (valid pick).  Agents that "
            "support action masking (e.g. PPOTransformerAgent) read it to "
            "constrain the policy to valid technicians.  Falls back to "
            "all-1 when every technician is busy, so a valid action "
            "always exists."
        ),
    )


class RandomizedScenarioConfig(BaseModel):
    """Per-episode factory randomisation.

    When ``enabled`` is True, the scenario builder is replaced with a
    sampler that, on every ``env.reset()``, draws a fresh combination
    of machines, technicians and product route from the configured
    pools.  Used during training to prevent the agent from overfitting
    to a single factory layout.

    The number of technicians is kept FIXED across episodes so the
    action space (``Discrete(n_techs)``) stays stable; their *profiles*
    are sampled from ``technician_templates``.  Machines vary in count
    (``n_machines_min``..``n_machines_max``) and templates.  The route
    is sampled from the set of machine types actually present in the
    drawn machines, so products always have somewhere to go.
    """

    enabled: bool = Field(
        default=False,
        description="When True, draw a new factory every env.reset().",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Seed for the per-episode sampler.  None = non-deterministic. "
            "Distinct from ``ExperimentConfig.seed`` so factory layouts "
            "can be repeated across agent seeds and vice-versa."
        ),
    )
    n_technicians: int = Field(
        default=4,
        gt=0,
        description="Fixed number of technicians (action space size).",
    )
    technician_templates: list[str] = Field(
        default_factory=lambda: ["expert", "senior", "generalist", "junior", "trainee"],
        description=(
            "Pool of technician templates the sampler can draw from "
            "(must exist in the technician template registry)."
        ),
    )
    n_machines_min: int = Field(
        default=10,
        gt=0,
        description="Lower bound on the number of machines sampled per episode.",
    )
    n_machines_max: int = Field(
        default=20,
        gt=0,
        description="Upper bound on the number of machines sampled per episode.",
    )
    machine_templates: list[str] = Field(
        default_factory=lambda: [
            "cnc_weibull",
            "assembly_mixed",
            "assembly_robot",
            "conveyor",
            "welder",
            "inspection",
        ],
        description=(
            "Pool of machine templates the sampler can draw from "
            "(must exist in the machine template registry)."
        ),
    )
    route_min_length: int = Field(
        default=2,
        gt=0,
        description="Lower bound on the product route length.",
    )
    route_max_length: int = Field(
        default=6,
        gt=0,
        description="Upper bound on the product route length (clamped by available types).",
    )
    randomize_eval: bool = Field(
        default=False,
        description=(
            "When False (the default), the eval environment uses a "
            "*single fixed factory* sampled once at construction time "
            "from this same pool — so the eval-return curve measures "
            "policy quality without scenario noise.  Set True to "
            "evaluate on freshly-sampled factories every eval episode "
            "(useful for measuring generalisation, but adds noise)."
        ),
    )
    eval_seed: int | None = Field(
        default=None,
        description=(
            "Seed used to draw the *fixed* eval scenario when "
            "``randomize_eval`` is False.  Defaults to ``seed + 1`` if "
            "left unset, so train and eval factories are distinct."
        ),
    )


class ProductConfig(BaseModel):
    """Configuration for a product type."""

    product_type: str = Field(
        default="generic",
        description="Unique name/type identifier for this product.",
    )
    route: list[str] = Field(
        default_factory=list,
        description="Ordered list of machine type names that define the production route.",
    )


# ---------------------------------------------------------------------------
# Top-level KATAConfig (the centralised settings object)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_FILE = "run_configs/config.json"


class KATAConfig(BaseSettings):
    """Centralised settings for KATA.

    Can be loaded from a JSON file (path set via ``KATA_CONF_PATH`` env var),
    from ``KATA_``-prefixed environment variables, or will fall back to the
    defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_prefix="KATA_",
        env_nested_delimiter="__",
        cli_parse_args=False,
    )

    # ------------------------------------------------------------------
    # Entity registries – keyed by a user-defined name/ID string
    # ------------------------------------------------------------------
    technicians: dict[str, TechnicianConfig] = Field(
        default={"technician_0": TechnicianConfig()},
        description="Registry of technician configurations keyed by technician name.",
    )
    machines: dict[str, MachineConfig] = Field(
        default={"machine_type_0": MachineConfig()},
        description="Registry of machine-type configurations keyed by machine type name.",
    )
    products: dict[str, ProductConfig] = Field(
        default={"product_type_0": ProductConfig()},
        description="Registry of product-type configurations keyed by product type name.",
    )

    # ------------------------------------------------------------------
    # Sub-environment configs
    # ------------------------------------------------------------------
    sim: SimEnvConfig = Field(
        default_factory=SimEnvConfig,
        description="Simulation-environment settings.",
    )
    gym: GymEnvConfig = Field(
        default_factory=GymEnvConfig,
        description="Gymnasium-wrapper settings.",
    )

    # ------------------------------------------------------------------
    # Factory configs
    # ------------------------------------------------------------------
    ticket_factory: SyntheticTicketFactoryConfig = Field(
        default_factory=SyntheticTicketFactoryConfig,
        description="Configuration for the SyntheticTicketFactory.",
    )
    randomized_scenario: RandomizedScenarioConfig = Field(
        default_factory=RandomizedScenarioConfig,
        description=(
            "When enabled, the runner replaces the static scenario "
            "builder with a per-episode random sampler that draws fresh "
            "technicians, machines and product routes from the "
            "configured pools.  Used to train policies that generalise "
            "across factory layouts."
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["KATAConfig"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load from: init kwargs > env vars > JSON file (if it exists) > defaults.

        The JSON file path is resolved dynamically from the ``KATA_CONF_PATH``
        environment variable so it can be changed at runtime without re-importing
        the module.
        """
        json_file = os.getenv("KATA_CONF_PATH", _DEFAULT_CONFIG_FILE)
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(cls, json_file=json_file),
        )


@lru_cache
def get_config() -> KATAConfig:
    """Return the singleton ``KATAConfig`` instance (cached after first call)."""
    return KATAConfig()
