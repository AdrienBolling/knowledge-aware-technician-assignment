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

from pydantic import BaseModel, Field
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


class DisruptionConfig(BaseModel):
    """Configuration for stochastic technician disruptions (e.g. sick leave)."""

    interrupt_on_disrupt: bool = Field(
        default=True,
        description="Whether an ongoing repair is pre-empted when a disruption starts.",
    )
    dis_dict: dict[str, dict[str, float]] = Field(
        default={"sick_leave": {"mu": 480.0, "sig": 120.0, "prob": 1.0}},
        description=(
            "Mapping of disruption type -> parameters. "
            "Each entry must have 'mu' (mean duration), 'sig' (std dev), "
            "and 'prob' (relative probability of that disruption type)."
        ),
    )


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
    observation_representation: Literal["structured", "tokens", "token_ids"] = Field(
        default="structured",
        description=(
            "Observation payload format. 'structured' keeps numeric fields, "
            "'tokens' returns fixed-size textual token tuples, "
            "'token_ids' returns integer ID sequences for Transformer input."
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
