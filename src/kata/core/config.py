"""
Centralised settings for KATA.

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

# ---------------------------------------------------------------------------
# Re-export entity-level configs so existing imports keep working.
# ---------------------------------------------------------------------------
from kata.entities.technicians.config import TechnicianConfig  # noqa: F401
from kata.entities.components.config import ComponentConfig  # noqa: F401
from kata.entities.machines.config import MachineConfig  # noqa: F401
from kata.entities.buffers.config import BufferConfig  # noqa: F401
from kata.entities.sources.config import SourceConfig  # noqa: F401
from kata.entities.sinks.config import SinkConfig  # noqa: F401
from kata.entities.routers.config import RouterConfig  # noqa: F401
from kata.entities.machine_feeder.config import MachineFeederConfig  # noqa: F401
from kata.entities.tech_dispatcher.config import TechDispatcherConfig  # noqa: F401
from kata.entities.production_line.config import ProductionLineConfig  # noqa: F401
from kata.EntityFactories.config import SyntheticTicketFactoryConfig  # noqa: F401
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
    """Global switches that control how repair time is computed."""

    knowledge_enabled: bool = Field(
        default=True,
        description="Apply knowledge multiplier to base repair time when True.",
    )
    fatigue_enabled: bool = Field(
        default=True,
        description="Apply fatigue multiplier to base repair time when True.",
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
        description="Fatigue model to use: 'exponential' or 'linear'.",
    )
    fatigue_alpha: float = Field(
        default=0.5,
        gt=0.0,
        description="Alpha parameter for the exponential fatigue model.",
    )


class SimEnvConfig(BaseModel):
    """Top-level configuration for the SimPy simulation environment."""

    disruptions: DisruptionConfig = Field(default_factory=DisruptionConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    technicians: GlobalTechniciansConfig = Field(default_factory=GlobalTechniciansConfig)


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
    """Composable reward settings for the Gym environment."""

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
    reward: GymRewardConfig = Field(
        default_factory=GymRewardConfig,
        description="Composable reward settings with configurable sub-components.",
    )
    observation_representation: Literal["structured", "tokens"] = Field(
        default="structured",
        description=(
            "Observation payload format. 'structured' keeps numeric fields, "
            "'tokens' returns fixed-size textual token tuples."
        ),
    )
    observation_mode: Literal["ticket_only", "broken_machine", "factory_level"] = Field(
        default="ticket_only",
        description="Level of context to include in token observations.",
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
    include_fatigue_in_observation: bool = Field(
        default=True,
        description="Include technicians' fatigue values in observation vectors.",
    )
    include_queue_size_in_observation: bool = Field(
        default=True,
        description="Include pending-repair queue size in observations.",
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
    """
    Centralised settings for KATA.

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
