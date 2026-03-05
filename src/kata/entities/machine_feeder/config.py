"""Pydantic configuration models for MachineFeeder entities."""

from pydantic import BaseModel, Field


class MachineFeederConfig(BaseModel):
    """Configuration for a MachineFeeder."""

    name: str = Field(
        default="feeder_0",
        description="Human-readable name for the feeder.",
    )
    machine_type: str = Field(
        default="generic",
        description="The type of machines this feeder distributes products to.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_feeder = MachineFeederConfig()

# Repository of named default feeder configurations
machine_feeder_config_registry: dict[str, MachineFeederConfig] = {
    "default": default_feeder,
}
