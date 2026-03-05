"""Pydantic configuration models for ProductionLine entities."""

from pydantic import BaseModel, Field

from kata.entities.machines.config import MachineConfig
from kata.entities.buffers.config import BufferConfig
from kata.entities.sources.config import SourceConfig
from kata.entities.sinks.config import SinkConfig


class ProductionLineConfig(BaseModel):
    """Configuration for a ProductionLine."""

    name: str = Field(
        default="production_line_0",
        description="Human-readable name for the production line.",
    )
    machines: dict[str, MachineConfig] = Field(
        default_factory=dict,
        description="Mapping of machine ID to MachineConfig for all machines on the line.",
    )
    buffers: dict[str, BufferConfig] = Field(
        default_factory=dict,
        description="Mapping of buffer ID to BufferConfig for all buffers on the line.",
    )
    sources: dict[str, SourceConfig] = Field(
        default_factory=dict,
        description="Mapping of source ID to SourceConfig for all sources on the line.",
    )
    sinks: dict[str, SinkConfig] = Field(
        default_factory=dict,
        description="Mapping of sink ID to SinkConfig for all sinks on the line.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_production_line = ProductionLineConfig(
    name="default_line_0",
    machines={"machine_0": MachineConfig()},
    buffers={
        "input_buffer_0": BufferConfig(name="input_buffer_0"),
        "output_buffer_0": BufferConfig(name="output_buffer_0"),
    },
    sources={"source_0": SourceConfig()},
    sinks={"sink_0": SinkConfig()},
)

# Repository of named default production line configurations
production_line_config_registry: dict[str, ProductionLineConfig] = {
    "default": default_production_line,
}
