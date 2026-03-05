"""Pydantic configuration models for Sink entities."""

from pydantic import BaseModel, Field


class SinkConfig(BaseModel):
    """Configuration for a Sink."""

    name: str = Field(
        default="sink",
        description="Human-readable name for the sink.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_sink = SinkConfig()

# Repository of named default sink configurations
sink_config_registry: dict[str, SinkConfig] = {
    "default": default_sink,
}
