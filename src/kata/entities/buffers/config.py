"""Pydantic configuration models for Buffer entities."""

from pydantic import BaseModel, Field


class BufferConfig(BaseModel):
    """Configuration for a Buffer."""

    name: str = Field(
        default="buffer",
        description="Human-readable name for the buffer.",
    )
    capacity: float = Field(
        default=float("inf"),
        gt=0.0,
        description="Maximum number of items the buffer can hold. Use `inf` for unlimited.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_buffer = BufferConfig()

finite_buffer = BufferConfig(name="finite_buffer", capacity=50.0)
large_buffer = BufferConfig(name="large_buffer", capacity=500.0)

# Repository of named default buffer configurations
buffer_config_registry: dict[str, BufferConfig] = {
    "default": default_buffer,
    "finite": finite_buffer,
    "large": large_buffer,
}
