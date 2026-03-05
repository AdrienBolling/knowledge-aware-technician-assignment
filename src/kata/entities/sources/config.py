"""Pydantic configuration models for Source entities."""

from typing import Optional

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """Configuration for a Source."""

    name: str = Field(
        default="source",
        description="Human-readable name for the source.",
    )
    interarrival_time: float = Field(
        default=10.0,
        gt=0.0,
        description="Average time between successive product arrivals.",
    )
    route: list[str] = Field(
        default_factory=list,
        description="Ordered list of machine type names that products must visit.",
    )
    max_products: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of products to generate (None = unlimited).",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_source = SourceConfig()

high_throughput_source = SourceConfig(
    name="high_throughput_source",
    interarrival_time=2.0,
    route=["assembly", "cnc"],
)

limited_source = SourceConfig(
    name="limited_source",
    interarrival_time=10.0,
    route=["assembly"],
    max_products=100,
)

# Repository of named default source configurations
source_config_registry: dict[str, SourceConfig] = {
    "default": default_source,
    "high_throughput": high_throughput_source,
    "limited": limited_source,
}
