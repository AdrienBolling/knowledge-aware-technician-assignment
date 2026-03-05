"""Pydantic configuration models for TechDispatcher entities."""

from pydantic import BaseModel, Field


class TechDispatcherConfig(BaseModel):
    """Configuration for a GymTechDispatcher."""

    repair_queue_capacity: int = Field(
        default=9999,
        gt=0,
        description="Maximum number of pending repair requests the queue can hold.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_tech_dispatcher = TechDispatcherConfig()

large_queue_dispatcher = TechDispatcherConfig(repair_queue_capacity=99999)

# Repository of named default dispatcher configurations
tech_dispatcher_config_registry: dict[str, TechDispatcherConfig] = {
    "default": default_tech_dispatcher,
    "large_queue": large_queue_dispatcher,
}
