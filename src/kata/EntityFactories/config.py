"""Pydantic configuration models for EntityFactory classes."""

from typing import Optional

from pydantic import BaseModel, Field


class SyntheticTicketFactoryConfig(BaseModel):
    """Configuration for a SyntheticTicketFactory."""

    priority_rules: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Mapping of component type or failure type to a base priority value. "
            "Example: {\"motor\": 10, \"sensor\": 5}."
        ),
    )
    add_randomness: bool = Field(
        default=False,
        description="Whether to add random variance to generated ticket priorities.",
    )
    random_priority_variance: int = Field(
        default=0,
        ge=0,
        description="+/- variance for priority when add_randomness is enabled.",
    )
    ticket_id_counter: int = Field(
        default=1,
        ge=1,
        description="Starting value for auto-incrementing ticket IDs.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_ticket_factory = SyntheticTicketFactoryConfig()

priority_aware_ticket_factory = SyntheticTicketFactoryConfig(
    priority_rules={
        "motor": 10,
        "pump": 9,
        "bearing": 7,
        "spindle": 8,
        "sensor": 5,
        "general_failure": 6,
    },
)

random_ticket_factory = SyntheticTicketFactoryConfig(
    priority_rules={
        "motor": 10,
        "pump": 9,
        "bearing": 7,
        "sensor": 5,
    },
    add_randomness=True,
    random_priority_variance=2,
)

# Repository of named default ticket factory configurations
synthetic_ticket_factory_config_registry: dict[str, SyntheticTicketFactoryConfig] = {
    "default": default_ticket_factory,
    "priority_aware": priority_aware_ticket_factory,
    "random": random_ticket_factory,
}
