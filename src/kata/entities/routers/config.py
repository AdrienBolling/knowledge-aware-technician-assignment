"""Pydantic configuration models for Router entities."""

from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    """Configuration for a Router."""

    name: str = Field(
        default="router_0",
        description="Human-readable name for the router.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_router = RouterConfig()

# Repository of named default router configurations
router_config_registry: dict[str, RouterConfig] = {
    "default": default_router,
}
