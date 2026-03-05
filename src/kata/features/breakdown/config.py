"""Pydantic configuration models for breakdown processes."""

from pydantic import BaseModel, Field


class SimpleBreakdownConfig(BaseModel):
    """Configuration for a SimpleBreakdownProcess."""

    failure_prob_working: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Probability of failure per time step while working.",
    )
    failure_prob_idle: float = Field(
        default=0.0001,
        ge=0.0,
        le=1.0,
        description="Probability of failure per time step while idle.",
    )


class WeibullBreakdownConfig(BaseModel):
    """Configuration for a WeibullBreakdownProcess."""

    shape: float = Field(
        default=2.0,
        gt=0.0,
        description="Weibull shape parameter (k).",
    )
    scale: float = Field(
        default=1000.0,
        gt=0.0,
        description="Weibull scale parameter (lambda).",
    )
    dt: int = Field(
        default=1,
        gt=0,
        description="Time step size.",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_simple_breakdown = SimpleBreakdownConfig()
default_weibull_breakdown = WeibullBreakdownConfig()

# Repository of named default configurations
breakdown_config_registry: dict[str, SimpleBreakdownConfig | WeibullBreakdownConfig] = {
    "default_simple": default_simple_breakdown,
    "default_weibull": default_weibull_breakdown,
    "high_reliability_simple": SimpleBreakdownConfig(
        failure_prob_working=0.0001,
        failure_prob_idle=0.00001,
    ),
    "low_reliability_simple": SimpleBreakdownConfig(
        failure_prob_working=0.01,
        failure_prob_idle=0.001,
    ),
    "early_life_weibull": WeibullBreakdownConfig(shape=0.5, scale=500.0),
    "wear_out_weibull": WeibullBreakdownConfig(shape=3.5, scale=800.0),
}
