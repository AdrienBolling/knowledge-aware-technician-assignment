"""Pydantic configuration models for machine components."""

from pydantic import BaseModel, Field

from kata.features.breakdown.config import SimpleBreakdownConfig, WeibullBreakdownConfig


class ComponentConfig(BaseModel):
    """Configuration for a MachineComponent."""

    component_id: str = Field(
        default="component_0",
        description="Unique identifier for the component.",
    )
    component_type: str = Field(
        default="generic",
        description="Type/category of the component (e.g., motor, bearing, sensor).",
    )
    base_repair_time: float = Field(
        default=10.0,
        gt=0.0,
        description="Base time required to repair this component.",
    )
    idle_degradation_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Factor to reduce degradation when machine is idle.",
    )
    breakdown_model: str = Field(
        default="simple",
        description="Type of breakdown model: 'simple' or 'weibull'.",
    )
    simple_breakdown: SimpleBreakdownConfig = Field(
        default_factory=SimpleBreakdownConfig,
        description="Config for simple breakdown model (used when breakdown_model='simple').",
    )
    weibull_breakdown: WeibullBreakdownConfig = Field(
        default_factory=WeibullBreakdownConfig,
        description="Config for Weibull breakdown model (used when breakdown_model='weibull').",
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_component = ComponentConfig()

motor_component = ComponentConfig(
    component_id="motor_0",
    component_type="motor",
    base_repair_time=50.0,
    idle_degradation_factor=0.05,
    breakdown_model="weibull",
    weibull_breakdown=WeibullBreakdownConfig(shape=2.5, scale=800.0),
)

bearing_component = ComponentConfig(
    component_id="bearing_0",
    component_type="bearing",
    base_repair_time=30.0,
    idle_degradation_factor=0.1,
    breakdown_model="simple",
    simple_breakdown=SimpleBreakdownConfig(
        failure_prob_working=0.002,
        failure_prob_idle=0.0002,
    ),
)

sensor_component = ComponentConfig(
    component_id="sensor_0",
    component_type="sensor",
    base_repair_time=15.0,
    idle_degradation_factor=0.2,
    breakdown_model="simple",
    simple_breakdown=SimpleBreakdownConfig(
        failure_prob_working=0.0005,
        failure_prob_idle=0.0001,
    ),
)

pump_component = ComponentConfig(
    component_id="pump_0",
    component_type="pump",
    base_repair_time=75.0,
    idle_degradation_factor=0.05,
    breakdown_model="weibull",
    weibull_breakdown=WeibullBreakdownConfig(shape=3.0, scale=1200.0),
)

# Repository of named default component configurations
component_config_registry: dict[str, ComponentConfig] = {
    "default": default_component,
    "motor": motor_component,
    "bearing": bearing_component,
    "sensor": sensor_component,
    "pump": pump_component,
}
