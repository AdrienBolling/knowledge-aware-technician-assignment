"""Pydantic configuration models for machines."""

from pydantic import BaseModel, Field

from kata.entities.components.config import ComponentConfig


class MachineConfig(BaseModel):
    """Configuration for a SimPy-based Machine or ComplexMachine."""

    machine_type: str = Field(
        default="generic",
        description="Type/category of the machine.",
    )
    process_time: int = Field(
        default=100,
        gt=0,
        description="Time units required to process one product.",
    )
    dt: int = Field(
        default=1,
        gt=0,
        description="Time step granularity for the breakdown driver.",
    )
    components: dict[str, ComponentConfig] = Field(
        default_factory=dict,
        description=(
            "Mapping of component ID to ComponentConfig. "
            "When non-empty the machine will be created as a ComplexMachine."
        ),
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_machine = MachineConfig()

assembly_machine = MachineConfig(
    machine_type="assembly",
    process_time=120,
    dt=1,
    components={
        "motor_0": ComponentConfig(
            component_id="motor_0",
            component_type="motor",
            base_repair_time=50.0,
        ),
        "bearing_0": ComponentConfig(
            component_id="bearing_0",
            component_type="bearing",
            base_repair_time=30.0,
        ),
    },
)

cnc_machine = MachineConfig(
    machine_type="cnc",
    process_time=200,
    dt=1,
    components={
        "spindle_0": ComponentConfig(
            component_id="spindle_0",
            component_type="spindle",
            base_repair_time=90.0,
        ),
        "coolant_pump_0": ComponentConfig(
            component_id="coolant_pump_0",
            component_type="pump",
            base_repair_time=40.0,
        ),
    },
)

# Repository of named default machine configurations
machine_config_registry: dict[str, MachineConfig] = {
    "default": default_machine,
    "assembly": assembly_machine,
    "cnc": cnc_machine,
}
