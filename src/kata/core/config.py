"""Pydantic model of the configuration file. Function get_config can be used by any module of the project to get the singleton configuration object."""

import os
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)
from pydantic import BaseModel
from typing import Annotated
from functools import lru_cache


class DisruptionConfig(BaseModel):
    """Configuration model for disruptions in the simulation environment."""

    interupt_on_disrupt: bool = True
    dis_dict: dict[str, dict[str, float]] = {
        "sick_leave": {"mu": 480.0, "sig": 120.0, "prob": 1.0},
    }


class RepairConfig(BaseModel):
    """COnfiguration for repair processes in the simulation environment."""

    knowledge_enabled: bool = (
        True  # is knowledge taken into account for repair time computation
    )
    fatigue_enabled: bool = (
        True  # is fatigue taken into account for repair time computation
    )


class TechnicianConfig(BaseModel): ...


class MachineConfig(BaseModel): ...


class GlobalTechniciansConfig(BaseModel):
    travel_time: int = 10  # travel time between machines in minutes
    fatigue_model: str = "exponential"  # can be either exponential or linear
    fatigue_alpha: float = 0.5  # parameter for exponential fatigue model


class SimEnvConfig(BaseModel):
    """Configuration model for the simulation environment."""

    disruptions: DisruptionConfig = DisruptionConfig()
    repair: RepairConfig = RepairConfig()
    technicians: GlobalTechniciansConfig = GlobalTechniciansConfig()


class GymEnvConfig(BaseModel): ...


class ProductConfig(BaseModel): ...


class KATAConfig(BaseSettings):
    """Pydantic model of the configuration file. The config file must be formatted as a json file."""

    config_file: str = os.getenv("KATA_CONF_PATH", "run_configs/config.json")
    model_config = SettingsConfigDict(json_file=config_file, cli_parse_args=True)

    technicians: dict[str, TechnicianConfig] = {"technician_1": TechnicianConfig()}
    machines: dict[str, MachineConfig] = {"machine_type_1": MachineConfig()}
    products: dict[str, ProductConfig] = {"product_type_1": ProductConfig()}
    sim: SimEnvConfig = SimEnvConfig()
    gym: GymEnvConfig = GymEnvConfig()


@lru_cache
def get_config() -> KATAConfig:
    """Get the singleton configuration object."""
    return KATAConfig()
