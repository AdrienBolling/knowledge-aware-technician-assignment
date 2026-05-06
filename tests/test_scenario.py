"""Tests for ScenarioBuilder config-driven factory construction."""

import os

import simpy

from kata.core.config import (
    GymEnvConfig,
    KATAConfig,
    MachineConfig,
    ProductConfig,
    TechnicianConfig,
)
from kata.entities.components.config import ComponentConfig
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder


def _minimal_config(**overrides) -> KATAConfig:
    # Point to a non-existent JSON so we only use the explicit kwargs
    old = os.environ.get("KATA_CONF_PATH")
    os.environ["KATA_CONF_PATH"] = "/dev/null/no_such_file.json"
    try:
        defaults = {
            "technicians": {"t0": TechnicianConfig(name="t0")},
            "machines": {"m0": MachineConfig(machine_type="generic", process_time=10)},
            "products": {
                "p0": ProductConfig(product_type="generic", route=["generic"])
            },
        }
        defaults.update(overrides)
        return KATAConfig(**defaults)
    finally:
        if old is None:
            os.environ.pop("KATA_CONF_PATH", None)
        else:
            os.environ["KATA_CONF_PATH"] = old


class TestScenarioBuilder:
    def test_build_returns_env_and_dispatcher(self):
        cfg = _minimal_config()
        builder = ScenarioBuilder(cfg)
        env, dispatcher = builder.build()

        assert isinstance(env, simpy.Environment)
        assert dispatcher is not None
        assert len(dispatcher.techs) == 1

    def test_machines_are_created(self):
        cfg = _minimal_config(
            machines={
                "m0": MachineConfig(machine_type="A", process_time=10),
                "m1": MachineConfig(machine_type="B", process_time=20),
            },
            products={"p0": ProductConfig(route=["A", "B"])},
        )
        builder = ScenarioBuilder(cfg)
        env, dispatcher = builder.build()
        # Machines should be exposed on dispatcher
        assert hasattr(dispatcher, "machines")
        assert len(dispatcher.machines) == 2

    def test_complex_machines_from_config(self):
        cfg = _minimal_config(
            machines={
                "cnc": MachineConfig(
                    machine_type="CNC",
                    process_time=50,
                    components={
                        "motor": ComponentConfig(
                            component_id="motor_0",
                            component_type="motor",
                            base_repair_time=40.0,
                        ),
                    },
                ),
            },
            products={"p0": ProductConfig(route=["CNC"])},
        )
        builder = ScenarioBuilder(cfg)
        env, dispatcher = builder.build()
        assert len(dispatcher.machines) == 1

    def test_multiple_technicians(self):
        cfg = _minimal_config(
            technicians={
                "expert": TechnicianConfig(name="expert", fatigue_lambda=0.005),
                "junior": TechnicianConfig(name="junior", fatigue_lambda=0.02),
            },
        )
        builder = ScenarioBuilder(cfg)
        env, dispatcher = builder.build()
        assert len(dispatcher.techs) == 2
        names = {t.name for t in dispatcher.techs}
        assert "expert" in names
        assert "junior" in names

    def test_simulation_runs_without_error(self):
        cfg = _minimal_config()
        builder = ScenarioBuilder(cfg)
        env, dispatcher = builder.build()
        env.run(until=100)  # should not raise


class TestScenarioWithKataEnv:
    def test_scenario_factory_integration(self):
        cfg = _minimal_config(
            gym=GymEnvConfig(max_episode_steps=10, max_sim_time=100.0),
        )

        def factory():
            builder = ScenarioBuilder(cfg)
            return builder.build()

        kata_env = KataEnv(scenario_factory=factory)
        obs, info = kata_env.reset()
        assert "sim_time" in obs or "tokens" in obs
