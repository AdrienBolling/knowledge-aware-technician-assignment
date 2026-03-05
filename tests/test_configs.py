"""
Tests for all Pydantic config models created alongside entity classes.

These tests validate:
- Config models instantiate with defaults
- Config models accept valid overrides
- Config model validation rejects bad values
- Default instances and registry dicts exist and are well-formed
- Centralised KATAConfig loads defaults and re-exports entity configs
- KATAConfig loads settings from a JSON file
"""

import json
import os
import tempfile

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Breakdown configs
# ---------------------------------------------------------------------------
from kata.features.breakdown.config import (
    SimpleBreakdownConfig,
    WeibullBreakdownConfig,
    breakdown_config_registry,
    default_simple_breakdown,
    default_weibull_breakdown,
)


class TestSimpleBreakdownConfig:
    def test_defaults(self):
        cfg = SimpleBreakdownConfig()
        assert cfg.failure_prob_working == 0.001
        assert cfg.failure_prob_idle == 0.0001

    def test_valid_override(self):
        cfg = SimpleBreakdownConfig(failure_prob_working=0.5, failure_prob_idle=0.1)
        assert cfg.failure_prob_working == 0.5
        assert cfg.failure_prob_idle == 0.1

    def test_invalid_probability(self):
        with pytest.raises(ValidationError):
            SimpleBreakdownConfig(failure_prob_working=1.5)


class TestWeibullBreakdownConfig:
    def test_defaults(self):
        cfg = WeibullBreakdownConfig()
        assert cfg.shape == 2.0
        assert cfg.scale == 1000.0
        assert cfg.dt == 1

    def test_invalid_shape(self):
        with pytest.raises(ValidationError):
            WeibullBreakdownConfig(shape=0.0)


class TestBreakdownRegistry:
    def test_registry_has_defaults(self):
        assert "default_simple" in breakdown_config_registry
        assert "default_weibull" in breakdown_config_registry

    def test_registry_instances(self):
        assert isinstance(breakdown_config_registry["default_simple"], SimpleBreakdownConfig)
        assert isinstance(breakdown_config_registry["default_weibull"], WeibullBreakdownConfig)

    def test_default_instances_exported(self):
        assert isinstance(default_simple_breakdown, SimpleBreakdownConfig)
        assert isinstance(default_weibull_breakdown, WeibullBreakdownConfig)


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------
from kata.entities.components.config import (
    ComponentConfig,
    component_config_registry,
    default_component,
    motor_component,
    bearing_component,
)


class TestComponentConfig:
    def test_defaults(self):
        cfg = ComponentConfig()
        assert cfg.component_id == "component_0"
        assert cfg.component_type == "generic"
        assert cfg.base_repair_time == 10.0
        assert cfg.idle_degradation_factor == 0.1
        assert cfg.breakdown_model == "simple"

    def test_nested_simple_breakdown_defaults(self):
        cfg = ComponentConfig()
        assert cfg.simple_breakdown.failure_prob_working == 0.001

    def test_invalid_idle_factor(self):
        with pytest.raises(ValidationError):
            ComponentConfig(idle_degradation_factor=1.5)


class TestComponentRegistry:
    def test_has_expected_keys(self):
        for key in ("default", "motor", "bearing", "sensor", "pump"):
            assert key in component_config_registry

    def test_motor_component_type(self):
        assert motor_component.component_type == "motor"
        assert motor_component.breakdown_model == "weibull"

    def test_bearing_component_type(self):
        assert bearing_component.component_type == "bearing"


# ---------------------------------------------------------------------------
# Machine configs
# ---------------------------------------------------------------------------
from kata.entities.machines.config import (
    MachineConfig,
    machine_config_registry,
    default_machine,
    assembly_machine,
)


class TestMachineConfig:
    def test_defaults(self):
        cfg = MachineConfig()
        assert cfg.machine_type == "generic"
        assert cfg.process_time == 100
        assert cfg.dt == 1
        assert cfg.components == {}

    def test_invalid_process_time(self):
        with pytest.raises(ValidationError):
            MachineConfig(process_time=0)

    def test_with_components(self):
        cfg = MachineConfig(
            machine_type="test",
            components={"c1": ComponentConfig(component_id="c1", component_type="motor")},
        )
        assert "c1" in cfg.components
        assert cfg.components["c1"].component_type == "motor"


class TestMachineRegistry:
    def test_has_expected_keys(self):
        for key in ("default", "assembly", "cnc"):
            assert key in machine_config_registry

    def test_assembly_has_components(self):
        assert len(assembly_machine.components) > 0


# ---------------------------------------------------------------------------
# Buffer configs
# ---------------------------------------------------------------------------
from kata.entities.buffers.config import (
    BufferConfig,
    buffer_config_registry,
    default_buffer,
)


class TestBufferConfig:
    def test_defaults(self):
        cfg = BufferConfig()
        assert cfg.name == "buffer"
        assert cfg.capacity == float("inf")

    def test_finite_capacity(self):
        cfg = BufferConfig(capacity=50.0)
        assert cfg.capacity == 50.0

    def test_invalid_capacity(self):
        with pytest.raises(ValidationError):
            BufferConfig(capacity=0.0)


class TestBufferRegistry:
    def test_has_default(self):
        assert "default" in buffer_config_registry
        assert isinstance(default_buffer, BufferConfig)


# ---------------------------------------------------------------------------
# Source configs
# ---------------------------------------------------------------------------
from kata.entities.sources.config import (
    SourceConfig,
    source_config_registry,
    default_source,
)


class TestSourceConfig:
    def test_defaults(self):
        cfg = SourceConfig()
        assert cfg.name == "source"
        assert cfg.interarrival_time == 10.0
        assert cfg.route == []
        assert cfg.max_products is None

    def test_with_route(self):
        cfg = SourceConfig(route=["assembly", "cnc"])
        assert cfg.route == ["assembly", "cnc"]

    def test_invalid_max_products(self):
        with pytest.raises(ValidationError):
            SourceConfig(max_products=0)


class TestSourceRegistry:
    def test_has_expected_keys(self):
        for key in ("default", "high_throughput", "limited"):
            assert key in source_config_registry


# ---------------------------------------------------------------------------
# Sink configs
# ---------------------------------------------------------------------------
from kata.entities.sinks.config import (
    SinkConfig,
    sink_config_registry,
    default_sink,
)


class TestSinkConfig:
    def test_defaults(self):
        cfg = SinkConfig()
        assert cfg.name == "sink"

    def test_registry(self):
        assert "default" in sink_config_registry
        assert isinstance(default_sink, SinkConfig)


# ---------------------------------------------------------------------------
# Router configs
# ---------------------------------------------------------------------------
from kata.entities.routers.config import (
    RouterConfig,
    router_config_registry,
    default_router,
)


class TestRouterConfig:
    def test_defaults(self):
        cfg = RouterConfig()
        assert cfg.name == "router"

    def test_registry(self):
        assert "default" in router_config_registry
        assert isinstance(default_router, RouterConfig)


# ---------------------------------------------------------------------------
# MachineFeeder configs
# ---------------------------------------------------------------------------
from kata.entities.machine_feeder.config import (
    MachineFeederConfig,
    machine_feeder_config_registry,
    default_feeder,
)


class TestMachineFeederConfig:
    def test_defaults(self):
        cfg = MachineFeederConfig()
        assert cfg.name == "feeder"
        assert cfg.machine_type == "generic"

    def test_registry(self):
        assert "default" in machine_feeder_config_registry
        assert isinstance(default_feeder, MachineFeederConfig)


# ---------------------------------------------------------------------------
# Technician configs
# ---------------------------------------------------------------------------
from kata.entities.technicians.config import (
    TechnicianConfig,
    technician_config_registry,
    default_technician,
    expert_technician,
    junior_technician,
)


class TestTechnicianConfig:
    def test_defaults(self):
        cfg = TechnicianConfig()
        assert cfg.name == "technician"
        assert cfg.fatigue_lambda == 0.01
        assert cfg.fatigue_mu == 0.05
        assert cfg.knowledge_k_shape == (10, 10)
        assert 0.0 < cfg.knowledge_propagation_sigma
        assert 0.0 < cfg.knowledge_transmission_factor <= 1.0
        assert 0.0 < cfg.knowledge_learning_rate <= 1.0

    def test_invalid_fatigue_lambda(self):
        with pytest.raises(ValidationError):
            TechnicianConfig(fatigue_lambda=0.0)

    def test_invalid_transmission_factor(self):
        with pytest.raises(ValidationError):
            TechnicianConfig(knowledge_transmission_factor=1.5)

    def test_expert_config(self):
        assert expert_technician.fatigue_lambda < default_technician.fatigue_lambda


class TestTechnicianRegistry:
    def test_has_expected_keys(self):
        for key in ("default", "expert", "junior"):
            assert key in technician_config_registry

    def test_registry_instances(self):
        for cfg in technician_config_registry.values():
            assert isinstance(cfg, TechnicianConfig)


# ---------------------------------------------------------------------------
# TechDispatcher configs
# ---------------------------------------------------------------------------
from kata.entities.tech_dispatcher.config import (
    TechDispatcherConfig,
    tech_dispatcher_config_registry,
    default_tech_dispatcher,
)


class TestTechDispatcherConfig:
    def test_defaults(self):
        cfg = TechDispatcherConfig()
        assert cfg.repair_queue_capacity == 9999

    def test_invalid_capacity(self):
        with pytest.raises(ValidationError):
            TechDispatcherConfig(repair_queue_capacity=0)

    def test_registry(self):
        assert "default" in tech_dispatcher_config_registry
        assert isinstance(default_tech_dispatcher, TechDispatcherConfig)


# ---------------------------------------------------------------------------
# ProductionLine configs
# ---------------------------------------------------------------------------
from kata.entities.production_line.config import (
    ProductionLineConfig,
    production_line_config_registry,
    default_production_line,
)


class TestProductionLineConfig:
    def test_defaults(self):
        cfg = ProductionLineConfig()
        assert cfg.name == "production_line"
        assert cfg.machines == {}
        assert cfg.buffers == {}
        assert cfg.sources == {}
        assert cfg.sinks == {}

    def test_default_instance_has_entities(self):
        assert len(default_production_line.machines) > 0
        assert len(default_production_line.buffers) > 0

    def test_registry(self):
        assert "default" in production_line_config_registry


# ---------------------------------------------------------------------------
# SyntheticTicketFactory configs
# ---------------------------------------------------------------------------
from kata.EntityFactories.config import (
    SyntheticTicketFactoryConfig,
    synthetic_ticket_factory_config_registry,
    default_ticket_factory,
    priority_aware_ticket_factory,
)


class TestSyntheticTicketFactoryConfig:
    def test_defaults(self):
        cfg = SyntheticTicketFactoryConfig()
        assert cfg.priority_rules == {}
        assert cfg.add_randomness is False
        assert cfg.random_priority_variance == 0
        assert cfg.ticket_id_counter == 1

    def test_priority_rules(self):
        assert "motor" in priority_aware_ticket_factory.priority_rules
        assert priority_aware_ticket_factory.priority_rules["motor"] == 10

    def test_invalid_variance(self):
        with pytest.raises(ValidationError):
            SyntheticTicketFactoryConfig(random_priority_variance=-1)

    def test_registry(self):
        for key in ("default", "priority_aware", "random"):
            assert key in synthetic_ticket_factory_config_registry


# ---------------------------------------------------------------------------
# Centralised KATAConfig (core/config.py)
# ---------------------------------------------------------------------------
from kata.core.config import (
    KATAConfig,
    get_config,
    DisruptionConfig,
    RepairConfig,
    GlobalTechniciansConfig,
    SimEnvConfig,
    GymEnvConfig,
    ProductConfig,
    # Re-exported entity configs
    TechnicianConfig as CoreTechnicianConfig,
    ComponentConfig as CoreComponentConfig,
    MachineConfig as CoreMachineConfig,
)


class TestCoreConfigReExports:
    """core/config.py must re-export entity configs for backward compatibility."""

    def test_technician_config_is_same_class(self):
        assert CoreTechnicianConfig is TechnicianConfig

    def test_component_config_is_same_class(self):
        assert CoreComponentConfig is ComponentConfig

    def test_machine_config_is_same_class(self):
        assert CoreMachineConfig is MachineConfig


class TestKATAConfigDefaults:
    def test_loads_with_defaults(self):
        cfg = KATAConfig()
        assert cfg.sim.technicians.travel_time == 10
        assert cfg.sim.repair.knowledge_enabled is True
        assert cfg.sim.repair.fatigue_enabled is True
        assert cfg.sim.disruptions.interrupt_on_disrupt is True
        assert cfg.gym.max_episode_steps == 10_000

    def test_technicians_registry(self):
        cfg = KATAConfig()
        assert len(cfg.technicians) >= 1
        for v in cfg.technicians.values():
            assert isinstance(v, TechnicianConfig)

    def test_machines_registry(self):
        cfg = KATAConfig()
        assert len(cfg.machines) >= 1
        for v in cfg.machines.values():
            assert isinstance(v, MachineConfig)

    def test_disruption_config(self):
        cfg = KATAConfig()
        assert "sick_leave" in cfg.sim.disruptions.dis_dict


class TestKATAConfigJsonLoading:
    """KATAConfig must be loadable from a JSON file."""

    def test_json_override(self):
        payload = {
            "sim": {
                "technicians": {
                    "travel_time": 25,
                    "fatigue_model": "linear",
                    "fatigue_alpha": 0.2,
                },
                "repair": {"knowledge_enabled": False},
            },
            "technicians": {
                "tech_a": {"name": "senior_tech", "fatigue_lambda": 0.003},
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(payload, f)
            tmp_path = f.name

        try:
            old_env = os.environ.get("KATA_CONF_PATH")
            os.environ["KATA_CONF_PATH"] = tmp_path

            # Instantiate a fresh KATAConfig (not the cached singleton)
            cfg = KATAConfig(_env_file=None)  # type: ignore[call-arg]
            assert cfg.sim.technicians.travel_time == 25
            assert cfg.sim.technicians.fatigue_model == "linear"
            assert cfg.sim.repair.knowledge_enabled is False
            assert cfg.technicians["tech_a"].name == "senior_tech"
        finally:
            os.unlink(tmp_path)
            if old_env is None:
                os.environ.pop("KATA_CONF_PATH", None)
            else:
                os.environ["KATA_CONF_PATH"] = old_env

    def test_get_config_returns_singleton(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2


class TestSimEnvSubConfigs:
    def test_disruption_config_defaults(self):
        cfg = DisruptionConfig()
        assert cfg.interrupt_on_disrupt is True
        assert "sick_leave" in cfg.dis_dict

    def test_repair_config_defaults(self):
        cfg = RepairConfig()
        assert cfg.knowledge_enabled is True
        assert cfg.fatigue_enabled is True

    def test_global_technicians_config(self):
        cfg = GlobalTechniciansConfig()
        assert cfg.travel_time == 10
        assert cfg.fatigue_model == "exponential"
        assert cfg.fatigue_alpha > 0

    def test_invalid_fatigue_model_alpha(self):
        with pytest.raises(ValidationError):
            GlobalTechniciansConfig(fatigue_alpha=0.0)

    def test_gym_env_config(self):
        cfg = GymEnvConfig()
        assert cfg.max_episode_steps == 10_000
        assert cfg.max_sim_time == 10_000.0
        assert cfg.invalid_action_mode == "penalize"
        assert cfg.ticket_wait_time_penalty >= 0.0
        assert cfg.include_fatigue_in_observation is True

    def test_gym_env_invalid_wait_penalty(self):
        with pytest.raises(ValidationError):
            GymEnvConfig(ticket_wait_time_penalty=-0.1)

    def test_product_config(self):
        cfg = ProductConfig()
        assert cfg.product_type == "generic"
        assert cfg.route == []
