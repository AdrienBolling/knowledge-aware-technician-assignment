"""Tests for template-based machine and technician factories."""

from __future__ import annotations

import pytest

from kata.entities.machines.config import MachineConfig
from kata.entities.technicians.config import TechnicianConfig
from kata.EntityFactories import (
    create_machine_config,
    create_machines_dict,
    create_technician_config,
    create_technicians_dict,
    list_machine_templates,
    list_technician_templates,
    register_machine_template,
    register_technician_template,
)


class TestMachineTemplates:
    def test_packaged_templates_are_present(self):
        names = set(list_machine_templates())
        # The defaults shipped with the project must remain available.
        for required in {
            "generic",
            "drill",
            "paint",
            "cnc_weibull",
            "assembly_mixed",
            "assembly_robot",
            "conveyor",
        }:
            assert required in names, f"missing template {required}"

    def test_create_returns_machine_config(self):
        cfg = create_machine_config("cnc_weibull")
        assert isinstance(cfg, MachineConfig)
        assert cfg.machine_type == "CNC"
        assert cfg.process_time == 200
        assert "spindle_0" in cfg.components
        assert "coolant_pump_0" in cfg.components
        assert cfg.components["spindle_0"].breakdown_model == "weibull"

    def test_create_with_overrides(self):
        cfg = create_machine_config("cnc_weibull", process_time=300)
        assert cfg.process_time == 300
        # Other fields are inherited from the template
        assert cfg.machine_type == "CNC"

    def test_create_unknown_template_raises(self):
        with pytest.raises(KeyError):
            create_machine_config("does_not_exist")

    def test_register_runtime_template(self):
        register_machine_template(
            "tmp_test_machine_template",
            {"machine_type": "Test", "process_time": 42, "dt": 1},
        )
        assert "tmp_test_machine_template" in list_machine_templates()
        cfg = create_machine_config("tmp_test_machine_template")
        assert cfg.machine_type == "Test"
        assert cfg.process_time == 42

    def test_create_machines_dict_pairs(self):
        machines = create_machines_dict(
            [("cnc_weibull", "cnc_a"), ("assembly_mixed", "asm_a")]
        )
        assert set(machines) == {"cnc_a", "asm_a"}
        assert all(isinstance(m, MachineConfig) for m in machines.values())
        assert machines["cnc_a"].machine_type == "CNC"

    def test_create_machines_dict_mapping(self):
        machines = create_machines_dict(
            {"drill_1": "drill", "drill_2": "drill", "paint_1": "paint"}
        )
        assert set(machines) == {"drill_1", "drill_2", "paint_1"}
        assert machines["drill_1"].machine_type == "Drill"
        assert machines["paint_1"].machine_type == "Paint"


class TestTechnicianTemplates:
    def test_packaged_profiles_are_present(self):
        names = set(list_technician_templates())
        for required in {"default", "expert", "junior", "senior", "generalist"}:
            assert required in names

    def test_create_returns_technician_config(self):
        cfg = create_technician_config("expert")
        assert isinstance(cfg, TechnicianConfig)
        assert cfg.fatigue_lambda == pytest.approx(0.005)
        assert cfg.knowledge_learning_rate == pytest.approx(0.15)

    def test_create_with_overrides(self):
        cfg = create_technician_config("junior", name="alice")
        assert cfg.name == "alice"
        assert cfg.fatigue_lambda == pytest.approx(0.02)

    def test_create_unknown_template_raises(self):
        with pytest.raises(KeyError):
            create_technician_config("nope")

    def test_register_runtime_template(self):
        register_technician_template(
            "tmp_test_tech_template",
            {
                "name": "tmp",
                "fatigue_lambda": 0.001,
                "fatigue_mu": 0.001,
                "knowledge_k_shape": [4, 4],
                "knowledge_propagation_sigma": 0.5,
                "knowledge_transmission_factor": 0.5,
                "knowledge_learning_rate": 0.1,
            },
        )
        cfg = create_technician_config("tmp_test_tech_template")
        assert cfg.knowledge_k_shape == (4, 4)

    def test_create_technicians_dict_sets_instance_name(self):
        techs = create_technicians_dict(
            [("expert", "expert_1"), ("junior", "junior_1")]
        )
        assert set(techs) == {"expert_1", "junior_1"}
        # The factory should override the template's name with the
        # instance ID so distinct techs are distinguishable.
        assert techs["expert_1"].name == "expert_1"
        assert techs["junior_1"].name == "junior_1"


class TestConfigTemplateExpansion:
    """The pydantic configs accept a `template` key that pulls in template defaults."""

    def test_machine_config_template_field(self):
        cfg = MachineConfig.model_validate({"template": "cnc_weibull"})
        assert cfg.machine_type == "CNC"
        assert cfg.process_time == 200

    def test_machine_config_template_with_override(self):
        cfg = MachineConfig.model_validate(
            {"template": "cnc_weibull", "process_time": 250}
        )
        assert cfg.process_time == 250
        # template-provided components are still present
        assert "spindle_0" in cfg.components

    def test_technician_config_template_field(self):
        cfg = TechnicianConfig.model_validate({"template": "expert"})
        assert cfg.fatigue_lambda == pytest.approx(0.005)
        assert cfg.knowledge_propagation_sigma == pytest.approx(1.5)

    def test_technician_config_template_with_override(self):
        cfg = TechnicianConfig.model_validate(
            {"template": "junior", "name": "trainee_42"}
        )
        assert cfg.name == "trainee_42"
        # The override does not clobber template defaults
        assert cfg.fatigue_lambda == pytest.approx(0.02)

    def test_unknown_template_raises_validation(self):
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError or KeyError
            MachineConfig.model_validate({"template": "no_such_template"})
