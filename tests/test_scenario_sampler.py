"""Tests for the per-episode randomised scenario sampler."""

from __future__ import annotations

import os

import pytest

from kata.core.config import (
    KATAConfig,
    RandomizedScenarioConfig,
)
from kata.EntityFactories import RandomScenarioSampler


def _base_cfg() -> KATAConfig:
    os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"
    return KATAConfig()


class TestRandomScenarioSampler:
    def test_samples_distinct_layouts_across_calls(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True,
            seed=42,
            n_technicians=4,
            technician_templates=["expert", "senior", "generalist", "junior"],
            n_machines_min=8,
            n_machines_max=15,
            machine_templates=[
                "cnc_weibull", "assembly_mixed", "assembly_robot",
                "conveyor", "welder", "inspection",
            ],
            route_min_length=2,
            route_max_length=5,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=42)

        sigs = []
        for _ in range(8):
            cfg = sampler.sample_config()
            sigs.append(
                (
                    tuple(sorted(cfg.machines.keys())),
                    tuple(list(cfg.products.values())[0].route),
                    tuple(t.name for t in cfg.technicians.values()),
                )
            )
        # Some pair must differ — non-deterministic content per episode.
        distinct = {s for s in sigs}
        assert len(distinct) > 1

    def test_action_space_size_stable(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0, n_technicians=4,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        for _ in range(5):
            cfg = sampler.sample_config()
            assert len(cfg.technicians) == 4

    def test_machines_count_within_bounds(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0, n_machines_min=5, n_machines_max=10,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        for _ in range(10):
            cfg = sampler.sample_config()
            assert 5 <= len(cfg.machines) <= 10

    def test_route_only_uses_present_machine_types(self):
        rcfg = RandomizedScenarioConfig(enabled=True, seed=7)
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=7)
        for _ in range(8):
            cfg = sampler.sample_config()
            types_present = {m.machine_type for m in cfg.machines.values()}
            route = list(cfg.products.values())[0].route
            assert set(route).issubset(types_present)
            assert len(route) >= 1

    def test_technician_names_unique(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0, n_technicians=4,
            technician_templates=["junior"],  # all same template
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        cfg = sampler.sample_config()
        names = [t.name for t in cfg.technicians.values()]
        assert len(set(names)) == len(names), "duplicate tech names would fail env init"

    def test_unknown_template_raises_at_construction(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            machine_templates=["no_such_template"],
        )
        with pytest.raises(ValueError, match="Unknown machine template"):
            RandomScenarioSampler(_base_cfg(), rcfg)

        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            technician_templates=["no_such_tech"],
        )
        with pytest.raises(ValueError, match="Unknown technician template"):
            RandomScenarioSampler(_base_cfg(), rcfg)

    def test_call_returns_simpy_scenario(self):
        rcfg = RandomizedScenarioConfig(enabled=True, seed=0)
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        sim_env, dispatcher = sampler()
        # Dispatcher has the expected fleet size and at least one machine
        assert len(dispatcher.techs) == rcfg.n_technicians
        assert hasattr(dispatcher, "machines")
        assert len(dispatcher.machines) > 0

    def test_seed_reproducibility(self):
        rcfg = RandomizedScenarioConfig(enabled=True, seed=999)
        a = RandomScenarioSampler(_base_cfg(), rcfg, seed=999)
        b = RandomScenarioSampler(_base_cfg(), rcfg, seed=999)
        for _ in range(3):
            cfg_a = a.sample_config()
            cfg_b = b.sample_config()
            assert sorted(cfg_a.machines) == sorted(cfg_b.machines)
            route_a = list(cfg_a.products.values())[0].route
            route_b = list(cfg_b.products.values())[0].route
            assert route_a == route_b

    def test_vocab_helpers(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            machine_templates=["cnc_weibull", "conveyor"],
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        assert "CNC" in sampler.all_machine_types()
        assert "Conveyor" in sampler.all_machine_types()
        # cnc_weibull has spindle + pump (+ rare drive); conveyor has motor + mechanical
        for ct in ("spindle", "pump", "motor", "mechanical"):
            assert ct in sampler.all_component_types()
