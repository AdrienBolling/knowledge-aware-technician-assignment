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


# ======================================================================
# Multi-episode reuse + variable-fleet modes
# ======================================================================


class TestEpisodesPerScenario:
    """``episodes_per_scenario`` keeps the same KATAConfig across k builds."""

    def test_k1_resamples_every_call(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=11, episodes_per_scenario=1,
            n_machines_min=8, n_machines_max=15,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=11)
        # Call __call__ a few times; sample_config is invoked each time.
        for _ in range(4):
            sampler()
        assert sampler._call_count == 4

    def test_k3_reuses_for_three_calls_then_resamples(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=22, episodes_per_scenario=3,
            n_machines_min=6, n_machines_max=10,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=22)
        seen_configs = []
        for _ in range(7):
            sampler()  # builds env; we just record the cached config identity
            seen_configs.append(sampler._cached_config)
        # The first 3 calls share one config object; calls 4-6 share the second;
        # the 7th call is the start of a 3rd block.
        assert seen_configs[0] is seen_configs[1] is seen_configs[2]
        assert seen_configs[3] is seen_configs[4] is seen_configs[5]
        assert seen_configs[0] is not seen_configs[3]
        assert seen_configs[6] is not seen_configs[3]
        # sample_config called exactly 3 times (start of each k-block).
        assert sampler._call_count == 3

    def test_reset_scenario_cache_forces_resample(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=33, episodes_per_scenario=5,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=33)
        sampler()
        first = sampler._cached_config
        sampler.reset_scenario_cache()
        sampler()
        second = sampler._cached_config
        assert first is not second


class TestVariableTechnicianCount:
    """``n_technicians_min/max`` and ``techs_per_machine_min/max`` modes."""

    def test_range_mode_draws_within_bounds(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            n_technicians_min=3, n_technicians_max=12,
            n_machines_min=8, n_machines_max=15,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        counts = []
        for _ in range(40):
            cfg = sampler.sample_config()
            counts.append(len(cfg.technicians))
        assert all(3 <= c <= 12 for c in counts)
        # With 40 draws over a 10-wide range we should see ≥3 distinct values.
        assert len(set(counts)) >= 3

    def test_ratio_mode_scales_with_machine_count(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            n_machines_min=20, n_machines_max=40,
            techs_per_machine_min=0.20, techs_per_machine_max=0.40,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        for _ in range(50):
            cfg = sampler.sample_config()
            n_m = len(cfg.machines)
            n_t = len(cfg.technicians)
            ratio = n_t / n_m
            # Round-off can push the realised ratio slightly outside the
            # [0.20, 0.40] band — allow one tech of slack on either side.
            lo = 0.20 - (1.0 / n_m)
            hi = 0.40 + (1.0 / n_m)
            assert lo <= ratio <= hi, (n_m, n_t, ratio)
            assert n_t >= 1

    def test_ratio_mode_respects_optional_absolute_clamps(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            n_machines_min=100, n_machines_max=100,
            techs_per_machine_min=0.50, techs_per_machine_max=0.50,
            n_technicians_min=10, n_technicians_max=20,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        for _ in range(5):
            cfg = sampler.sample_config()
            # Raw ratio gives 50 techs, clamp to [10, 20] → exactly 20.
            assert len(cfg.technicians) == 20

    def test_ratio_takes_precedence_over_range(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=0,
            n_machines_min=50, n_machines_max=50,
            n_technicians_min=2, n_technicians_max=4,
            techs_per_machine_min=0.10, techs_per_machine_max=0.10,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        cfg = sampler.sample_config()
        # Ratio gives 5 techs; range mode would give 2-4.  Ratio wins,
        # then is clamped to the [2,4] bound → exactly 4.
        assert len(cfg.technicians) == 4

    def test_fixed_mode_still_works(self):
        rcfg = RandomizedScenarioConfig(enabled=True, seed=0, n_technicians=7)
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=0)
        for _ in range(5):
            cfg = sampler.sample_config()
            assert len(cfg.technicians) == 7

    def test_route_stays_coherent_with_resampled_machines(self):
        rcfg = RandomizedScenarioConfig(
            enabled=True, seed=44, episodes_per_scenario=2,
            n_machines_min=4, n_machines_max=8,
            techs_per_machine_min=0.3, techs_per_machine_max=0.6,
            route_min_length=1, route_max_length=4,
        )
        sampler = RandomScenarioSampler(_base_cfg(), rcfg, seed=44)
        for _ in range(6):
            sampler()  # forces sample / rebuild
            cfg = sampler._cached_config
            assert cfg is not None
            types_present = {m.machine_type for m in cfg.machines.values()}
            route = list(cfg.products.values())[0].route
            assert set(route).issubset(types_present)
            assert len(cfg.technicians) >= 1


class TestVariableFleetValidation:
    def test_half_set_range_is_rejected(self):
        with pytest.raises(ValueError, match="n_technicians_min and n_technicians_max"):
            RandomScenarioSampler(
                _base_cfg(),
                RandomizedScenarioConfig(enabled=True, n_technicians_min=2),
            )

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ValueError, match="n_technicians_max"):
            RandomScenarioSampler(
                _base_cfg(),
                RandomizedScenarioConfig(
                    enabled=True, n_technicians_min=10, n_technicians_max=3,
                ),
            )

    def test_half_set_ratio_is_rejected(self):
        with pytest.raises(ValueError, match="techs_per_machine"):
            RandomScenarioSampler(
                _base_cfg(),
                RandomizedScenarioConfig(enabled=True, techs_per_machine_min=0.1),
            )

    def test_inverted_ratio_is_rejected(self):
        with pytest.raises(ValueError, match="techs_per_machine_max"):
            RandomScenarioSampler(
                _base_cfg(),
                RandomizedScenarioConfig(
                    enabled=True,
                    techs_per_machine_min=0.5,
                    techs_per_machine_max=0.2,
                ),
            )
