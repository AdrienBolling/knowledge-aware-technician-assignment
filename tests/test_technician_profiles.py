"""Tests for the technician-profile preloading mechanism.

Covers:

* :class:`TechnicianConfig.initial_knowledge_grid_path` default + override.
* :class:`GymTechnician` load semantics — file present, file missing,
  shape/parameter mismatch.
* The bundled profile artefacts under
  ``src/kata/resources/technician_profiles/`` exist and load cleanly
  via every shipped template (junior / senior / generalist / expert /
  trainee / motor_specialist / electronics_specialist).
* The profile builder is deterministic from a seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ongoing import KnowledgeGrid

from kata.core.config import TechnicianConfig
from kata.entities.technicians.GymTechnician import GymTechnician
from kata.EntityFactories.technician_factory import get_template
from kata.EntityFactories.technician_profile_builder import (
    DEFAULT_OUTPUT_DIR,
    ProfileSpec,
    build_default_profiles,
    build_one,
    default_profiles,
    gaussian_at,
    uniform_sampler,
)


# ---------------------------------------------------------------------------
# Config-level behaviour
# ---------------------------------------------------------------------------


class TestTechnicianConfigGridPath:
    def test_defaults_to_none(self):
        cfg = TechnicianConfig()
        assert cfg.initial_knowledge_grid_path is None

    def test_accepts_string(self):
        cfg = TechnicianConfig(initial_knowledge_grid_path="some/path.npz")
        assert cfg.initial_knowledge_grid_path == "some/path.npz"

    def test_templated_junior_carries_grid_path(self):
        """The bundled ``junior`` template references a pre-built grid."""
        tpl = get_template("junior")
        assert "initial_knowledge_grid_path" in tpl
        assert tpl["initial_knowledge_grid_path"].endswith(".npz")


# ---------------------------------------------------------------------------
# GymTechnician load semantics
# ---------------------------------------------------------------------------


class TestGymTechnicianGridLoad:
    def test_no_path_yields_empty_grid(self):
        tech = GymTechnician(TechnicianConfig())
        # A freshly-built grid has zero accumulated experience.
        assert tech.knowledge_grid.get_max_experiences() == 0.0

    def test_loaded_grid_has_nonzero_experience(self, tmp_path):
        # Build a profile + save, then construct a technician that
        # points at the saved file.
        out = tmp_path / "tiny.npz"
        spec = ProfileSpec(
            name="tiny",
            grid_shape=(10, 10),
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.7,
            n_tickets=20,
            embedding_sampler=uniform_sampler,
        )
        build_one(spec, seed=0).save(out)
        cfg = TechnicianConfig(
            knowledge_k_shape=(10, 10),
            knowledge_propagation_sigma=1.0,
            knowledge_transmission_factor=0.5,
            knowledge_learning_rate=0.7,
            initial_knowledge_grid_path=str(out),
        )
        tech = GymTechnician(cfg)
        assert tech.knowledge_grid.get_max_experiences() > 0.0
        assert tech.knowledge_grid.knowledge_volume() > 0.0

    def test_missing_file_warns_and_falls_back(self, tmp_path):
        cfg = TechnicianConfig(
            initial_knowledge_grid_path=str(tmp_path / "does_not_exist.npz")
        )
        with pytest.warns(RuntimeWarning, match="does not exist"):
            tech = GymTechnician(cfg)
        # Fell back to an empty grid rather than raising.
        assert tech.knowledge_grid.get_max_experiences() == 0.0

    def test_shape_mismatch_raises(self, tmp_path):
        # Save a 5x5 grid then try to load it into a 10x10-configured tech.
        out = tmp_path / "wrong_shape.npz"
        spec = ProfileSpec(
            name="wrong_shape",
            grid_shape=(5, 5),
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.7,
            n_tickets=10,
            embedding_sampler=uniform_sampler,
        )
        build_one(spec, seed=0).save(out)
        cfg = TechnicianConfig(
            knowledge_k_shape=(10, 10),
            knowledge_propagation_sigma=1.0,
            knowledge_transmission_factor=0.5,
            knowledge_learning_rate=0.7,
            initial_knowledge_grid_path=str(out),
        )
        with pytest.raises(ValueError, match="shape"):
            GymTechnician(cfg)

    def test_parameter_mismatch_raises(self, tmp_path):
        # Save with learning_rate=0.7 but configure the tech for 0.6.
        out = tmp_path / "wrong_lr.npz"
        spec = ProfileSpec(
            name="wrong_lr",
            grid_shape=(10, 10),
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.7,
            n_tickets=10,
            embedding_sampler=uniform_sampler,
        )
        build_one(spec, seed=0).save(out)
        cfg = TechnicianConfig(
            knowledge_k_shape=(10, 10),
            knowledge_propagation_sigma=1.0,
            knowledge_transmission_factor=0.5,
            knowledge_learning_rate=0.6,  # mismatch
            initial_knowledge_grid_path=str(out),
        )
        with pytest.raises(ValueError, match="learning_rate"):
            GymTechnician(cfg)


# ---------------------------------------------------------------------------
# Bundled artefacts
# ---------------------------------------------------------------------------


class TestBundledProfiles:
    """Every shipped template that references a profile must load cleanly."""

    @pytest.mark.parametrize(
        "template_name",
        [
            "expert",
            "senior",
            "generalist",
            "junior",
            "trainee",
            "motor_specialist",
            "electronics_specialist",
        ],
    )
    def test_template_grid_loads(self, template_name):
        tpl = get_template(template_name)
        path = tpl.get("initial_knowledge_grid_path")
        assert path is not None, f"template {template_name} has no grid path"
        cfg = TechnicianConfig(**tpl)
        tech = GymTechnician(cfg)
        # Every profile should carry *some* accumulated knowledge —
        # the threshold is loose to accommodate the trainee profile
        # whose tiny propagation sigma keeps single-bump peaks just
        # under 1.0 experience-equivalents.
        assert tech.knowledge_grid.knowledge_volume() > 0.0
        assert tech.knowledge_grid.get_max_knowledge() > 0.0

    def test_specialist_profiles_are_actually_specialised(self):
        """The motor / electronics specialists should have *higher*
        specialisation than the matching generalist profile.
        """
        gen = KnowledgeGrid.load(DEFAULT_OUTPUT_DIR / "generalist.npz")
        motor = KnowledgeGrid.load(DEFAULT_OUTPUT_DIR / "motor_specialist.npz")
        elec = KnowledgeGrid.load(DEFAULT_OUTPUT_DIR / "electronics_specialist.npz")
        assert motor.specialisation_index() > gen.specialisation_index()
        assert elec.specialisation_index() > gen.specialisation_index()


# ---------------------------------------------------------------------------
# Builder determinism
# ---------------------------------------------------------------------------


class TestBuilderDeterminism:
    def test_same_seed_produces_identical_grids(self, tmp_path):
        spec = default_profiles()[0]  # trainee --- cheapest
        g1 = build_one(spec, seed=42)
        g2 = build_one(spec, seed=42)
        assert np.array_equal(g1._grid, g2._grid)

    def test_different_seeds_diverge(self, tmp_path):
        spec = default_profiles()[3]  # senior --- 500 tickets
        g1 = build_one(spec, seed=1)
        g2 = build_one(spec, seed=2)
        assert not np.array_equal(g1._grid, g2._grid)

    def test_build_default_profiles_writes_everything(self, tmp_path):
        written = build_default_profiles(output_dir=tmp_path, seed=0, verbose=False)
        names = [p.name for p in written]
        # Every spec name should produce a file.
        for spec in default_profiles():
            assert f"{spec.name}.npz" in names
        # Plus the sidecar manifest.
        assert (tmp_path / "profiles.txt").is_file()
