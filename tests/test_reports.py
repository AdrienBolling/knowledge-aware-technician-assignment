"""Tests for the per-experiment metric reports.

Covers both the ``ReportWriter`` helper in isolation *and* an
end-to-end run through ``Experiment`` to confirm the runner writes
``episode_metrics.csv``, ``step_metrics.csv`` and ``config.json``
at the expected location.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from experiment.reports import ReportWriter


class TestReportWriter:
    def test_directory_created_on_init(self):
        with tempfile.TemporaryDirectory() as td:
            w = ReportWriter(td, "my_exp")
            assert w.dir == Path(td) / "my_exp"
            assert w.dir.is_dir()

    def test_flush_writes_csvs_with_expected_columns(self):
        with tempfile.TemporaryDirectory() as td:
            w = ReportWriter(td, "exp-A")
            w.add_step({"phase": "train", "episode": 1, "step": 1, "reward": 0.1})
            w.add_step({"phase": "train", "episode": 1, "step": 2, "reward": 0.3})
            w.add_episode(
                {"phase": "train", "episode": 1, "return": 0.4, "length": 2}
            )
            paths = w.flush()
            assert paths["episode_metrics"] is not None
            assert paths["step_metrics"] is not None

            ep_df = pd.read_csv(paths["episode_metrics"])
            st_df = pd.read_csv(paths["step_metrics"])
            assert len(ep_df) == 1
            assert len(st_df) == 2
            assert set(["phase", "episode", "return", "length"]).issubset(ep_df.columns)
            assert set(["phase", "episode", "step", "reward"]).issubset(st_df.columns)

    def test_save_config_persists_json(self):
        with tempfile.TemporaryDirectory() as td:
            w = ReportWriter(td, "exp-B")
            payload = {"exp_id": "exp-B", "env": {"x": 1}, "agent": {"y": 2}}
            path = w.save_config(payload)
            assert path.exists()
            assert json.loads(path.read_text()) == payload

    def test_flush_handles_empty_buffers(self):
        with tempfile.TemporaryDirectory() as td:
            w = ReportWriter(td, "exp-empty")
            paths = w.flush()
            assert paths["episode_metrics"] is None
            assert paths["step_metrics"] is None


class TestRunnerWritesReports:
    """A tiny end-to-end run must produce the three expected files."""

    def test_train_run_creates_all_three_artefacts(self):
        os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

        with tempfile.TemporaryDirectory() as td:
            from kata.core.config import (
                GymEnvConfig,
                KATAConfig,
                MachineConfig,
                ProductConfig,
                TechnicianConfig,
            )
            from kata.scenario import ScenarioBuilder
            from experiment.config import (
                AgentConfig,
                CheckpointConfig,
                EvalConfig,
                ExperimentConfig,
                ReportsConfig,
                WandbConfig,
            )
            from experiment.runner import Experiment

            env_cfg = KATAConfig(
                technicians={
                    "alpha": TechnicianConfig(name="alpha"),
                    "beta": TechnicianConfig(name="beta"),
                },
                machines={
                    "m0": MachineConfig(machine_type="generic", process_time=5),
                },
                products={
                    "p0": ProductConfig(product_type="generic", route=["generic"])
                },
                gym=GymEnvConfig(
                    max_episode_steps=4,
                    max_sim_time=200.0,
                    observation_representation="structured",
                ),
            )
            agent_cfg = AgentConfig(agent_type="random", params={})
            exp_cfg = ExperimentConfig(
                mode="train",
                seed=0,
                n_episodes=2,
                log_interval=1,
                eval=EvalConfig(enabled=False),
                checkpoint=CheckpointConfig(enabled=False),
                wandb=WandbConfig(enabled=False),
                reports=ReportsConfig(enabled=True, dir=td, exp_id="unit-test"),
            )
            exp = Experiment(env_cfg, agent_cfg, exp_cfg, quiet=True)
            # The agent registry expects this scenario_factory wiring;
            # bypass by replacing the env's scenario_factory shortcut.
            exp.env._scenario_factory = lambda: ScenarioBuilder(env_cfg).build()
            exp.eval_env._scenario_factory = lambda: ScenarioBuilder(env_cfg).build()

            exp.run()

            base = Path(td) / "unit-test"
            assert (base / "config.json").exists()
            assert (base / "episode_metrics.csv").exists()
            assert (base / "step_metrics.csv").exists()

            cfg_snapshot = json.loads((base / "config.json").read_text())
            assert cfg_snapshot["exp_id"] == "unit-test"
            assert "env" in cfg_snapshot
            assert "agent" in cfg_snapshot
            assert "experiment" in cfg_snapshot

            ep_df = pd.read_csv(base / "episode_metrics.csv")
            assert len(ep_df) == 2  # n_episodes
            # Mandatory columns for episode reports
            for col in (
                "phase",
                "episode",
                "seed",
                "return",
                "length",
                "sim_time_end",
                "wall_clock_seconds",
                "unique_techs_used",
                "loss",
                "entropy",
            ):
                assert col in ep_df.columns, f"missing episode column: {col}"
            assert (ep_df["phase"] == "train").all()
            assert (ep_df["wall_clock_seconds"] >= 0).all()

            st_df = pd.read_csv(base / "step_metrics.csv")
            assert len(st_df) > 0
            # Mandatory columns for step reports
            for col in (
                "phase",
                "episode",
                "step",
                "seed",
                "sim_time",
                "pending_queue_size",
                "has_open_ticket",
                "ticket_machine_id",
                "ticket_machine_type",
                "ticket_component_type",
                "ticket_wait_time",
                "action",
                "reward",
            ):
                assert col in st_df.columns, f"missing step column: {col}"
            assert (st_df["phase"] == "train").all()
