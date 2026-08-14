"""Step-metric episode aggregation (2026-08-14 snapshot-bug fix).

Pins two behaviours:

* ``eval_human_vs_performance._episode_kpis`` reports step metrics as
  episode MEANS (accumulated over every decision) and episode metrics as
  their terminal value — never the old terminal-snapshot-for-everything.
* ``repair_stepmetric_columns.repair_dir`` retroactively rebuilds the
  recoverable columns of an existing ``episodes.csv`` from
  ``steps.csv.gz``, excluding ``mttr_rolling``'s pre-first-repair 0.0
  placeholder, leaving ``repair_time_delta`` (not in the step records)
  untouched, and staying idempotent with a one-time backup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def repair_mod():
    return _load("repair_stepmetric_columns")


def test_episode_kpis_means_step_metrics_and_passes_episode_metrics():
    eval_mod = _load("eval_human_vs_performance")
    # Simulated accumulation over a 3-decision episode:
    sums = {"repair_quality": 0.3 + 0.6 + 0.9, "mttr_rolling": 100.0 + 80.0}
    counts = {"repair_quality": 3, "mttr_rolling": 2}  # one 0.0 was skipped
    final_metrics = {
        "repair_quality": 0.9,      # terminal snapshot — must NOT be reported
        "mttr_rolling": 80.0,       # terminal snapshot — must NOT be reported
        "mttr": 92.5,               # episode metric, merged at termination only
        "finished_products": 41.0,  # episode metric
        "tech/alice": 0.4,          # per-tech series — excluded from KPIs
    }
    kpis = eval_mod._episode_kpis(final_metrics, sums, counts)
    assert kpis["repair_quality"] == pytest.approx(0.6)
    assert kpis["mttr_rolling"] == pytest.approx(90.0)
    assert kpis["mttr"] == pytest.approx(92.5)
    assert kpis["finished_products"] == pytest.approx(41.0)
    assert "tech/alice" not in kpis


def test_repair_dir_rebuilds_from_steps(tmp_path, repair_mod):
    ep = pd.DataFrame({
        "agent": ["a1", "a1", "a2"],
        "episode": [0, 1, 0],
        # terminal snapshots (wrong):
        "repair_quality": [0.10, 0.90, 0.50],
        "mttr_rolling": [126.0, 80.0, 99.0],
        "repair_time_delta_per": [5.0, -3.0, 0.0],
        "repair_time_delta": [7.0, -2.0, 1.0],  # unrecoverable — must survive
        "finished_products": [10.0, 11.0, 12.0],
    })
    st = pd.DataFrame({
        "agent": ["a1"] * 6 + ["a1"] * 2 + ["a2"] * 2,
        "episode": [0] * 6 + [1] * 2 + [0] * 2,
        "step": [1, 2, 3, 4, 5, 6, 1, 2, 1, 2],
        "repair_quality": [0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.9, 0.7, 0.5, 0.1],
        # leading zeros = pre-first-repair placeholders, must be excluded:
        "mttr_rolling": [0.0, 0.0, 100.0, 110.0, 120.0, 130.0, 80.0, 60.0, 99.0, 101.0],
        "repair_time_delta_per": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -3.0, -1.0, 0.0, 2.0],
    })
    ep.to_csv(tmp_path / "episodes.csv", index=False)
    st.to_csv(tmp_path / "steps.csv.gz", index=False, compression="gzip")

    manifest = repair_mod.repair_dir(tmp_path)
    fixed = pd.read_csv(tmp_path / "episodes.csv")
    r = fixed.set_index(["agent", "episode"])

    assert r.loc[("a1", 0), "repair_quality"] == pytest.approx(0.4)
    assert r.loc[("a1", 0), "mttr_rolling"] == pytest.approx(115.0)  # zeros out
    assert r.loc[("a1", 0), "repair_time_delta_per"] == pytest.approx(3.5)
    assert r.loc[("a1", 1), "repair_quality"] == pytest.approx(0.8)
    assert r.loc[("a1", 1), "mttr_rolling"] == pytest.approx(70.0)
    assert r.loc[("a2", 0), "mttr_rolling"] == pytest.approx(100.0)
    # untouched columns:
    assert list(r["repair_time_delta"]) == [7.0, -2.0, 1.0]
    assert list(r["finished_products"]) == [10.0, 11.0, 12.0]
    # backup + manifest:
    assert (tmp_path / "episodes.csv.presnapshotfix").is_file()
    assert manifest["columns_still_terminal_snapshot"] == ["repair_time_delta"]
    assert (tmp_path / "stepmetric_repair.json").is_file()

    # idempotent: second run recomputes to the same values, keeps ORIGINAL backup
    backup_before = (tmp_path / "episodes.csv.presnapshotfix").read_bytes()
    repair_mod.repair_dir(tmp_path)
    fixed2 = pd.read_csv(tmp_path / "episodes.csv")
    pd.testing.assert_frame_equal(fixed, fixed2)
    assert (tmp_path / "episodes.csv.presnapshotfix").read_bytes() == backup_before
