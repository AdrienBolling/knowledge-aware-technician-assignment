"""Lifecycle event log of the benchmark harness (``--merge`` semantics).

A rerun agent's rows in ``lifecycle_events.csv`` must be replaced exactly
like its rows in ``episodes.csv`` and ``steps.csv.gz`` — also when the rerun
fires no lifecycle event (for example a ``--sim`` override that ends before
the first event).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def eval_mod():
    name = "eval_human_vs_performance"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(agent: str, targets: list[str]) -> list[dict]:
    return [
        {"agent": agent, "episode": 0, "time": 800000.0 + i,
         "kind": "retire_technician", "target": t}
        for i, t in enumerate(targets)
    ]


def _seed_file(eval_mod, path: Path) -> None:
    eval_mod.write_lifecycle_log(
        path, _rows("random", ["expert_6"]) + _rows("topsis", ["expert_7"]),
        rerun={"random", "topsis"}, merge=False,
    )


def test_merge_empty_rerun_drops_stale_rows(tmp_path, eval_mod):
    path = tmp_path / "lifecycle_events.csv"
    _seed_file(eval_mod, path)
    eval_mod.write_lifecycle_log(path, [], rerun={"random"}, merge=True)
    df = pd.read_csv(path)
    assert set(df["agent"]) == {"topsis"}
    assert df["target"].tolist() == ["expert_7"]


def test_merge_empty_rerun_of_only_agent_removes_file(tmp_path, eval_mod):
    path = tmp_path / "lifecycle_events.csv"
    eval_mod.write_lifecycle_log(path, _rows("random", ["expert_6"]),
                                 rerun={"random"}, merge=False)
    assert path.is_file()
    eval_mod.write_lifecycle_log(path, [], rerun={"random"}, merge=True)
    assert not path.exists()


def test_merge_replaces_rerun_rows_and_keeps_others(tmp_path, eval_mod):
    path = tmp_path / "lifecycle_events.csv"
    _seed_file(eval_mod, path)
    eval_mod.write_lifecycle_log(path, _rows("random", ["junior_1", "trainee_8"]),
                                 rerun={"random"}, merge=True)
    df = pd.read_csv(path)
    assert df[df["agent"] == "topsis"]["target"].tolist() == ["expert_7"]
    assert df[df["agent"] == "random"]["target"].tolist() == ["junior_1", "trainee_8"]
    assert df.columns.tolist() == eval_mod.LIFECYCLE_LOG_COLUMNS


def test_no_merge_overwrites_and_empty_run_removes_file(tmp_path, eval_mod):
    path = tmp_path / "lifecycle_events.csv"
    _seed_file(eval_mod, path)
    eval_mod.write_lifecycle_log(path, _rows("round_robin", ["senior_17"]),
                                 rerun={"round_robin"}, merge=False)
    assert set(pd.read_csv(path)["agent"]) == {"round_robin"}
    eval_mod.write_lifecycle_log(path, [], rerun={"round_robin"}, merge=False)
    assert not path.exists()
