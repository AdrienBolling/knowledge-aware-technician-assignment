"""Per-experiment metric reports as pandas DataFrames.

Each ``Experiment`` instance can write three artefacts under
``reports/<exp_id>/``:

* ``config.json``         – the full env + agent + experiment config
                            snapshot, so an ``exp_id`` is always
                            traceable to what produced it.
* ``episode_metrics.csv`` – one row per episode (``phase`` column
                            distinguishes ``train`` / ``inline_eval``
                            / ``eval``).
* ``step_metrics.csv``    – one row per env step.

The runner appends to in-memory lists during the run and flushes
once at the end, so writing is cheap during training.  Re-runs of
the same ``exp_id`` overwrite the previous artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ReportWriter:
    """Accumulator + writer for per-experiment CSV reports."""

    def __init__(self, reports_dir: str | Path, exp_id: str) -> None:
        self.exp_id = str(exp_id)
        self.dir = Path(reports_dir) / self.exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._step_rows: list[dict[str, Any]] = []
        self._episode_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_step(self, row: dict[str, Any]) -> None:
        """Append one step-level row (flat dict of scalars)."""
        self._step_rows.append(dict(row))

    def add_episode(self, row: dict[str, Any]) -> None:
        """Append one episode-level row (flat dict of scalars)."""
        self._episode_rows.append(dict(row))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self, config: dict[str, Any]) -> Path:
        """Persist a config snapshot as JSON.  Returns the file path."""
        path = self.dir / "config.json"
        with path.open("w") as f:
            json.dump(config, f, indent=2, default=str)
        return path

    def flush(self) -> dict[str, Path | None]:
        """Write the accumulated rows to disk.

        Returns
        -------
        dict
            ``{"episode_metrics": Path | None, "step_metrics": Path | None}``
            pointing to the written CSVs (or ``None`` if no rows were
            collected for that category).
        """
        out: dict[str, Path | None] = {
            "episode_metrics": None,
            "step_metrics": None,
        }
        if self._episode_rows:
            path = self.dir / "episode_metrics.csv"
            pd.DataFrame(self._episode_rows).to_csv(path, index=False)
            out["episode_metrics"] = path
        if self._step_rows:
            path = self.dir / "step_metrics.csv"
            pd.DataFrame(self._step_rows).to_csv(path, index=False)
            out["step_metrics"] = path
        return out

    # ------------------------------------------------------------------
    # Read-back helpers (purely a convenience for tests / notebooks)
    # ------------------------------------------------------------------

    def read_episode_metrics(self) -> pd.DataFrame:
        path = self.dir / "episode_metrics.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def read_step_metrics(self) -> pd.DataFrame:
        path = self.dir / "step_metrics.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


__all__ = ["ReportWriter"]
