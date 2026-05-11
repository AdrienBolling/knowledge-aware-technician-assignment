"""Shared helpers for the analysis notebooks.

The notebooks rely on the per-experiment CSV reports written by
``experiment.runner.Experiment`` under ``reports/<exp_id>/``.

Typical usage from a notebook::

    from utils import load_run, list_runs, group_columns, rolling_mean
    ep, st, cfg = load_run("ppo_xl_randomized_500")
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_REPORTS_ROOT = Path.cwd() / "reports"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _resolve_reports_root(reports_root: str | Path | None) -> Path:
    """Resolve the root reports directory.

    Notebooks may be opened from ``analysis/`` or from the project root.
    We walk up a couple of levels looking for a ``reports/`` sibling.
    """
    if reports_root is not None:
        return Path(reports_root).expanduser().resolve()
    cwd = Path.cwd()
    for candidate in (cwd / "reports", cwd.parent / "reports", cwd.parent.parent / "reports"):
        if candidate.exists():
            return candidate.resolve()
    return (cwd / "reports").resolve()


def list_runs(reports_root: str | Path | None = None) -> pd.DataFrame:
    """Return a DataFrame of available runs sorted by most-recently modified."""
    root = _resolve_reports_root(reports_root)
    rows = []
    if root.exists():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            ep_path = d / "episode_metrics.csv"
            st_path = d / "step_metrics.csv"
            cfg_path = d / "config.json"
            mtime = max(
                (p.stat().st_mtime for p in (ep_path, st_path, cfg_path) if p.exists()),
                default=0.0,
            )
            rows.append(
                {
                    "exp_id": d.name,
                    "has_episodes": ep_path.exists(),
                    "has_steps": st_path.exists(),
                    "has_config": cfg_path.exists(),
                    "modified": pd.to_datetime(mtime, unit="s") if mtime else pd.NaT,
                    "path": str(d),
                }
            )
    df = pd.DataFrame(rows).sort_values("modified", ascending=False).reset_index(drop=True)
    return df


def latest_run(reports_root: str | Path | None = None) -> str | None:
    """Return the most-recently modified run's ``exp_id`` (or None)."""
    df = list_runs(reports_root)
    df = df[df["has_episodes"]]
    if df.empty:
        return None
    return str(df.iloc[0]["exp_id"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_run(
    exp_id: str,
    reports_root: str | Path | None = None,
    *,
    with_steps: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Load (episode_df, step_df, config) for a single ``exp_id``.

    ``with_steps=False`` skips reading the (potentially large) step CSV.
    """
    root = _resolve_reports_root(reports_root)
    run_dir = root / exp_id
    if not run_dir.exists():
        msg = f"No run directory at {run_dir}"
        raise FileNotFoundError(msg)
    ep_path = run_dir / "episode_metrics.csv"
    st_path = run_dir / "step_metrics.csv"
    cfg_path = run_dir / "config.json"
    ep_df = pd.read_csv(ep_path) if ep_path.exists() else pd.DataFrame()
    st_df = pd.read_csv(st_path) if (with_steps and st_path.exists()) else None
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return ep_df, st_df, cfg


def load_runs(
    exp_ids: list[str],
    reports_root: str | Path | None = None,
    *,
    with_steps: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]]:
    """Bulk-load several runs into a ``{exp_id: (ep, st, cfg)}`` mapping."""
    return {eid: load_run(eid, reports_root, with_steps=with_steps) for eid in exp_ids}


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def group_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return columns starting with ``prefix + "/"``."""
    pre = prefix.rstrip("/") + "/"
    return [c for c in df.columns if c.startswith(pre)]


def group_columns_by_prefix(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return ``{prefix: [columns…]}`` for every ``<prefix>/<sub>`` column."""
    out: dict[str, list[str]] = defaultdict(list)
    for c in df.columns:
        if "/" in c:
            out[c.split("/", 1)[0]].append(c)
    return dict(out)


def strip_prefix(columns: list[str], prefix: str) -> list[str]:
    """``[\"a/foo\", \"a/bar\"]`` -> ``[\"foo\", \"bar\"]``."""
    pre = prefix.rstrip("/") + "/"
    return [c[len(pre):] if c.startswith(pre) else c for c in columns]


def rolling_mean(s: pd.Series, window: int = 10) -> pd.Series:
    """Rolling mean with min_periods=1 (first window grows monotonically)."""
    return s.rolling(window=window, min_periods=1).mean()


__all__ = [
    "DEFAULT_REPORTS_ROOT",
    "group_columns",
    "group_columns_by_prefix",
    "latest_run",
    "list_runs",
    "load_run",
    "load_runs",
    "rolling_mean",
    "strip_prefix",
]
