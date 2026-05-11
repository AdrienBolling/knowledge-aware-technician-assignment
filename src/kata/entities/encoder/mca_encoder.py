"""MCA-based encoder for mapping RepairRequests to KnowledgeGrid coordinates.

Uses Multiple Correspondence Analysis (via ``prince``) to embed categorical
repair-request features into a low-dimensional continuous space, then
discretises the result onto the knowledge grid.

The encoder must be *fitted* on a collection of repair requests before it can
encode new ones.  The typical workflow is:

1. Run a warmup phase that collects repair requests under a heuristic policy.
2. Call ``fit(requests)`` to train the MCA.
3. Use ``encode(request)`` during the real episode.

Validity diagnostics
--------------------
After ``fit``, the encoder exposes:

* ``fit_diagnostics`` — a dict containing eigenvalues, explained inertia,
  cumulative inertia, training-set size and unique-feature-tuple count.
* ``is_well_trained()`` — returns True only when the cumulative inertia of
  the kept components clears a configurable threshold.

A WARNING is also emitted via ``logging`` when the fit looks degenerate
(too little inertia, or the produced coords collapse to a single cell).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import prince
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kata.entities.requests.RepairRequest import RepairRequest

logger = logging.getLogger(__name__)


def _features_from_request(request: RepairRequest) -> dict[str, str]:
    """Extract categorical features from a RepairRequest.

    Only includes features that carry *independent* information:

    * ``machine_type`` — the type of the broken machine (e.g. ``CNC``).
    * ``component_type`` — the failed component's type (``motor``, …).

    ``component_id`` is intentionally **not** included: it is in 1-to-1
    correspondence with ``component_type`` (each id maps to exactly one
    type) and therefore adds noise to the MCA without contributing
    independent variance.  ``repair_bucket`` is similarly redundant —
    every component has a fixed ``base_repair_time`` so the bucket is
    determined by ``component_type``.
    """
    machine_type: str = getattr(request.machine, "mtype", "unknown")
    comp_info = request.get_failed_component_info()
    component_type = comp_info["component_type"] if comp_info else "none"
    return {
        "machine_type": machine_type,
        "component_type": component_type,
    }


class MCAEncoder:
    """Encoder that uses fitted MCA to map repair requests to grid coordinates.

    Parameters
    ----------
    grid_shape:
        Shape of the target KnowledgeGrid.
    n_components:
        Number of MCA components to keep.  Must be >= len(grid_shape).
    min_cumulative_inertia:
        Threshold (in [0, 1]) under which ``is_well_trained`` returns
        ``False`` — the MCA fit captured too little of the variance.

    """

    def __init__(
        self,
        grid_shape: tuple[int, ...] = (10, 10),
        n_components: int = 2,
        min_cumulative_inertia: float = 0.30,
    ) -> None:
        self.grid_shape = grid_shape
        self.n_components = max(n_components, len(grid_shape))
        self.min_cumulative_inertia = float(min_cumulative_inertia)

        self._mca: prince.MCA | None = None
        self._fitted = False
        # Robust normalisation bounds (5th / 95th percentiles of the
        # training-set MCA coordinates).  Using percentiles instead of
        # raw min/max prevents a single outlier from squashing all
        # other categories into adjacent cells.
        self._col_low: NDArray[np.float64] | None = None
        self._col_high: NDArray[np.float64] | None = None

        self.fit_diagnostics: dict[str, float | int | list[float]] = {}

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, requests: list[RepairRequest]) -> MCAEncoder:
        """Fit the MCA on categorical features extracted from *requests*.

        At least 2 distinct feature tuples are needed for a meaningful
        fit — when fewer are seen, the encoder stays in fallback mode
        and ``fitted`` remains ``False``.
        """
        if len(requests) < 2:
            logger.warning(
                "MCAEncoder.fit: only %d request(s) collected — "
                "encoder will fall back to hash encoding.",
                len(requests),
            )
            return self

        df = pd.DataFrame([_features_from_request(r) for r in requests]).astype(str)
        n_unique_combos = df.drop_duplicates().shape[0]

        if n_unique_combos < 2:
            logger.warning(
                "MCAEncoder.fit: collected %d requests but they all share "
                "the same feature tuple — fit is degenerate, falling back.",
                len(requests),
            )
            return self

        # Drop columns that have only one unique value — prince refuses to
        # fit them and they contribute nothing anyway.
        df = df.loc[:, df.nunique() > 1]
        if df.shape[1] == 0:
            logger.warning(
                "MCAEncoder.fit: no informative columns after dropping "
                "constants; falling back to hash encoding."
            )
            return self

        n_comp = min(self.n_components, max(df.shape[1], 2))
        self._mca = prince.MCA(n_components=n_comp)
        self._mca.fit(df)

        # Diagnostics
        eigenvalues = list(map(float, np.asarray(getattr(self._mca, "eigenvalues_", []))))
        explained = self._explained_inertia(eigenvalues)
        cumulative = float(np.sum(explained)) if explained else 0.0
        self.fit_diagnostics = {
            "n_requests": int(len(requests)),
            "n_unique_feature_tuples": int(n_unique_combos),
            "n_columns_used": int(df.shape[1]),
            "n_components_fit": int(n_comp),
            "eigenvalues": eigenvalues,
            "explained_inertia": list(explained),
            "cumulative_inertia": cumulative,
        }

        coords = self._mca.transform(df).values  # (N, n_comp)
        # 5th / 95th percentile normalisation — robust to outliers
        self._col_low = np.percentile(coords, 5, axis=0)
        self._col_high = np.percentile(coords, 95, axis=0)
        span = self._col_high - self._col_low
        # Where the span is degenerate (a near-constant component),
        # widen it slightly so normalisation maps into the grid centre.
        degenerate = span < 1e-9
        if degenerate.any():
            self._col_low[degenerate] -= 0.5
            self._col_high[degenerate] += 0.5

        self._fitted = True

        if cumulative < self.min_cumulative_inertia:
            logger.warning(
                "MCAEncoder.fit: cumulative inertia %.2f%% is below the "
                "configured threshold %.0f%% — embedding may be weak.  "
                "Diagnostics: %s",
                cumulative * 100,
                self.min_cumulative_inertia * 100,
                self.fit_diagnostics,
            )

        return self

    @staticmethod
    def _explained_inertia(eigenvalues: list[float]) -> list[float]:
        """Return the fraction of total inertia carried by each component."""
        if not eigenvalues:
            return []
        total = float(sum(eigenvalues))
        if total <= 0:
            return [0.0] * len(eigenvalues)
        return [float(e / total) for e in eigenvalues]

    def is_well_trained(self) -> bool:
        """Return True iff the MCA fit cleared the inertia threshold."""
        if not self._fitted:
            return False
        cum = float(self.fit_diagnostics.get("cumulative_inertia", 0.0))
        return cum >= self.min_cumulative_inertia

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Return the grid coordinate for *request*.

        If the MCA has not been fitted yet, falls back to deterministic
        hashing (same behaviour as ``HashEncoder``).
        """
        if not self._fitted or self._mca is None:
            return self._fallback_encode(request)

        features = _features_from_request(request)
        df = pd.DataFrame([features]).astype(str)
        # Keep only columns the MCA was actually fitted on
        cols = list(getattr(self._mca, "columns_", df.columns))
        df = df.loc[:, [c for c in cols if c in df.columns]]
        if df.shape[1] == 0:
            return self._fallback_encode(request)

        coords = self._mca.transform(df).values[0]  # (n_comp,)

        # Robust normalisation: clip into [low, high] then linearly map
        # to [0, 1] before discretising.
        normalised = (coords - self._col_low) / (self._col_high - self._col_low)
        normalised = np.clip(normalised, 0.0, 1.0)

        grid_coords = []
        for i, dim in enumerate(self.grid_shape):
            idx = i if i < len(normalised) else 0
            # Uniform binning: each grid bin spans 1/dim of the [0, 1]
            # interval, with the last bin capped to ``dim - 1``.
            bin_idx = int(normalised[idx] * dim)
            grid_coords.append(min(bin_idx, dim - 1))
        return np.array(grid_coords, dtype=np.intp)

    def _fallback_encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Hash-based fallback when MCA is not yet fitted."""
        features = _features_from_request(request)
        key = f"{features['machine_type']}::{features['component_type']}"
        h = hash(key)
        return np.array([abs(h) % dim for dim in self.grid_shape], dtype=np.intp)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the fitted encoder."""
        if not self._fitted:
            return "MCAEncoder(not fitted)"
        d = self.fit_diagnostics
        ei = ", ".join(f"{x * 100:.1f}%" for x in d.get("explained_inertia", []))
        return (
            "MCAEncoder("
            f"n_requests={d['n_requests']}, "
            f"unique_combos={d['n_unique_feature_tuples']}, "
            f"n_components={d['n_components_fit']}, "
            f"explained=[{ei}], "
            f"cumulative={d['cumulative_inertia'] * 100:.1f}%, "
            f"well_trained={self.is_well_trained()})"
        )
