"""MCA-based encoder for mapping RepairRequests to KnowledgeGrid coordinates.

Uses Multiple Correspondence Analysis (via ``prince``) to embed categorical
repair-request features into a low-dimensional continuous space, then
discretises the result onto the knowledge grid.

The encoder must be *fitted* on a collection of repair requests before it can
encode new ones.  The typical workflow is:

1. Run a warmup phase that collects repair requests under a heuristic policy.
2. Call ``fit(requests)`` to train the MCA.
3. Use ``encode(request)`` during the real episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import prince
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kata.entities.requests.RepairRequest import RepairRequest


def _features_from_request(request: RepairRequest) -> dict[str, str]:
    """Extract categorical features from a RepairRequest as strings."""
    machine_type: str = getattr(request.machine, "mtype", "unknown")
    comp_info = request.get_failed_component_info()
    component_type = comp_info["component_type"] if comp_info else "none"
    component_id = comp_info["component_id"] if comp_info else "none"
    repair_bucket = _bucket_repair_time(request.get_repair_time())
    return {
        "machine_type": machine_type,
        "component_type": component_type,
        "component_id": component_id,
        "repair_bucket": repair_bucket,
    }


def _bucket_repair_time(t: float) -> str:
    """Discretise repair time into categorical buckets."""
    if t <= 5:
        return "very_short"
    if t <= 15:
        return "short"
    if t <= 30:
        return "medium"
    if t <= 60:
        return "long"
    return "very_long"


class MCAEncoder:
    """Encoder that uses fitted MCA to map repair requests to grid coordinates.

    Parameters
    ----------
    grid_shape:
        Shape of the target KnowledgeGrid.
    n_components:
        Number of MCA components to keep.  Must be >= len(grid_shape).

    """

    def __init__(
        self,
        grid_shape: tuple[int, ...] = (10, 10),
        n_components: int = 2,
    ) -> None:
        self.grid_shape = grid_shape
        self.n_components = max(n_components, len(grid_shape))
        self._mca: prince.MCA | None = None
        self._fitted = False
        # Store min/max of each MCA component for normalisation
        self._col_min: NDArray[np.float64] | None = None
        self._col_max: NDArray[np.float64] | None = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, requests: list[RepairRequest]) -> MCAEncoder:
        """Fit the MCA on categorical features extracted from *requests*.

        At least ``n_components + 1`` distinct requests are needed for a
        meaningful fit.  If fewer are provided, the encoder falls back to
        hash-based encoding.
        """
        if len(requests) < 2:
            return self

        df = pd.DataFrame([_features_from_request(r) for r in requests])

        # prince.MCA expects all columns to be categorical / object
        for col in df.columns:
            df[col] = df[col].astype(str)

        n_comp = min(self.n_components, len(df.columns))
        self._mca = prince.MCA(n_components=n_comp)
        self._mca.fit(df)

        # Transform training data to get normalisation bounds
        coords = self._mca.transform(df).values  # (N, n_comp)
        self._col_min = coords.min(axis=0)
        self._col_max = coords.max(axis=0)
        # Avoid division by zero
        span = self._col_max - self._col_min
        span[span == 0] = 1.0
        self._col_max = self._col_min + span

        self._fitted = True
        return self

    def encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Return the grid coordinate for *request*.

        If the MCA has not been fitted yet, falls back to deterministic
        hashing (same behaviour as ``HashEncoder``).
        """
        if not self._fitted or self._mca is None:
            return self._fallback_encode(request)

        features = _features_from_request(request)
        df = pd.DataFrame([features])
        for col in df.columns:
            df[col] = df[col].astype(str)

        coords = self._mca.transform(df).values[0]  # (n_comp,)

        # Normalise to [0, 1] using training bounds, then map to grid
        normalised = (coords - self._col_min) / (self._col_max - self._col_min)
        normalised = np.clip(normalised, 0.0, 1.0)

        # Map first len(grid_shape) components to grid indices
        grid_coords = []
        for i, dim in enumerate(self.grid_shape):
            idx = i if i < len(normalised) else 0
            grid_coords.append(int(normalised[idx] * (dim - 1)))
        return np.array(grid_coords, dtype=np.intp)

    def _fallback_encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Hash-based fallback when MCA is not yet fitted."""
        features = _features_from_request(request)
        key = f"{features['machine_type']}::{features['component_type']}"
        h = hash(key)
        return np.array([abs(h) % dim for dim in self.grid_shape], dtype=np.intp)
