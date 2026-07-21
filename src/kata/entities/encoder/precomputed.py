"""Precomputed ticket->grid encoder loaded from a canonical artifact.

The artifact (``run_configs/embeddings/ticket_grid_som.json``) is fitted
OFFLINE (SOM on real-world-observable ticket features; see the paper's
encoder discussion) and versioned like the frozen vocabulary, so every
run shares one deterministic transfer geometry and no per-env warmup is
needed.  Unknown keys fall back to a hash placement projected onto the
grid's BORDER ring: the curated interior stays reserved for known
failure types, and unseen ones land in structurally-separated, low-
transfer edge cells until the map is refitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kata.entities.encoder.base import HashEncoder


class PrecomputedEncoder:
    """Table lookup from a fitted embedding artifact, hash-to-border fallback."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.path = str(artifact_path)
        art = json.loads(Path(artifact_path).read_text())
        self.grid_shape = tuple(art.get("grid_shape", (10, 10)))
        self.embedding_bounds = np.asarray(
            art.get("embedding_bounds", [[0.0, 100.0]] * len(self.grid_shape)),
            dtype=np.float64,
        )
        self._table: dict[str, np.ndarray] = {
            key: np.asarray(coords, dtype=np.float64)
            for key, coords in art["placements"].items()
        }
        self._hash = HashEncoder(
            grid_shape=self.grid_shape, embedding_bounds=self.embedding_bounds
        )

    def _key(self, request) -> str:
        machine_type = getattr(request.machine, "mtype", "unknown")
        comp_info = request.get_failed_component_info()
        component_type = comp_info["component_type"] if comp_info else "none"
        return f"{machine_type}:{component_type}"

    def _border_project(self, emb: np.ndarray) -> np.ndarray:
        """Project a fallback embedding onto the grid's border ring."""
        cell_sizes = (self.embedding_bounds[:, 1] - self.embedding_bounds[:, 0]) / np.asarray(
            self.grid_shape, dtype=np.float64
        )
        cells = np.clip(
            ((emb - self.embedding_bounds[:, 0]) // cell_sizes).astype(int),
            0,
            np.asarray(self.grid_shape) - 1,
        )
        interior = all(0 < c < s - 1 for c, s in zip(cells, self.grid_shape))
        if interior:
            # Push the axis closest to an edge onto that edge.
            dists = [min(c, s - 1 - c) for c, s in zip(cells, self.grid_shape)]
            ax = int(np.argmin(dists))
            s = self.grid_shape[ax]
            cells[ax] = 0 if cells[ax] < s / 2 else s - 1
        return self.embedding_bounds[:, 0] + (cells + 0.5) * cell_sizes

    def encode(self, request) -> np.ndarray:
        hit = self._table.get(self._key(request))
        if hit is not None:
            return hit.copy()
        return self._border_project(self._hash.encode(request))
