"""Encoder protocol and implementations for mapping RepairRequests to grid coordinates.

The encoder converts a :class:`RepairRequest` into a real-valued
**embedding** living inside the :class:`KnowledgeGrid`'s bounded
embedding space (default ``[0, 100]`` per axis).  The grid then maps
that embedding to a cell internally and applies its Gaussian bump.

Previously the encoders returned integer *grid coordinates* directly
(``[0, shape - 1]`` per axis); the grid then re-interpreted those
values as embeddings in ``[0, 100]²`` and clipped every input to the
``(0, 0)`` corner — collapsing every (machine_type, component_type)
pair onto a single cell.  Emitting embeddings in the grid's space
fixes that contract.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kata.entities.requests.RepairRequest import RepairRequest


# Default embedding bounds matching ``ongoing.KnowledgeGrid``'s default
# ``embedding_bounds = [[0, 100], [0, 100]]``.  Encoders default to
# producing values in this box so the grid receives well-distributed
# input out of the box.
DEFAULT_EMBEDDING_BOUNDS: NDArray[np.float64] = np.array(
    [[0.0, 100.0], [0.0, 100.0]], dtype=np.float64
)


def _signature_to_float(key: str, dim: int) -> float:
    """Deterministically hash a string + dimension into ``[0, 1)``.

    Uses BLAKE2b with the dimension index as a salt so each axis is
    independent — the same ``key`` produces decorrelated coordinates
    on different axes.  Python's built-in ``hash`` is salted per
    interpreter run and therefore non-portable; BLAKE2b is stable
    across processes.
    """
    h = hashlib.blake2b(
        f"{key}::dim{dim}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") / (1 << 64)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RequestEncoder(Protocol):
    """Encodes a RepairRequest into an embedding for the KnowledgeGrid.

    The returned array must lie inside the grid's ``embedding_bounds``;
    coordinates are floats, not grid indices.
    """

    def encode(self, request: RepairRequest) -> NDArray[np.float64]:
        """Return the embedding for *request*."""
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class HashEncoder:
    """Deterministic hash-based encoder.

    Maps each ``(machine_type, component_type)`` pair to a fixed point
    in the grid's embedding space by hashing the key independently on
    each axis.  Fast, reproducible, and the values are spread evenly
    across ``embedding_bounds`` so distinct repair types reliably
    land in different grid cells.

    Parameters
    ----------
    grid_shape:
        Grid shape — kept for backwards compatibility; only used to
        infer the embedding dimensionality when ``embedding_bounds``
        is left at the default.
    embedding_bounds:
        ``(n_dims, 2)`` array of ``[lo, hi]`` bounds.  Must match
        whatever bounds the :class:`KnowledgeGrid` consuming this
        encoder was constructed with (the default is identical to the
        grid's default).
    """

    def __init__(
        self,
        grid_shape: tuple[int, ...] = (10, 10),
        embedding_bounds: NDArray[np.float64] | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        if embedding_bounds is None:
            embedding_bounds = np.array(
                [[0.0, 100.0]] * len(grid_shape), dtype=np.float64
            )
        self.embedding_bounds = np.asarray(embedding_bounds, dtype=np.float64)
        if self.embedding_bounds.shape[0] != len(grid_shape):
            msg = (
                f"embedding_bounds has {self.embedding_bounds.shape[0]} rows "
                f"but grid_shape has {len(grid_shape)} dimensions"
            )
            raise ValueError(msg)

    def encode(self, request: RepairRequest) -> NDArray[np.float64]:
        """Return the embedding derived from the failure-key hash."""
        machine_type: str = getattr(request.machine, "mtype", "unknown")
        comp_info = request.get_failed_component_info()
        component_type: str = comp_info["component_type"] if comp_info else "none"
        key = f"{machine_type}::{component_type}"

        out = np.empty(self.embedding_bounds.shape[0], dtype=np.float64)
        for d, (lo, hi) in enumerate(self.embedding_bounds):
            u = _signature_to_float(key, d)
            out[d] = float(lo) + u * (float(hi) - float(lo))
        return out


class LookupEncoder:
    """Table-based encoder with explicit ``(machine_type, component_type)`` → embedding mapping.

    Useful when you want full control over where each failure type
    lands in the embedding space.  Unknown keys fall back to the
    HashEncoder so behaviour is graceful.

    The ``lookup`` table maps ``(machine_type, component_type)`` →
    a coordinate tuple **in the embedding space** (e.g. ``(50.0, 25.0)``
    when ``embedding_bounds = [[0, 100], [0, 100]]``).
    """

    def __init__(
        self,
        lookup: dict[tuple[str, str], tuple[float, ...]] | None = None,
        grid_shape: tuple[int, ...] = (10, 10),
        embedding_bounds: NDArray[np.float64] | None = None,
    ) -> None:
        self.lookup: dict[tuple[str, str], tuple[float, ...]] = lookup or {}
        self._fallback = HashEncoder(grid_shape, embedding_bounds=embedding_bounds)
        self.embedding_bounds = self._fallback.embedding_bounds

    def encode(self, request: RepairRequest) -> NDArray[np.float64]:
        """Return the embedding from the lookup or fall back to the hash."""
        machine_type: str = getattr(request.machine, "mtype", "unknown")
        comp_info = request.get_failed_component_info()
        component_type: str = comp_info["component_type"] if comp_info else "none"

        key = (machine_type, component_type)
        if key in self.lookup:
            return np.array(self.lookup[key], dtype=np.float64)
        return self._fallback.encode(request)


# ---------------------------------------------------------------------------
# Default singleton (used by GymTechnician when no encoder is injected)
# ---------------------------------------------------------------------------

ENCODER: RequestEncoder = HashEncoder()
