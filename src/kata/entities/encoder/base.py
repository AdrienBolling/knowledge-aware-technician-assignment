"""Encoder protocol and implementations for mapping RepairRequests to grid coordinates.

The encoder converts a RepairRequest into a coordinate (tuple of ints) that can
be placed on a technician's KnowledgeGrid.  Different encoder strategies allow
experiments with how repair knowledge is spatially organised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from kata.entities.requests.RepairRequest import RepairRequest


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RequestEncoder(Protocol):
    """Encodes a RepairRequest into a grid coordinate for the KnowledgeGrid."""

    def encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Return the grid coordinate for *request*."""
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class HashEncoder:
    """Deterministic hash-based encoder.

    Maps (machine_type, component_type) pairs to a fixed grid coordinate by
    hashing.  Fast and reproducible — no fitting required.
    """

    def __init__(self, grid_shape: tuple[int, ...] = (10, 10)) -> None:
        """Initialise with the target grid shape."""
        self.grid_shape = grid_shape

    def encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Return a grid coordinate derived from hashing the failure key."""
        machine_type: str = getattr(request.machine, "mtype", "unknown")
        comp_info = request.get_failed_component_info()
        component_type: str = comp_info["component_type"] if comp_info else "none"

        key = f"{machine_type}::{component_type}"
        h = hash(key)
        return np.array([abs(h) % dim for dim in self.grid_shape], dtype=np.intp)


class LookupEncoder:
    """Table-based encoder with explicit (machine_type, component_type) -> coord mapping.

    Useful when you want full control over which grid cell each failure type
    maps to.  Unknown keys fall back to a deterministic hash.
    """

    def __init__(
        self,
        lookup: dict[tuple[str, str], tuple[int, ...]] | None = None,
        grid_shape: tuple[int, ...] = (10, 10),
    ) -> None:
        """Initialise with an optional explicit lookup table."""
        self.lookup: dict[tuple[str, str], tuple[int, ...]] = lookup or {}
        self._fallback = HashEncoder(grid_shape)

    def encode(self, request: RepairRequest) -> NDArray[np.intp]:
        """Return a grid coordinate, preferring the lookup table."""
        machine_type: str = getattr(request.machine, "mtype", "unknown")
        comp_info = request.get_failed_component_info()
        component_type: str = comp_info["component_type"] if comp_info else "none"

        key = (machine_type, component_type)
        if key in self.lookup:
            return np.array(self.lookup[key], dtype=np.intp)
        return self._fallback.encode(request)


# ---------------------------------------------------------------------------
# Default singleton (used by GymTechnician when no encoder is injected)
# ---------------------------------------------------------------------------

ENCODER: RequestEncoder = HashEncoder()
