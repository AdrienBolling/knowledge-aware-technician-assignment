"""Online running mean / variance estimator.

Used to normalise rewards (or any other scalar stream) during training
without ever holding the full history.  Implements Welford-style
parallel updates so it handles both scalar samples and batches.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RunningMeanStd:
    """Numerically-stable running mean / variance estimator.

    Parameters
    ----------
    epsilon:
        Initial sample count.  Tiny epsilon prevents a divide-by-zero
        at the first update and biases variance toward 1 early on, so
        the consumer can divide by ``sqrt(var)`` from step 1 without
        risking blow-ups.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = float(epsilon)

    def update(self, x: float | NDArray[np.floating]) -> None:
        """Fold ``x`` (scalar or 1-D array) into the running statistics."""
        arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        batch_mean = float(arr.mean())
        batch_var = float(arr.var()) if arr.size > 1 else 0.0
        batch_count = arr.size

        delta = batch_mean - self.mean
        tot = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / tot
        new_var = m2 / tot

        self.mean = new_mean
        self.var = max(new_var, 1e-8)
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


__all__ = ["RunningMeanStd"]
