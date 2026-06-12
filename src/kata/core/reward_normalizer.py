"""Per-component running-stats standardiser for reward streams.

A single :class:`RewardNormalizer` owns one
:class:`agents.networks.running_stats.RunningMeanStd` per reward
component name and applies full standardisation — ``(raw - mean) / std``
— before the value is multiplied by its coefficient.  This puts
heterogeneously-scaled components on a comparable footing
(e.g. a ``[-1, 0]`` busy-tech penalty and a ``[0, 100]``
knowledge-increment delta both end up unit-variance and zero-mean
after enough samples), so coefficients act as *relative weights*
rather than absolute magnitudes.

The mean-shift turns the per-component reward into a signed
"deviation from the typical magnitude": a value of zero means "as
expected", positive means "better than average", negative means
"worse than average".  This is the same signal regime PPO's
advantage normalisation operates in, just applied per-component
rather than to the scalarised return.

Lifecycle
---------
* Stats accumulate across episodes within a single :class:`KataEnv`
  instance (see ``env.reset()`` — the normaliser is *not* reset).
* Eval-time freezing is the consumer's responsibility.  Call
  :meth:`freeze` after a short warmup if you want every component to
  be normalised against the *warmup-time* statistics.  This is what
  the benchmark notebooks do.
"""

from __future__ import annotations

from typing import Mapping

from agents.networks.running_stats import RunningMeanStd


class RewardNormalizer:
    """One :class:`RunningMeanStd` per named reward component.

    Parameters
    ----------
    epsilon:
        Seeded into each new component's running-stats so the first
        few normalisations don't divide by a near-zero std.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon: float = float(epsilon)
        self._stats: dict[str, RunningMeanStd] = {}
        self._frozen: bool = False

    # ------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Stop updating statistics; subsequent normalises read frozen std."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume updating statistics from the next call."""
        self._frozen = False

    # ------------------------------------------------------------------
    def normalize(self, name: str, raw: float) -> float:
        """Standardise ``raw`` against the running stats of ``name``.

        Returns ``(raw - mean) / std`` using the per-component running
        :class:`RunningMeanStd`.  Updates the running stats first
        (unless frozen) so the very first sample sees its own
        influence in the divisor.  A bias of ``epsilon`` keeps the
        divisor away from zero in the early-training regime; the
        ``RunningMeanStd`` initial variance of 1.0 ensures the very
        first normalisation is well-defined.
        """
        rms = self._stats.get(name)
        if rms is None:
            rms = RunningMeanStd(epsilon=self.epsilon)
            self._stats[name] = rms
        if not self._frozen:
            rms.update(float(raw))
        std = max(rms.std, 1e-8)
        return (float(raw) - rms.mean) / std

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, dict[str, float]]:
        """Serialisable snapshot of every component's stats."""
        return {
            name: rms.state_dict() for name, rms in self._stats.items()
        }

    def load_state_dict(self, state: Mapping[str, Mapping[str, float]]) -> None:
        """Restore stats from :meth:`state_dict` output."""
        self._stats.clear()
        for name, sub in state.items():
            rms = RunningMeanStd(epsilon=self.epsilon)
            rms.load_state_dict(dict(sub))
            self._stats[name] = rms

    # ------------------------------------------------------------------
    def has_component(self, name: str) -> bool:
        return name in self._stats

    def std(self, name: str) -> float:
        """Current running std for ``name`` (or 1.0 if unseen)."""
        rms = self._stats.get(name)
        return float(rms.std) if rms is not None else 1.0


__all__ = ["RewardNormalizer"]
