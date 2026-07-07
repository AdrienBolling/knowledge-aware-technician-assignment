"""Random scenario sampler with optional multi-episode reuse.

Plugged into :class:`kata.env.KataEnv` in place of the static
``ScenarioBuilder(cfg).build()`` factory.  On each call (= each
``env.reset()``) it returns a SimPy environment + dispatcher built
from a ``KATAConfig`` sampled from the configured template pools.

What varies between sampled scenarios
-------------------------------------
* **Technician profiles** — drawn from
  ``RandomizedScenarioConfig.technician_templates``.
* **Technician count** — fixed (``n_technicians``), or sampled in a
  range, or derived from ``n_machines`` via a coherence ratio.  See
  ``RandomizedScenarioConfig`` for which mode applies.
* **Machine count + templates** — drawn from
  ``RandomizedScenarioConfig.machine_templates``.  The count is
  sampled in ``[n_machines_min, n_machines_max]``.
* **Product route** — a random subset (without replacement) of the
  machine types actually present in the sampled machines, length
  sampled in ``[route_min_length, route_max_length]`` and clamped by
  the number of distinct types available.

Episodes per scenario
---------------------
``RandomizedScenarioConfig.episodes_per_scenario`` (``k``) controls
how often a fresh scenario is drawn:

* ``k = 1`` (default) — every ``env.reset()`` draws a fresh scenario,
  matching the original behaviour.
* ``k > 1``           — the same sampled scenario is reused for ``k``
  consecutive resets, then a new one is drawn.  Useful for letting an
  agent settle on a single layout for several episodes while still
  training across many layouts overall.

Each reuse still builds a *fresh* SimPy environment — only the
structural ``KATAConfig`` (machines, technicians, routes) is held
constant inside one ``k``-block.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from kata.EntityFactories.machine_factory import (
    create_config_from_template as _machine_from_template,
)
from kata.EntityFactories.machine_factory import (
    list_templates as _list_machine_templates,
)
from kata.EntityFactories.technician_factory import (
    create_config_from_template as _tech_from_template,
)
from kata.EntityFactories.technician_factory import (
    list_templates as _list_technician_templates,
)

if TYPE_CHECKING:
    import simpy

    from kata.core.config import KATAConfig, RandomizedScenarioConfig
    from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher


class RandomScenarioSampler:
    """Callable that builds a randomised SimPy scenario per call."""

    def __init__(
        self,
        base_cfg: "KATAConfig",
        rcfg: "RandomizedScenarioConfig",
        seed: int | None = None,
    ) -> None:
        self._base = base_cfg
        self._cfg = rcfg
        self._rng = random.Random(seed if seed is not None else rcfg.seed)
        # ``_call_count`` counts ``sample_config()`` invocations
        # (= scenario re-samples).  ``_cached_config`` holds the most
        # recently sampled ``KATAConfig`` and ``_scenario_age`` counts
        # ``__call__`` invocations since that sample so the cache can
        # be reused for ``episodes_per_scenario`` consecutive builds.
        self._call_count = 0
        self._cached_config: KATAConfig | None = None
        self._scenario_age = 0

        # Validate template pools at construction time so misconfigured
        # runs fail loudly before training starts.
        avail_machines = set(_list_machine_templates())
        unknown_m = [t for t in rcfg.machine_templates if t not in avail_machines]
        if unknown_m:
            msg = (
                f"Unknown machine template(s) in randomized_scenario.machine_templates: "
                f"{unknown_m}.  Known: {sorted(avail_machines)}"
            )
            raise ValueError(msg)

        avail_techs = set(_list_technician_templates())
        unknown_t = [t for t in rcfg.technician_templates if t not in avail_techs]
        if unknown_t:
            msg = (
                f"Unknown technician template(s) in randomized_scenario.technician_templates: "
                f"{unknown_t}.  Known: {sorted(avail_techs)}"
            )
            raise ValueError(msg)

        if rcfg.n_machines_max < rcfg.n_machines_min:
            msg = (
                f"randomized_scenario.n_machines_max ({rcfg.n_machines_max}) must be "
                f">= n_machines_min ({rcfg.n_machines_min})"
            )
            raise ValueError(msg)
        if rcfg.route_max_length < rcfg.route_min_length:
            msg = (
                f"randomized_scenario.route_max_length ({rcfg.route_max_length}) must be "
                f">= route_min_length ({rcfg.route_min_length})"
            )
            raise ValueError(msg)

        # -- Variable-fleet validation -----------------------------------
        # Both bounds must be set together — half-set range is ambiguous.
        if (rcfg.n_technicians_min is None) ^ (rcfg.n_technicians_max is None):
            msg = (
                "randomized_scenario.n_technicians_min and "
                "n_technicians_max must be set together (or both left "
                "unset to keep the fixed-fleet behaviour)"
            )
            raise ValueError(msg)
        if (rcfg.n_technicians_min is not None
                and rcfg.n_technicians_max is not None
                and rcfg.n_technicians_max < rcfg.n_technicians_min):
            msg = (
                f"randomized_scenario.n_technicians_max ({rcfg.n_technicians_max}) "
                f"must be >= n_technicians_min ({rcfg.n_technicians_min})"
            )
            raise ValueError(msg)

        if (rcfg.techs_per_machine_min is None) ^ (rcfg.techs_per_machine_max is None):
            msg = (
                "randomized_scenario.techs_per_machine_min and "
                "techs_per_machine_max must be set together"
            )
            raise ValueError(msg)
        if (rcfg.techs_per_machine_min is not None
                and rcfg.techs_per_machine_max is not None
                and rcfg.techs_per_machine_max < rcfg.techs_per_machine_min):
            msg = (
                f"randomized_scenario.techs_per_machine_max "
                f"({rcfg.techs_per_machine_max}) must be >= "
                f"techs_per_machine_min ({rcfg.techs_per_machine_min})"
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Helpers used by the runner to pre-populate vocabularies
    # ------------------------------------------------------------------

    def all_machine_types(self) -> list[str]:
        """Every machine_type any drawn machine could ever have."""
        return sorted({_machine_from_template(t).machine_type for t in self._cfg.machine_templates})

    def all_component_types(self) -> list[str]:
        """Every component_type any drawn machine could ever expose."""
        out: set[str] = set()
        for t in self._cfg.machine_templates:
            for c in _machine_from_template(t).components.values():
                out.add(c.component_type)
        return sorted(out)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_config(self) -> "KATAConfig":
        from kata.core.config import KATAConfig, ProductConfig

        rng = self._rng
        cfg = self._cfg

        # -- Machines (sampled first so the tech count can scale with them) -
        n_machines = rng.randint(cfg.n_machines_min, cfg.n_machines_max)
        machines = {}
        for i in range(n_machines):
            template = rng.choice(cfg.machine_templates)
            # Build a stable, unique key — ``<idx>_<template>``
            machines[f"m{i:02d}_{template}"] = _machine_from_template(template)

        # -- Technicians (count via ratio / range / fixed, in that order) ---
        n_techs = self._sample_n_technicians(n_machines)
        technicians = {}
        for i in range(n_techs):
            template = rng.choice(cfg.technician_templates)
            name = f"{template}_{i}"
            technicians[name] = _tech_from_template(template, name=name)

        # -- Product route ------------------------------------------------
        types_present = sorted({m.machine_type for m in machines.values()})
        if not types_present:
            types_present = ["generic"]
        upper = min(cfg.route_max_length, len(types_present))
        lower = min(cfg.route_min_length, upper)
        route_len = rng.randint(max(1, lower), max(1, upper))
        route = rng.sample(types_present, k=route_len)

        products = {
            "widget": ProductConfig(product_type="widget", route=route),
        }

        # Compose a fresh KATAConfig keeping the base sim / gym / reward
        # settings but with the freshly sampled fleet & route.
        new_cfg = self._base.model_copy(
            update={
                "technicians": technicians,
                "machines": machines,
                "products": products,
            }
        )
        self._call_count += 1
        return new_cfg

    # ------------------------------------------------------------------
    # Internal: technician-count sampling
    # ------------------------------------------------------------------

    def _sample_n_technicians(self, n_machines: int) -> int:
        """Pick the number of technicians for the current scenario.

        Precedence (highest first):

        * **coherence-ratio** — both ``techs_per_machine_min/max`` set.
          ``round(n_machines * uniform(min, max))``.  Result is clamped
          to ``[n_technicians_min, n_technicians_max]`` when those
          bounds are also set, and to ``>= 1`` so an empty fleet is
          impossible.
        * **range** — both ``n_technicians_min/max`` set.  Uniform
          random integer in the closed range.
        * **fixed** (default) — ``cfg.n_technicians``.
        """
        cfg = self._cfg
        rng = self._rng

        if cfg.techs_per_machine_min is not None and cfg.techs_per_machine_max is not None:
            ratio = rng.uniform(
                float(cfg.techs_per_machine_min),
                float(cfg.techs_per_machine_max),
            )
            n_techs = int(round(n_machines * ratio))
            lo = cfg.n_technicians_min if cfg.n_technicians_min is not None else 1
            hi = cfg.n_technicians_max  # may be None — only upper-clamp when set
            n_techs = max(int(lo), n_techs)
            if hi is not None:
                n_techs = min(int(hi), n_techs)
            return max(1, n_techs)

        if cfg.n_technicians_min is not None and cfg.n_technicians_max is not None:
            return rng.randint(int(cfg.n_technicians_min), int(cfg.n_technicians_max))

        return int(cfg.n_technicians)

    # ------------------------------------------------------------------
    # Cache control (used by callable + tests)
    # ------------------------------------------------------------------

    def reset_scenario_cache(self) -> None:
        """Forget the cached scenario so the next ``__call__`` re-samples."""
        self._cached_config = None
        self._scenario_age = 0

    # ------------------------------------------------------------------
    # Callable: matches the scenario_factory contract of KataEnv
    # ------------------------------------------------------------------

    def __call__(self) -> "tuple[simpy.Environment, GymTechDispatcher]":
        # Local import to keep the heavy SimPy import out of module load.
        from kata.scenario import ScenarioBuilder

        # Reuse the cached config for ``episodes_per_scenario`` consecutive
        # builds before drawing a fresh one.  Each build still produces a
        # brand-new SimPy environment — only the structural config (fleet,
        # routes) is held constant inside one ``k``-block.
        k = max(1, int(self._cfg.episodes_per_scenario))
        if self._cached_config is None or self._scenario_age >= k:
            self._cached_config = self.sample_config()
            self._scenario_age = 0
        self._scenario_age += 1
        return ScenarioBuilder(self._cached_config).build()


__all__ = ["RandomScenarioSampler"]
