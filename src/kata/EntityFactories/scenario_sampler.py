"""Per-episode random scenario sampler.

Plugged into :class:`kata.env.KataEnv` in place of the static
``ScenarioBuilder(cfg).build()`` factory.  On every call (= every
``env.reset()``) it draws a fresh ``KATAConfig`` from the configured
template pools and hands back a SimPy environment + dispatcher.

What varies per episode
-----------------------
* **Technician profiles** — drawn from
  ``RandomizedScenarioConfig.technician_templates``.  The *number* of
  technicians stays fixed so the action space (``Discrete(n_techs)``)
  is stable across episodes.
* **Machine count + templates** — drawn from
  ``RandomizedScenarioConfig.machine_templates``.  The count is
  sampled in ``[n_machines_min, n_machines_max]``.
* **Product route** — a random subset (without replacement) of the
  machine types actually present in the sampled machines, length
  sampled in ``[route_min_length, route_max_length]`` and clamped by
  the number of distinct types available.
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
        self._call_count = 0

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

        # -- Technicians --------------------------------------------------
        technicians = {}
        for i in range(cfg.n_technicians):
            template = rng.choice(cfg.technician_templates)
            name = f"{template}_{i}"
            technicians[name] = _tech_from_template(template, name=name)

        # -- Machines -----------------------------------------------------
        n_machines = rng.randint(cfg.n_machines_min, cfg.n_machines_max)
        machines = {}
        for i in range(n_machines):
            template = rng.choice(cfg.machine_templates)
            # Build a stable, unique key — ``<idx>_<template>``
            machines[f"m{i:02d}_{template}"] = _machine_from_template(template)

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
    # Callable: matches the scenario_factory contract of KataEnv
    # ------------------------------------------------------------------

    def __call__(self) -> "tuple[simpy.Environment, GymTechDispatcher]":
        # Local import to keep the heavy SimPy import out of module load.
        from kata.scenario import ScenarioBuilder

        cfg = self.sample_config()
        return ScenarioBuilder(cfg).build()


__all__ = ["RandomScenarioSampler"]
