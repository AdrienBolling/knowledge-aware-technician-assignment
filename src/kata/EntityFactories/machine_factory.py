"""Template-based machine factory.

This module loads machine *templates* — named ``MachineConfig`` blueprints —
from JSON resources and provides convenience helpers to:

* List the available templates.
* Resolve a template name (with optional inline overrides) into a
  ``MachineConfig`` instance.
* Build a ``machines`` dict (the format expected by ``KATAConfig``) from
  a list of ``(template_name, instance_id)`` pairs.

The default templates ship with the package under
``src/kata/resources/templates/machine_templates.json``.  At runtime new
templates can be registered via ``register_template`` for one-off tweaks
without touching the resource file.
"""

from __future__ import annotations

import importlib.resources as res
import json
from typing import Any

from kata.entities.machines.config import MachineConfig

_TEMPLATE_RESOURCE = ("kata.resources.templates", "machine_templates.json")


def _load_templates_from_resource() -> dict[str, dict[str, Any]]:
    pkg, name = _TEMPLATE_RESOURCE
    with res.files(pkg).joinpath(name).open("r") as f:
        return json.load(f)


# Registry populated lazily on first access.  Keys are template names,
# values are raw config dicts (so callers may apply overrides before
# constructing a ``MachineConfig``).
_machine_templates: dict[str, dict[str, Any]] = _load_templates_from_resource()


def list_templates() -> list[str]:
    """Return the names of all currently registered machine templates."""
    return sorted(_machine_templates.keys())


def get_template(name: str) -> dict[str, Any]:
    """Return a deep-copyable raw template dict by name."""
    if name not in _machine_templates:
        msg = (
            f"Unknown machine template: {name!r}. "
            f"Available: {list_templates()}"
        )
        raise KeyError(msg)
    # Return a shallow copy of the top-level keys; nested dicts (e.g.
    # ``components``) are also copied so callers can mutate freely.
    return {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in _machine_templates[name].items()
    }


def register_template(name: str, config: dict[str, Any] | MachineConfig) -> None:
    """Register a new template at runtime (does not modify the JSON file)."""
    if isinstance(config, MachineConfig):
        config = config.model_dump()
    _machine_templates[name] = dict(config)


def create_config_from_template(
    template_name: str,
    **overrides: Any,
) -> MachineConfig:
    """Instantiate a ``MachineConfig`` from a template, with optional overrides.

    Example
    -------
    >>> cfg = create_config_from_template("cnc_weibull", process_time=250)
    >>> cfg.process_time
    250
    """
    raw = get_template(template_name)
    raw.update(overrides)
    return MachineConfig(**raw)


def create_machines_dict(
    instances: list[tuple[str, str]] | dict[str, str],
) -> dict[str, MachineConfig]:
    """Build a ``machines`` dict (id -> ``MachineConfig``) from templates.

    Parameters
    ----------
    instances:
        Either a list of ``(template_name, instance_id)`` pairs or a
        mapping ``{instance_id: template_name}``.  ``instance_id`` is
        the key the resulting dict will use (and matches the format of
        ``KATAConfig.machines``).

    Example
    -------
    >>> machines = create_machines_dict(
    ...     [("cnc_weibull", "cnc_1"), ("cnc_weibull", "cnc_2")]
    ... )
    >>> sorted(machines)
    ['cnc_1', 'cnc_2']
    """
    if isinstance(instances, dict):
        items = [(tpl, mid) for mid, tpl in instances.items()]
    else:
        items = list(instances)

    out: dict[str, MachineConfig] = {}
    for template_name, instance_id in items:
        out[instance_id] = create_config_from_template(template_name)
    return out


__all__ = [
    "create_config_from_template",
    "create_machines_dict",
    "get_template",
    "list_templates",
    "register_template",
]
