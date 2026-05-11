"""Template-based technician factory.

Mirrors :mod:`kata.EntityFactories.machine_factory` but for technician
profiles (``TechnicianConfig``).  Default profiles ship with the
package under ``src/kata/resources/templates/technician_templates.json``
and represent common skill-level archetypes (``expert``, ``senior``,
``generalist``, ``junior``, ``trainee``, …).
"""

from __future__ import annotations

import importlib.resources as res
import json
from typing import Any

from kata.entities.technicians.config import TechnicianConfig

_TEMPLATE_RESOURCE = ("kata.resources.templates", "technician_templates.json")


def _load_templates_from_resource() -> dict[str, dict[str, Any]]:
    pkg, name = _TEMPLATE_RESOURCE
    with res.files(pkg).joinpath(name).open("r") as f:
        return json.load(f)


_technician_templates: dict[str, dict[str, Any]] = _load_templates_from_resource()


def list_templates() -> list[str]:
    """Return the names of all currently registered technician templates."""
    return sorted(_technician_templates.keys())


def get_template(name: str) -> dict[str, Any]:
    """Return a copy of the raw template dict by name."""
    if name not in _technician_templates:
        msg = (
            f"Unknown technician template: {name!r}. "
            f"Available: {list_templates()}"
        )
        raise KeyError(msg)
    return {
        k: (list(v) if isinstance(v, list) else v)
        for k, v in _technician_templates[name].items()
    }


def register_template(name: str, config: dict[str, Any] | TechnicianConfig) -> None:
    """Register a new technician profile at runtime."""
    if isinstance(config, TechnicianConfig):
        config = config.model_dump()
    _technician_templates[name] = dict(config)


def create_config_from_template(
    template_name: str,
    **overrides: Any,
) -> TechnicianConfig:
    """Instantiate a ``TechnicianConfig`` from a template with optional overrides.

    Example
    -------
    >>> cfg = create_config_from_template("expert", name="alice")
    >>> cfg.name
    'alice'
    """
    raw = get_template(template_name)
    raw.update(overrides)
    return TechnicianConfig(**raw)


def create_technicians_dict(
    instances: list[tuple[str, str]] | dict[str, str],
) -> dict[str, TechnicianConfig]:
    """Build a ``technicians`` dict (id -> ``TechnicianConfig``) from templates.

    Parameters
    ----------
    instances:
        Either a list of ``(template_name, instance_id)`` pairs or a
        mapping ``{instance_id: template_name}``.  ``instance_id`` is
        used as the dict key (matches the format of
        ``KATAConfig.technicians``); the per-instance ``name`` field is
        also set to ``instance_id`` so the resulting techs are
        distinguishable.

    Example
    -------
    >>> techs = create_technicians_dict(
    ...     [("expert", "expert_1"), ("junior", "junior_1")]
    ... )
    >>> sorted(techs)
    ['expert_1', 'junior_1']
    """
    if isinstance(instances, dict):
        items = [(tpl, tid) for tid, tpl in instances.items()]
    else:
        items = list(instances)

    out: dict[str, TechnicianConfig] = {}
    for template_name, instance_id in items:
        out[instance_id] = create_config_from_template(
            template_name,
            name=instance_id,
        )
    return out


__all__ = [
    "create_config_from_template",
    "create_technicians_dict",
    "get_template",
    "list_templates",
    "register_template",
]
