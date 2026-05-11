"""Backwards-compatible re-exports of the machine template registry.

The single source of truth for machine templates is
:mod:`kata.EntityFactories.machine_factory`.  This module simply
re-exposes the registry under its historical name so older callers
keep working.
"""

from kata.EntityFactories.machine_factory import (
    _machine_templates as machine_templates,
    create_config_from_template,
    list_templates,
    register_template,
)

# ``machine_registry`` historically mapped a template name to a callable
# that returns a ``MachineConfig``.  The new factory exposes the same
# behaviour through ``create_config_from_template``; keep the dict for
# code that iterates over it.
machine_registry: dict[str, callable] = {
    name: (lambda n=name: create_config_from_template(n))
    for name in machine_templates
}

__all__ = [
    "create_config_from_template",
    "list_templates",
    "machine_registry",
    "machine_templates",
    "register_template",
]
