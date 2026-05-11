"""Factories that build entity configs / instances from named templates.

Heavy SimPy-dependent factories (e.g.
:func:`create_complex_machine_from_template`) are *not* imported eagerly
here to avoid circular-import issues with :mod:`kata.core.config`.
Import them directly from their submodule when needed::

    from kata.EntityFactories.complex_machine_factory import (
        create_complex_machine_from_template,
    )
"""

from kata.EntityFactories.machine_factory import (
    create_config_from_template as create_machine_config,
)
from kata.EntityFactories.machine_factory import (
    create_machines_dict,
    list_templates as list_machine_templates,
    register_template as register_machine_template,
)
from kata.EntityFactories.technician_factory import (
    create_config_from_template as create_technician_config,
)
from kata.EntityFactories.technician_factory import (
    create_technicians_dict,
    list_templates as list_technician_templates,
    register_template as register_technician_template,
)


def __getattr__(name: str):  # pragma: no cover - thin lazy proxy
    """Lazy access to SimPy-dependent helpers, avoiding circular imports."""
    if name == "create_complex_machine_from_config":
        from kata.EntityFactories.complex_machine_factory import (
            create_complex_machine_from_config,
        )

        return create_complex_machine_from_config
    if name == "create_complex_machine_from_template":
        from kata.EntityFactories.complex_machine_factory import (
            create_complex_machine_from_template,
        )

        return create_complex_machine_from_template
    if name == "SyntheticTicketFactory":
        from kata.EntityFactories.synthetic_ticket_factory import (
            SyntheticTicketFactory,
        )

        return SyntheticTicketFactory
    if name == "RandomScenarioSampler":
        from kata.EntityFactories.scenario_sampler import RandomScenarioSampler

        return RandomScenarioSampler
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "RandomScenarioSampler",
    "SyntheticTicketFactory",
    "create_complex_machine_from_config",
    "create_complex_machine_from_template",
    "create_machine_config",
    "create_machines_dict",
    "create_technician_config",
    "create_technicians_dict",
    "list_machine_templates",
    "list_technician_templates",
    "register_machine_template",
    "register_technician_template",
]
