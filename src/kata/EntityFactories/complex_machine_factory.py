"""Factory for creating ComplexMachine instances with components from configuration."""

from typing import Any

import simpy as sp

from kata.entities.components.component import MachineComponent
from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.machines.config import MachineConfig
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
from kata.features.breakdown.simple_breakdown import (
    SimpleBreakdownProcess,
    WeibullBreakdownProcess,
)


def _component_iter(components: Any) -> list[dict[str, Any]]:
    """Normalise the ``components`` field to a list of dicts.

    The current schema stores components as a ``{component_id: config}``
    dict (matching ``MachineConfig.components``).  Older config payloads
    use a list of dicts.  We accept both.
    """
    if components is None:
        return []
    if isinstance(components, dict):
        return [dict(v) for v in components.values()]
    return [dict(v) for v in components]


def _build_breakdown_process(
    comp_config: dict[str, Any],
    dt: int,
) -> SimpleBreakdownProcess | WeibullBreakdownProcess:
    """Build a breakdown process from a component config dict."""
    breakdown_model = comp_config.get("breakdown_model", "simple")

    if breakdown_model == "weibull":
        weibull = comp_config.get("weibull_breakdown")
        if weibull is not None:
            shape = float(weibull.get("shape", 2.0))
            scale = float(weibull.get("scale", 1000.0))
        else:
            shape = float(comp_config.get("weibull_shape", 2.0))
            scale = float(comp_config.get("weibull_scale", 1000.0))
        return WeibullBreakdownProcess(shape=shape, scale=scale, dt=dt)

    simple = comp_config.get("simple_breakdown")
    degradation_rate = comp_config.get("degradation_rate", 0.001)
    idle_factor = comp_config.get("idle_degradation_factor", 0.1)
    if simple is not None:
        failure_prob_working = float(
            simple.get("failure_prob_working", degradation_rate)
        )
        failure_prob_idle = float(
            simple.get("failure_prob_idle", degradation_rate * idle_factor)
        )
    else:
        failure_prob_working = float(
            comp_config.get("failure_prob_working", degradation_rate)
        )
        failure_prob_idle = float(
            comp_config.get("failure_prob_idle", degradation_rate * idle_factor)
        )
    return SimpleBreakdownProcess(
        failure_prob_working=failure_prob_working,
        failure_prob_idle=failure_prob_idle,
    )


def create_complex_machine_from_config(
    env: sp.Environment,
    machine_id: int,
    machine_config: MachineConfig | dict[str, Any],
    input_buffer: sp.Store,
    output_buffer: sp.Store,
    tech_dispatcher: GymTechDispatcher,
    dt: int = 1,
) -> ComplexMachine:
    """Create a ComplexMachine from a configuration dict or ``MachineConfig``.

    Args:
        env: SimPy environment
        machine_id: Unique machine identifier
        machine_config: Either a ``MachineConfig`` instance or a raw dict
            with the same shape (``machine_type``, ``process_time``,
            ``components``...).
        input_buffer: Input buffer for products
        output_buffer: Output buffer for products
        tech_dispatcher: Technician dispatcher
        dt: Time step for degradation checking

    Returns:
        ComplexMachine instance with configured components

    """
    if isinstance(machine_config, MachineConfig):
        cfg = machine_config.model_dump()
    else:
        cfg = dict(machine_config)

    mtype = cfg.get("machine_type") or cfg.get("type") or "generic"
    process_time = int(cfg.get("process_time", 100))
    cfg_dt = int(cfg.get("dt", dt))

    components: list[MachineComponent] = []
    for comp_config in _component_iter(cfg.get("components")):
        component_id = comp_config.get("component_id", "unknown")
        component_type = comp_config.get("component_type", "generic")
        base_repair_time = float(comp_config.get("base_repair_time", 10.0))
        idle_factor = float(comp_config.get("idle_degradation_factor", 0.1))

        breakdown_process = _build_breakdown_process(comp_config, cfg_dt)

        components.append(
            MachineComponent(
                component_id=component_id,
                component_type=component_type,
                breakdown_process=breakdown_process,
                base_repair_time=base_repair_time,
                idle_degradation_factor=idle_factor,
            )
        )

    return ComplexMachine(
        env=env,
        machine_id=machine_id,
        mtype=mtype,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        components=components,
        process_time=process_time,
        dt=cfg_dt,
    )


def create_complex_machine_from_template(
    env: sp.Environment,
    machine_id: int,
    template_name: str,
    input_buffer: sp.Store,
    output_buffer: sp.Store,
    tech_dispatcher: GymTechDispatcher,
    dt: int = 1,
    **overrides: Any,
) -> ComplexMachine:
    """Build a ``ComplexMachine`` directly from a named machine template.

    Convenience wrapper that resolves the template via
    :func:`kata.EntityFactories.machine_factory.create_config_from_template`
    and then delegates to :func:`create_complex_machine_from_config`.
    """
    from kata.EntityFactories.machine_factory import create_config_from_template

    cfg = create_config_from_template(template_name, **overrides)
    return create_complex_machine_from_config(
        env=env,
        machine_id=machine_id,
        machine_config=cfg,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        dt=dt,
    )
