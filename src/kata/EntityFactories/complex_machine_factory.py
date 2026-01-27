"""
Factory for creating ComplexMachine instances with components from configuration.
"""
from typing import List
import simpy as sp

from kata.entities.machines.complex_machine import ComplexMachine
from kata.entities.components.component import MachineComponent
from kata.features.breakdown.simple_breakdown import SimpleBreakdownProcess, WeibullBreakdownProcess
from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher


def create_complex_machine_from_config(
    env: sp.Environment,
    machine_id: int,
    machine_config: dict,
    input_buffer: sp.Store,
    output_buffer: sp.Store,
    tech_dispatcher: GymTechDispatcher,
    dt: int = 1,
) -> ComplexMachine:
    """
    Create a ComplexMachine from a configuration dictionary.
    
    Args:
        env: SimPy environment
        machine_id: Unique machine identifier
        machine_config: Configuration dict with machine parameters and components
        input_buffer: Input buffer for products
        output_buffer: Output buffer for products
        tech_dispatcher: Technician dispatcher
        dt: Time step for degradation checking
        
    Returns:
        ComplexMachine instance with configured components
    """
    # Extract basic machine parameters
    mtype = machine_config.get("type", "generic")
    process_time = machine_config.get("process_time", 100)
    
    # Build components from configuration
    components = []
    component_configs = machine_config.get("components", [])
    
    for comp_config in component_configs:
        component_id = comp_config.get("component_id", "unknown")
        component_type = comp_config.get("component_type", "generic")
        base_repair_time = comp_config.get("base_repair_time", 10.0)
        degradation_rate = comp_config.get("degradation_rate", 0.001)
        idle_factor = comp_config.get("idle_degradation_factor", 0.1)
        
        # Determine breakdown model
        breakdown_model = comp_config.get("breakdown_model", "simple")
        if breakdown_model == "weibull":
            weibull_shape = comp_config.get("weibull_shape", 2.0)
            weibull_scale = comp_config.get("weibull_scale", 1000.0)
            breakdown_process = WeibullBreakdownProcess(
                shape=weibull_shape,
                scale=weibull_scale,
                dt=dt,
            )
        else:
            # Use simple breakdown model with degradation rate
            failure_prob_working = comp_config.get("failure_prob_working", degradation_rate)
            failure_prob_idle = comp_config.get("failure_prob_idle", degradation_rate * idle_factor)
            breakdown_process = SimpleBreakdownProcess(
                failure_prob_working=failure_prob_working,
                failure_prob_idle=failure_prob_idle,
            )
        
        # Create the component
        component = MachineComponent(
            component_id=component_id,
            component_type=component_type,
            breakdown_process=breakdown_process,
            base_repair_time=base_repair_time,
            idle_degradation_factor=idle_factor,
        )
        components.append(component)
    
    # Create and return the ComplexMachine
    return ComplexMachine(
        env=env,
        machine_id=machine_id,
        mtype=mtype,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        tech_dispatcher=tech_dispatcher,
        components=components,
        process_time=process_time,
        dt=dt,
    )
