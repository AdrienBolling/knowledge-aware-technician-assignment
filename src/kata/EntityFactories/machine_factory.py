"""
This file describes the numpy template corresponding to a machine.

It also implements several machine templates and a MachineFactory class that can be used to create machines according to the specified template.
"""

"""
Numpy template for a machine entity.
machine = np.array([
    0, # machine status (-1: maintenance, 0: idle, 1: running)
    50, # production rate (production steps per timestep)
    5, # in_buffer (number of items in the input buffer)
    10, # out_buffer (number of items in the output buffer)
    75, # current production step (when <0, the current product is finished)
    0.7, # current failure probability (0-1)
])

Each machine is an object, where the attributes are used to store the init parameters of the machine. In addition there's a method used to get an initialized numpy array for the machine.
The numpy array will be used to store the state of the machine for simulation purposes, the class will only be used for initilization and attributes reading purposes.
"""

import numpy as np
import importlib.resources as res
import json

class Machine:
    def __init__(
        self,
        prod_rate: int,
        name: str = "Machine",
        brand: str = "Generic",
        type: str = "Generic",
        components: list = None,
        weibull_k: float = 1.0,
        weibull_lambda: float = 1.0,
    ):
        self.prod_rate = prod_rate
        self.name = name
        self.brand = brand
        self.type = type
        self.components = components if components is not None else []
        self.weibull_k = weibull_k  # Weibull shape parameter for failure distribution
        self.weibull_lambda = weibull_lambda
        
    def get_state_numpy_array(self) -> np.ndarray:
        """
        Returns a numpy array with the initial state of the machine.
        """
        return np.array([
            0,  # machine status (-1: maintenance, 0: idle, 1: running)
            self.prod_rate,  # production rate (production steps per timestep)
            0,  # in_buffer (number of items in the input buffer)
            0,  # out_buffer (number of items in the output buffer)
            -1,  # current production step (when <0, the current product is finished)
            0.0,  # current failure probability (0-1)
        ])
        
    def get_parameters_numpy_array(self) -> np.ndarray:
        """
        Returns a numpy array with the parameters of the machine.
        """
        return np.array([
            self.weibull_k,  # Weibull shape parameter
            self.weibull_lambda,  # Weibull scale parameter
        ])

class MachineFactory:
    @staticmethod
    def create_machine(
        prod_rate: int,
        name: str = "Machine",
        brand: str = "Generic",
        type: str = "Generic",
        components: list = None,
        weibull_k: float = 1.0,
        weibull_lambda: float = 1.0,
    ) -> Machine:
        """
        Creates a machine with the specified parameters.
        """
        return Machine(prod_rate, name, brand, type, components, weibull_k, weibull_lambda)
    

    @staticmethod
    def create_machine_from_template(
        template_name: str,
    ) -> Machine:
        """
        Creates a machine from a template.
        A template is a predefined set of parameters for a machine.
        They are stored under the template json file 
        """
        
        with res.files("kata.resources").joinpath("machine_templates.json").open("r") as f:
            template = json.load(f)
            
        if template_name not in template:
            raise ValueError(f"Template '{template_name}' not found in machine templates.")
        
        params = template[template_name]
        try:
            machine = MachineFactory.create_machine(**params)
        except TypeError as e:
            raise ValueError(f"Error creating machine from template '{template_name}': {e}")
        
        return machine
    
    @staticmethod
    def create_machines_from_templates(
        template_names: list,
    ) -> list:
        """
        Creates multiple machines from a list of templates.
        """
        machines = []
        for template_name in template_names:
            machine = MachineFactory.create_machine_from_template(template_name)
            machines.append(machine)
        return machines