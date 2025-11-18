import importlib.resources as res
import json
from functools import partial
from typing import Callable

from kata.core.common import ArgType
from kata.entities.machines.machine import Machine

"""
This file contains a registry of the registered machine pre-confingurations available in kata/resources/templates/machine_tempaltes.json
It is also possible to register new machine configurations by using the "register_machine" function to temporarily add it to the registery during runtime.
Permanent registration requires adding the new configuration file to the machine_templates folder.
"""
machine_templates = {}
# Load the machine templates from the resources
ref = res.files("kata.resources.templates") / "machine_templates.json"
with res.as_file(path=ref) as file_path:
    with file_path.open("r") as f:
        machine_templates: dict[str, dict[str, ArgType]] = json.load(f)


machine_registry: dict[str, Callable[..., Machine]] = {
    name: partial(Machine, **machine_config)
    for name, machine_config in machine_templates.items()
}
