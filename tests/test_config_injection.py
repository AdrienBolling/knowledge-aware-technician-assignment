"""KATA-1 regression: the run's ``sim`` config must reach the entities.

``GymTechnician`` and ``GymTechDispatcher`` used to read a module-level
``CONFIG = get_config()`` snapshot taken at *import* time.  Every
launcher points ``KATA_CONF_PATH`` at a nonexistent file, so that
singleton was pure pydantic defaults: the per-run JSON's ``sim.*`` block
never reached the simulated entities.  Two silent consequences, live in
every training and evaluation run ever recorded:

* ``sim.technicians.travel_time`` was 10 (the pydantic default) although
  every benchmark config asks for 15;
* ``sim.repair.failure_wise_knowledge_parameters`` was ``False`` although
  every benchmark config sets it ``true`` — i.e. the per-component
  knowledge response (``ComponentConfig.min_repair_fraction`` /
  ``knowledge_sensitivity``) was dead code.

The fix injects the ``sim`` sub-model through both constructors.  These
tests build a real world (baseline.json layout, like
``tests/test_lifecycle.py``) with a *distinctive* travel time and assert
the entities actually see it.  They fail on pre-fix code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kata.core.config import KATAConfig
from kata.entities.requests.RepairRequest import RepairRequest
from kata.EntityFactories.scenario_sampler import RandomScenarioSampler
from kata.scenario import ScenarioBuilder

BASELINE = Path("run_configs/benchmark_suite/baseline.json")

# Deliberately unlike both the pydantic default (10) and the value the
# benchmark configs carry (15), so a passing assertion can only come
# from *this* config object.
TRAVEL_TIME = 17

# The global knowledge response the per-component overrides must beat.
GLOBAL_FLOOR = 0.3
GLOBAL_ALPHA = 0.15


@pytest.fixture(scope="module")
def base_cfg() -> KATAConfig:
    raw = json.loads(BASELINE.read_text())
    raw["sim"]["technicians"]["travel_time"] = TRAVEL_TIME
    raw["sim"]["repair"]["failure_wise_knowledge_parameters"] = True
    raw["sim"]["repair"]["min_repair_fraction"] = GLOBAL_FLOOR
    raw["sim"]["repair"]["knowledge_sensitivity"] = GLOBAL_ALPHA
    return KATAConfig(**raw)


@pytest.fixture(scope="module")
def world(base_cfg):
    """A real (env, dispatcher) pair built through ScenarioBuilder."""
    sampler = RandomScenarioSampler(base_cfg, base_cfg.randomized_scenario, seed=4321)
    scenario = sampler.sample_config()
    # The sampler must forward the base ``sim`` block untouched — it is
    # the only carrier of the run's simulation parameters.
    assert scenario.sim.technicians.travel_time == TRAVEL_TIME
    return ScenarioBuilder(scenario).build(), scenario


def _overriding_component(dispatcher):
    """Return ``(machine, component)`` whose component overrides both
    knowledge parameters with values different from the global ones."""
    for machine in dispatcher.machines.values():
        for comp in getattr(machine, "components", []) or []:
            floor, alpha = comp.get_knowledge_parameters()
            if (
                floor is not None
                and alpha is not None
                and not math.isclose(floor, GLOBAL_FLOOR)
                and not math.isclose(alpha, GLOBAL_ALPHA)
            ):
                return machine, comp
    pytest.skip("no component with distinctive knowledge overrides in this park")


def test_travel_time_reaches_the_technician(world):
    """(a) The technician's effective travel time is the run config's."""
    (_sim_env, dispatcher), _scenario = world
    machine = next(iter(dispatcher.machines.values()))
    tech = dispatcher.techs[0]
    assert tech.travel_time(machine) == TRAVEL_TIME

    # And it did NOT come from the process-wide singleton — which is
    # exactly what the pre-fix code read.
    from kata import get_config

    assert get_config().sim.technicians.travel_time != TRAVEL_TIME


def test_sim_config_is_injected_not_captured(world):
    """The very same ``sim`` object reaches dispatcher and every tech."""
    (_sim_env, dispatcher), scenario = world
    assert dispatcher.sim_cfg is scenario.sim
    for tech in dispatcher.techs:
        assert tech._sim_cfg is scenario.sim


def test_failure_wise_knowledge_parameters_are_live(world):
    """(b) The per-component knowledge response actually branches.

    Pre-fix, ``failure_wise_knowledge_parameters`` read ``False`` from
    the default singleton, so the multiplier always used the *global*
    (floor, alpha).  Here it must match the *component's* parameters
    exactly and differ from the global ones.
    """
    (_sim_env, dispatcher), _scenario = world
    machine, comp = _overriding_component(dispatcher)
    floor, alpha = comp.get_knowledge_parameters()

    machine.failed_component = comp
    request = RepairRequest(machine=machine, created_at=0)
    assert request.get_knowledge_parameters() == (floor, alpha)

    # Earn knowledge on this failure key so the multiplier is not
    # pinned at 1 (at k = 0 every parameterisation gives exactly 1).
    tech = dispatcher.techs[0]
    for i in range(5):
        tech.repair_finished(request, when=float(10 * (i + 1)))

    knowledge = float(tech.knowledge_grid.get_knowledge(tech.encoder.encode(request)))
    assert knowledge > 0.0

    m = tech.get_knowledge_multiplier(request)
    per_component = floor + (1.0 - floor) * math.exp(-alpha * knowledge)
    global_only = GLOBAL_FLOOR + (1.0 - GLOBAL_FLOOR) * math.exp(
        -GLOBAL_ALPHA * knowledge
    )

    assert math.isclose(m, per_component, rel_tol=1e-9)
    # The discriminating half: with the flag inert (pre-fix) this is
    # what the multiplier would have been.
    assert not math.isclose(m, global_only, rel_tol=1e-6)


def test_failure_wise_parameters_differ_across_failure_keys(world):
    """Different components resolve to different knowledge responses."""
    (_sim_env, dispatcher), _scenario = world
    seen: dict[tuple[float | None, float | None], str] = {}
    for machine in dispatcher.machines.values():
        for comp in getattr(machine, "components", []) or []:
            seen.setdefault(comp.get_knowledge_parameters(), comp.get_type())
    # The bundled machine templates carry per-component overrides; if
    # they all collapsed to one value the flag would be untestable.
    assert len(seen) > 1, f"expected several per-failure responses, got {seen}"


def test_lifecycle_hire_inherits_the_run_config(base_cfg, world):
    """The env's mid-episode hire path also injects the run's ``sim``."""
    import numpy as np

    from kata.core.config import LifecycleEventConfig
    from kata.env import KataEnv

    (_sim_env, _dispatcher), scenario = world
    gym_cfg = base_cfg.gym.model_copy(
        update={
            "max_episode_steps": 2000,
            "max_sim_time": 1500.0,
            "observation_representation": "structured",
            "lifecycle_events": [
                LifecycleEventConfig(
                    time=300.0, kind="add_technician", template="junior", count=1
                )
            ],
        }
    )
    env = KataEnv(
        scenario_factory=lambda c=scenario: ScenarioBuilder(c).build(),
        config=gym_cfg,
    )
    _obs, info = env.reset(seed=11)
    n0 = len(env.dispatcher.techs)
    for _ in range(2000):
        if info["sim_time"] >= 400.0:
            break
        mask = env._action_mask()
        action = int(np.flatnonzero(mask)[0]) if mask.any() else 0
        _obs, _r, term, trunc, info = env.step(action)
        if term or trunc:
            break

    assert len(env.dispatcher.techs) == n0 + 1
    hire = env.dispatcher.techs[-1]
    machine = next(iter(env.dispatcher.machines.values()))
    assert hire.travel_time(machine) == TRAVEL_TIME
    assert hire._sim_cfg is env.dispatcher.sim_cfg


def test_entity_modules_have_no_module_level_config():
    """(c) Meta-guard: no import-time ``get_config()`` capture may return.

    The whole defect was a module-level ``CONFIG`` snapshot; keep the
    modules free of one so it cannot silently come back.
    """
    import re

    import kata.entities.tech_dispatcher.GymTechDispatcher as disp_mod
    import kata.entities.technicians.GymTechnician as tech_mod

    for module in (tech_mod, disp_mod):
        assert not hasattr(module, "CONFIG"), (
            f"{module.__name__} re-introduced a module-level config capture"
        )
        source = Path(module.__file__).read_text()
        # Any module-level ``NAME = get_config()`` binding (the exact
        # shape of the defect).  Prose mentions of the old code in
        # docstrings are indented behind backticks and don't match.
        assert not re.search(r"^\s*\w+\s*=\s*get_config\(", source, re.M), (
            f"{module.__name__} binds get_config() again — inject the config "
            f"through the constructor instead"
        )


def test_constructors_require_sim_cfg():
    """Required-parameter loudness: no default may creep back in."""
    import inspect

    from kata.entities.tech_dispatcher.GymTechDispatcher import GymTechDispatcher
    from kata.entities.technicians.GymTechnician import GymTechnician

    for cls in (GymTechnician, GymTechDispatcher):
        param = inspect.signature(cls.__init__).parameters["sim_cfg"]
        assert param.default is inspect.Parameter.empty, (
            f"{cls.__name__}.sim_cfg gained a default — a missing injection "
            f"must fail loudly, not fall back to defaults"
        )
