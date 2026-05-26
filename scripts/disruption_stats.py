#!/usr/bin/env python
"""Measure actual disruption behaviour against the configured intent.

For each named disruption type in an env config, this script reports:

* Configured intent (rate / coefficient / interval) and the *expected*
  number of events per technician over the simulated horizon.
* Observed firing counts, mean inter-arrival time, and total downtime
  per technician.
* Aggregate fleet downtime as a fraction of episode time --- a quick
  health check that disruptions aren't dominating the simulation.

Usage::

    python scripts/disruption_stats.py \\
        --env run_configs/factory_long.json \\
        --episodes 3 \\
        --max-sim-time 100000 \\
        --policy least_busy

If ``--max-sim-time`` is omitted the script uses whatever the env
config defines.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "src"))


@contextmanager
def quiet_stdout() -> Any:
    """Suppress noisy SimPy prints from machines / routers / sources."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--env", required=True, help="Path to env JSON (KATAConfig).")
    p.add_argument("--episodes", type=int, default=3, help="Episodes to average over.")
    p.add_argument(
        "--max-sim-time",
        type=float,
        default=None,
        help="Override env's max_sim_time for the test.",
    )
    p.add_argument(
        "--policy",
        default="least_busy",
        choices=["random", "round_robin", "least_busy", "least_fatigued", "shortest_queue"],
        help="Heuristic agent used to drive the env.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_env(env_path: Path, max_sim_time: float | None, seed: int):
    from kata import get_config
    from kata.core.config import KATAConfig
    from kata.env import KataEnv
    from kata.EntityFactories import RandomScenarioSampler
    from kata.scenario import ScenarioBuilder

    os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"
    cfg = KATAConfig(**json.loads(env_path.read_text()))
    if max_sim_time is not None:
        cfg.gym = cfg.gym.model_copy(update={"max_sim_time": float(max_sim_time)})
    # ---------------------------------------------------------------
    # IMPORTANT: the simulator's module-level ``CONFIG = get_config()``
    # imports cache the singleton at import time.  Any code path that
    # reads ``CONFIG.sim.…`` (e.g. ``GymTechnician`` constructor,
    # ``GymTechDispatcher`` disruption spawn loop) sees whatever the
    # singleton points to AT THAT MOMENT, not the local ``cfg`` we
    # just loaded.  Two failure modes if you forget:
    #   1. Mutating the singleton with ``cached.sim = cfg.sim`` works
    #      (in-place attribute replacement); ``model_copy(...)`` does
    #      NOT because it returns a new object the cached singleton
    #      doesn't reference.
    #   2. Forgetting to sync entirely silently uses defaults.
    # If you ever refactor away the singleton, delete this whole
    # block and pass ``cfg`` explicitly through the simulator's
    # construction APIs.
    cached = get_config()
    cached.sim = cfg.sim
    cached.gym = cfg.gym

    rcfg = cfg.randomized_scenario
    if rcfg.enabled:
        sampler = RandomScenarioSampler(cfg, rcfg, seed=seed)

        def factory():
            return ScenarioBuilder(sampler.sample_config()).build()
    else:
        def factory():
            return ScenarioBuilder(cfg).build()

    with quiet_stdout():
        env = KataEnv(scenario_factory=factory, config=cfg.gym)
    return env, cfg


def expected_per_tech(cfg: Any, horizon: float) -> dict[str, dict[str, float]]:
    """Per-type expected counts and mean durations from the config."""
    out: dict[str, dict[str, float]] = {}
    for name, dis in cfg.sim.disruptions.dis_dict.items():
        if dis.trigger == "random":
            exp_count = (dis.rate or 0.0) * horizon
            note = f"rate={dis.rate:g} → mean interval {1.0 / max(dis.rate, 1e-30):.1f}"
        elif dis.trigger == "fatigue":
            # Bound assuming F ∈ [0, 1] on average — at the extremes
            # the actual rate ranges between 0 (idle tech) and
            # coef * poll per poll (fully exhausted).
            f_min = 0.0
            f_max = (dis.fatigue_coefficient or 0.0) * dis.poll_interval
            exp_count = (
                f"≤ {(f_max / dis.poll_interval) * horizon:.1f} "
                f"(if F=1 throughout)"
            )
            note = (
                f"coef={dis.fatigue_coefficient:g}, poll={dis.poll_interval:g} → "
                f"per-poll p ∈ [{f_min:.4f}, {f_max:.4f}]"
            )
        elif dis.trigger == "periodic":
            exp_count = horizon / max(dis.interval, 1e-9)
            note = (
                f"interval={dis.interval:g} (±{dis.jitter:g} jitter) → "
                f"≈ {exp_count:.1f} firings in horizon"
            )
        else:
            exp_count = 0.0
            note = ""
        out[name] = {
            "trigger": dis.trigger,
            "expected_count_per_tech": exp_count,
            "preemptive": dis.preemptive,
            "duration_mu": dis.duration_mu,
            "duration_sig": dis.duration_sig,
            "note": note,
        }
    return out


def run_episode(env, policy, seed):
    with quiet_stdout():
        obs, _ = env.reset(seed=seed)
    while True:
        action = policy.select_action(obs, deterministic=False)
        with quiet_stdout():
            obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    sim_time = float(info.get("sim_time", 0.0))
    techs = list(env.dispatcher.techs)
    per_tech = []
    for t in techs:
        per_tech.append(
            {
                "name": str(getattr(t, "name", f"tech_{t.id}")),
                "id": int(t.id),
                "total": int(t.disruption_count),
                "by_type": dict(t.disruption_counts_by_type),
                "final_fatigue": float(getattr(t, "fatigue", 0.0)),
            }
        )
    return {"sim_time": sim_time, "techs": per_tech}


def aggregate(episode_results):
    """Pool per-tech counts across episodes and per-tech."""
    by_type: dict[str, list[int]] = {}
    totals: list[int] = []
    fatigues: list[float] = []
    sim_times: list[float] = []
    for ep in episode_results:
        sim_times.append(ep["sim_time"])
        for t in ep["techs"]:
            totals.append(t["total"])
            fatigues.append(t["final_fatigue"])
            for k, v in t["by_type"].items():
                by_type.setdefault(k, []).append(v)
    return {
        "sim_times": sim_times,
        "totals": totals,
        "fatigues": fatigues,
        "by_type": by_type,
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("disruption_stats")

    from agents import (
        LeastBusyAgent,
        LeastFatiguedAgent,
        RandomAgent,
        RoundRobinAgent,
        ShortestQueueAgent,
    )

    policy_cls = {
        "random": RandomAgent,
        "round_robin": RoundRobinAgent,
        "least_busy": LeastBusyAgent,
        "least_fatigued": LeastFatiguedAgent,
        "shortest_queue": ShortestQueueAgent,
    }[args.policy]

    env_path = Path(args.env).resolve()
    t0 = time.time()
    episodes = []
    expected_table = None
    n_techs = None
    for ep in range(args.episodes):
        env, cfg = build_env(env_path, args.max_sim_time, seed=args.seed + ep)
        if expected_table is None:
            expected_table = expected_per_tech(cfg, float(cfg.gym.max_sim_time))
            n_techs = (
                cfg.randomized_scenario.n_technicians
                if cfg.randomized_scenario.enabled
                else len(cfg.technicians)
            )
        policy = policy_cls(n_actions=env.action_space.n)
        result = run_episode(env, policy, seed=args.seed + ep)
        log.info(
            "ep %d: sim_time=%.0f  fleet_disruption_count=%d",
            ep,
            result["sim_time"],
            sum(t["total"] for t in result["techs"]),
        )
        episodes.append(result)
    duration = time.time() - t0

    agg = aggregate(episodes)
    horizon = float(np.mean(agg["sim_times"]))

    log.info("\nCONFIG INTENT (per technician, over %.0f sim time):", horizon)
    log.info("-" * 88)
    for name, info in expected_table.items():
        log.info(
            "  %-12s  trigger=%-9s  preempt=%-5s  duration~%g±%g  expected_count=%s",
            name,
            info["trigger"],
            str(info["preemptive"]),
            info["duration_mu"],
            info["duration_sig"],
            info["expected_count_per_tech"],
        )
        log.info(f"      ↳ {info['note']}")

    log.info(
        "\nOBSERVED (over %d episodes × %d techs = %d samples):",
        args.episodes,
        n_techs,
        len(agg["totals"]),
    )
    log.info("-" * 88)
    log.info("  total disruptions per tech:")
    log.info(
        "      mean=%6.2f  median=%6.2f  std=%6.2f  min=%d  max=%d",
        mean(agg["totals"]),
        median(agg["totals"]),
        stdev(agg["totals"]) if len(agg["totals"]) > 1 else 0.0,
        min(agg["totals"]),
        max(agg["totals"]),
    )
    log.info("  final fatigue level per tech:")
    log.info(
        "      mean=%6.3f  median=%6.3f  std=%6.3f  min=%.3f  max=%.3f",
        mean(agg["fatigues"]),
        median(agg["fatigues"]),
        stdev(agg["fatigues"]) if len(agg["fatigues"]) > 1 else 0.0,
        min(agg["fatigues"]),
        max(agg["fatigues"]),
    )
    log.info("  per-type counts per technician:")
    for name in expected_table:
        samples = agg["by_type"].get(name, [0] * len(agg["totals"]))
        # Pad with zeros for techs that never experienced this type.
        while len(samples) < len(agg["totals"]):
            samples.append(0)
        log.info(
            "      %-12s  mean=%6.2f  median=%6.2f  std=%6.2f  min=%d  max=%d",
            name,
            mean(samples),
            median(samples),
            stdev(samples) if len(samples) > 1 else 0.0,
            min(samples),
            max(samples),
        )

    # Approximate fleet downtime as Σ_type (count_per_tech_mean × duration_mu)
    log.info(
        "\nAPPROX downtime per tech (mean count × duration_mu), as fraction of horizon:"
    )
    log.info("-" * 88)
    total_downtime = 0.0
    for name, info in expected_table.items():
        samples = agg["by_type"].get(name, [0] * len(agg["totals"]))
        m = mean(samples) if samples else 0.0
        dt = m * info["duration_mu"]
        total_downtime += dt
        log.info(
            "      %-12s  mean_count=%5.1f × duration_mu=%5.0f = %7.1f sim time  (%.2f%%)",
            name,
            m,
            info["duration_mu"],
            dt,
            100.0 * dt / max(horizon, 1.0),
        )
    log.info("      " + "-" * 80)
    log.info(
        "      %-12s  total=%7.1f sim time  (%.2f%% of horizon)",
        "ALL",
        total_downtime,
        100.0 * total_downtime / max(horizon, 1.0),
    )

    log.info("\nWall time: %.1fs", duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
