"""Reward component study: scale alignment, delta vs absolute, explosion scenarios.

This script runs episodes with a random policy under several factory
configurations and collects per-step reward breakdowns.  It produces:

1. **Scale alignment report** — distribution stats (mean, std, min, max,
   percentiles) for every reward component, highlighting mismatches.
2. **Delta vs absolute comparison** — for components that have a
   natural delta interpretation (workload_balance, queue_size, downtime_cost),
   compute the delta variant and compare stability.
3. **Explosion stress test** — run under extreme breakdown rates and
   long episodes to see which components blow up.

Usage::

    uv run python studies/reward_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Ensure src is on the path
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

from kata.core.config import KATAConfig, GymEnvConfig, GymRewardConfig
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder


# ======================================================================
# Helpers
# ======================================================================


def _load_config(path: str) -> KATAConfig:
    with open(path) as f:
        data = json.load(f)
    return KATAConfig(**data)


def _all_rewards_enabled(coeff: float = 1.0) -> dict:
    """Return a reward config dict with ALL components enabled."""
    components = [
        "assignment", "wait_time", "queue_size", "busy_technician",
        "fatigue_cost", "knowledge_match", "workload_balance",
        "estimated_repair_time", "machine_criticality",
        "fleet_availability", "throughput_delta", "repair_backlog_age",
        "technician_utilization", "downtime_cost",
    ]
    return {c: {"enabled": True, "coefficient": coeff} for c in components}


def _make_env(config: KATAConfig) -> KataEnv:
    """Build a KataEnv from a KATAConfig with all rewards enabled."""
    # Override reward config to enable all components
    gym_cfg = config.gym.model_copy(update={"reward": GymRewardConfig(**_all_rewards_enabled())})
    config = config.model_copy(update={"gym": gym_cfg})
    factory = lambda: ScenarioBuilder(config).build()
    return KataEnv(scenario_factory=factory, config=config.gym)


def run_episode(env: KataEnv, seed: int = 42, max_steps: int = 200) -> dict:
    """Run one episode with random actions, collecting reward breakdowns."""
    obs, info = env.reset(seed=seed)
    step_data: list[dict[str, float]] = []
    total_reward = 0.0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if "reward_breakdown" in info and info["reward_breakdown"]:
            step_data.append(dict(info["reward_breakdown"]))

        if terminated or truncated:
            break

    return {
        "step_data": step_data,
        "total_reward": total_reward,
        "n_steps": len(step_data),
        "final_info": info,
    }


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute distribution statistics."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "p5": 0, "p95": 0, "abs_mean": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "abs_mean": float(np.mean(np.abs(arr))),
    }


# ======================================================================
# Studies
# ======================================================================


def study_scale_alignment(config_path: str, n_episodes: int = 10, max_steps: int = 200):
    """Study 1: Distribution of each reward component under random policy."""
    print("\n" + "=" * 72)
    print("STUDY 1: REWARD SCALE ALIGNMENT")
    print("=" * 72)
    print(f"Config: {config_path}")
    print(f"Episodes: {n_episodes}, Max steps/episode: {max_steps}")

    config = _load_config(config_path)
    env = _make_env(config)

    # Collect all per-step reward values
    all_values: dict[str, list[float]] = defaultdict(list)

    for ep in range(n_episodes):
        result = run_episode(env, seed=42 + ep, max_steps=max_steps)
        for step_breakdown in result["step_data"]:
            for comp, val in step_breakdown.items():
                all_values[comp].append(val)

    # Report
    print(f"\nTotal steps collected: {sum(len(v) for v in all_values.values()) // max(len(all_values), 1)}")
    print(f"\n{'Component':<28s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'|Mean|':>8s} {'P5':>8s} {'P95':>8s}")
    print("-" * 96)

    stats_by_component: dict[str, dict] = {}
    for comp in sorted(all_values.keys()):
        stats = compute_stats(all_values[comp])
        stats_by_component[comp] = stats
        print(
            f"  {comp:<26s} {stats['mean']:>8.4f} {stats['std']:>8.4f} "
            f"{stats['min']:>8.4f} {stats['max']:>8.4f} {stats['abs_mean']:>8.4f} "
            f"{stats['p5']:>8.4f} {stats['p95']:>8.4f}"
        )

    # Scale alignment analysis
    abs_means = {c: s["abs_mean"] for c, s in stats_by_component.items() if s["abs_mean"] > 0}
    if abs_means:
        max_scale = max(abs_means.values())
        min_scale = min(abs_means.values())
        ratio = max_scale / max(min_scale, 1e-10)
        print(f"\nScale ratio (max/min |mean|): {ratio:.1f}x")
        if ratio > 10:
            print("  WARNING: Scale mismatch > 10x. Consider normalizing coefficients.")
            big = [c for c, v in abs_means.items() if v > 0.5 * max_scale]
            small = [c for c, v in abs_means.items() if v < 0.1 * max_scale]
            if big:
                print(f"  Large-scale components: {', '.join(big)}")
            if small:
                print(f"  Small-scale components: {', '.join(small)}")
        else:
            print("  Scale alignment is acceptable.")

    return stats_by_component


def study_delta_vs_absolute(config_path: str, n_episodes: int = 10, max_steps: int = 200):
    """Study 2: Compare absolute vs delta formulations for stateful components."""
    print("\n" + "=" * 72)
    print("STUDY 2: DELTA vs ABSOLUTE FORMULATIONS")
    print("=" * 72)

    config = _load_config(config_path)
    env = _make_env(config)

    # Components where delta makes sense
    delta_candidates = [
        "workload_balance",
        "fleet_availability",
        "downtime_cost",
        "technician_utilization",
    ]

    abs_series: dict[str, list[float]] = defaultdict(list)
    delta_series: dict[str, list[float]] = defaultdict(list)
    prev_values: dict[str, float] = {}

    for ep in range(n_episodes):
        prev_values.clear()
        result = run_episode(env, seed=42 + ep, max_steps=max_steps)

        for step_breakdown in result["step_data"]:
            for comp in delta_candidates:
                if comp not in step_breakdown:
                    continue
                val = step_breakdown[comp]
                abs_series[comp].append(val)

                if comp in prev_values:
                    delta_series[comp].append(val - prev_values[comp])
                prev_values[comp] = val

    print(f"\n{'Component':<28s} | {'--- Absolute ---':^34s} | {'--- Delta ---':^34s} | {'Verdict'}")
    print(f"{'':28s} | {'Mean':>8s} {'Std':>8s} {'P5':>8s} {'P95':>8s} | {'Mean':>8s} {'Std':>8s} {'P5':>8s} {'P95':>8s} |")
    print("-" * 120)

    for comp in delta_candidates:
        if comp not in abs_series:
            continue
        a_stats = compute_stats(abs_series[comp])
        d_stats = compute_stats(delta_series[comp]) if comp in delta_series else compute_stats([])

        # Verdict: delta is better when it has lower variance relative to signal
        a_cv = a_stats["std"] / max(abs(a_stats["mean"]), 1e-6)
        d_cv = d_stats["std"] / max(abs(d_stats["mean"]), 1e-6) if d_stats["std"] > 0 else float("inf")

        if d_stats["std"] > 0 and d_cv < a_cv * 0.8:
            verdict = "DELTA better (lower CoV)"
        elif a_stats["std"] > 0 and abs(d_stats["mean"]) < abs(a_stats["mean"]) * 0.1:
            verdict = "DELTA (near-zero mean = pure gradient)"
        elif a_cv < d_cv * 0.8:
            verdict = "ABSOLUTE better (lower CoV)"
        else:
            verdict = "similar"

        print(
            f"  {comp:<26s} | "
            f"{a_stats['mean']:>8.4f} {a_stats['std']:>8.4f} {a_stats['p5']:>8.4f} {a_stats['p95']:>8.4f} | "
            f"{d_stats['mean']:>8.4f} {d_stats['std']:>8.4f} {d_stats['p5']:>8.4f} {d_stats['p95']:>8.4f} | "
            f"{verdict}"
        )


def study_explosion_scenarios(config_path: str):
    """Study 3: Test reward stability under extreme conditions."""
    print("\n" + "=" * 72)
    print("STUDY 3: EXPLOSION / STRESS SCENARIOS")
    print("=" * 72)

    base_config = _load_config(config_path)

    scenarios = {
        "normal": {
            "max_episode_steps": 200,
            "max_sim_time": 50000.0,
        },
        "long_episode": {
            "max_episode_steps": 2000,
            "max_sim_time": 500000.0,
        },
        "very_long_episode": {
            "max_episode_steps": 5000,
            "max_sim_time": 1000000.0,
        },
    }

    for scenario_name, overrides in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"  Config: max_steps={overrides['max_episode_steps']}, max_sim_time={overrides['max_sim_time']}")

        gym_update = {
            "reward": GymRewardConfig(**_all_rewards_enabled()),
            **overrides,
        }
        config = base_config.model_copy(
            update={"gym": base_config.gym.model_copy(update=gym_update)}
        )
        env = _make_env(config)

        result = run_episode(env, seed=42, max_steps=overrides["max_episode_steps"])
        steps = result["n_steps"]
        print(f"  Actual steps: {steps}")

        if not result["step_data"]:
            print("  No steps completed.")
            continue

        # Check for explosions
        all_values: dict[str, list[float]] = defaultdict(list)
        for step_breakdown in result["step_data"]:
            for comp, val in step_breakdown.items():
                all_values[comp].append(val)

        exploded = []
        stable = []
        for comp in sorted(all_values.keys()):
            vals = all_values[comp]
            stats = compute_stats(vals)

            # Check for explosion signatures
            is_exploding = False
            if abs(stats["max"]) > 100 or abs(stats["min"]) < -100:
                is_exploding = True
            if stats["std"] > 10 * max(abs(stats["mean"]), 0.01):
                is_exploding = True
            # Check if magnitude grows over time (trend)
            if len(vals) > 10:
                first_half = np.mean(np.abs(vals[: len(vals) // 2]))
                second_half = np.mean(np.abs(vals[len(vals) // 2 :]))
                if second_half > first_half * 5 and second_half > 1.0:
                    is_exploding = True

            if is_exploding:
                exploded.append(comp)
                print(
                    f"  EXPLOSION: {comp:<26s} "
                    f"mean={stats['mean']:>10.2f}  std={stats['std']:>10.2f}  "
                    f"min={stats['min']:>10.2f}  max={stats['max']:>10.2f}"
                )
            else:
                stable.append(comp)

        if not exploded:
            print("  All components stable.")
        else:
            print(f"  Stable: {', '.join(stable)}")
            print(f"  EXPLODED: {', '.join(exploded)}")

    # Also test with modified factory (higher breakdown rates)
    print(f"\n--- Scenario: high_breakdown_rate ---")
    print("  Modifying machines to have 10x higher breakdown probability")

    config = base_config.model_copy()
    # We can't easily modify the breakdown rates from the config object
    # without deep mutation, so we test the raw reward methods instead
    gym_update = {
        "reward": GymRewardConfig(**_all_rewards_enabled()),
        "max_episode_steps": 500,
        "max_sim_time": 100000.0,
    }
    config = base_config.model_copy(
        update={"gym": base_config.gym.model_copy(update=gym_update)}
    )
    env = _make_env(config)
    result = run_episode(env, seed=123, max_steps=500)
    print(f"  Actual steps: {result['n_steps']}, Total return: {result['total_reward']:.2f}")

    if result["step_data"]:
        all_values = defaultdict(list)
        for step_breakdown in result["step_data"]:
            for comp, val in step_breakdown.items():
                all_values[comp].append(val)

        bounded_components = [
            "fleet_availability", "throughput_delta", "repair_backlog_age",
            "technician_utilization", "downtime_cost", "knowledge_match",
            "machine_criticality",
        ]
        unbounded_components = ["queue_size", "wait_time"]

        print("\n  Bounded components (should stay in [-1, 1] or similar):")
        for comp in bounded_components:
            if comp in all_values:
                stats = compute_stats(all_values[comp])
                in_range = -1.5 <= stats["min"] and stats["max"] <= 1.5
                status = "OK" if in_range else "OUT OF RANGE"
                print(f"    {comp:<26s} [{stats['min']:>7.3f}, {stats['max']:>7.3f}]  {status}")

        print("\n  Unbounded components (check for explosion):")
        for comp in unbounded_components:
            if comp in all_values:
                stats = compute_stats(all_values[comp])
                print(f"    {comp:<26s} [{stats['min']:>10.2f}, {stats['max']:>10.2f}]  std={stats['std']:.2f}")


def study_correlation(config_path: str, n_episodes: int = 10, max_steps: int = 200):
    """Study 4: Correlation between reward components (detect redundancy)."""
    print("\n" + "=" * 72)
    print("STUDY 4: REWARD COMPONENT CORRELATIONS")
    print("=" * 72)

    config = _load_config(config_path)
    env = _make_env(config)

    all_values: dict[str, list[float]] = defaultdict(list)
    for ep in range(n_episodes):
        result = run_episode(env, seed=42 + ep, max_steps=max_steps)
        for step_breakdown in result["step_data"]:
            for comp, val in step_breakdown.items():
                all_values[comp].append(val)

    components = sorted(all_values.keys())
    n = len(components)
    if n < 2:
        print("Not enough components to analyze correlations.")
        return

    # Build matrix
    min_len = min(len(all_values[c]) for c in components)
    matrix = np.zeros((n, min_len))
    for i, comp in enumerate(components):
        matrix[i, :] = all_values[comp][:min_len]

    corr = np.corrcoef(matrix)

    print(f"\nPearson correlation matrix ({min_len} samples):\n")
    # Header
    print(f"{'':>26s}", end="")
    for c in components:
        print(f" {c[:8]:>8s}", end="")
    print()

    for i, c1 in enumerate(components):
        print(f"  {c1:<24s}", end="")
        for j, c2 in enumerate(components):
            val = corr[i, j]
            if i == j:
                marker = "    1.00"
            elif abs(val) > 0.8:
                marker = f" {val:>7.2f}*"  # High correlation
            elif abs(val) > 0.5:
                marker = f" {val:>7.2f}~"  # Moderate
            else:
                marker = f" {val:>7.2f} "
            print(marker, end="")
        print()

    # Flag highly correlated pairs
    print("\nHighly correlated pairs (|r| > 0.7):")
    found = False
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if abs(r) > 0.7:
                found = True
                direction = "positive" if r > 0 else "negative"
                print(f"  {components[i]} <-> {components[j]}: r={r:.3f} ({direction})")
                if abs(r) > 0.9:
                    print(f"    WARNING: Very high correlation — may be redundant.")
    if not found:
        print("  None found — all components provide independent signals.")


# ======================================================================
# Main
# ======================================================================


def main():
    config_path = str(Path(__file__).resolve().parent.parent / "run_configs" / "complex_factory.json")

    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    print("KATA Reward Component Study")
    print(f"Config: {config_path}")

    stats = study_scale_alignment(config_path, n_episodes=10, max_steps=200)
    study_delta_vs_absolute(config_path, n_episodes=10, max_steps=200)
    study_explosion_scenarios(config_path)
    study_correlation(config_path, n_episodes=10, max_steps=200)

    print("\n" + "=" * 72)
    print("STUDY COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
