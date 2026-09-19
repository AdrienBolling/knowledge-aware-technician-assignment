"""Calibrate the v5 reward normalisation scales of the sequential baselines.

HTT-RL trained on the v5 reward stack with running per-component
standardisation (``RewardNormalizer``), but that normaliser state was
never checkpointed.  The model-based baselines of
``agents.baselines.sequential`` need the scales ``sigma_c`` to weigh the
components as the training objective did.  This script measures them on
the TRAINING worlds (``train_multiscale_v5`` sampler, horizons
U(sim_min, sim_max), world seeds disjoint from every benchmark seed): it
runs the Topsis rule (HTT-RL's behaviour-cloning teacher) with the v5
reward stack computed raw (normalisation off, unit coefficients), pools
the per-decision component values over all worlds, and writes the pooled
standard deviations into the ``sigma`` block of the planner parameters
JSON.  Episode-end components (one sample per episode) use the standard
deviation across episodes.

Usage::

    PYTHONHASHSEED=0 uv run --no-sync python scripts/calibrate_seqbase.py \\
        --worlds 12 --workers 12 --out run_configs/agents/seqbase_mpc.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TRAIN_CFG = ROOT / "run_configs/benchmark_suite/train_multiscale_v5.json"
TERMINAL = ("terminal_finished_products", "terminal_fleet_knowledge")


def _worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def run_world(task: tuple) -> dict:
    """One Topsis episode on one training world -> raw v5 component sums."""
    world_seed, sim_min, sim_max, steps_cap = task
    import eval_human_vs_performance as ev
    from agents import TopsisAgent
    from agents.baselines.sequential import V5_COMPONENTS

    cfg = ev.KATAConfig(**json.loads(TRAIN_CFG.read_text()))
    reward = cfg.gym.reward.model_copy(deep=True)
    reward.normalize_components = False
    for name in V5_COMPONENTS:
        comp = getattr(reward, name)
        if comp.enabled:
            comp.coefficient = 1.0
    cfg.gym = cfg.gym.model_copy(update={
        "reward": reward,
        "max_episode_steps": int(steps_cap),
        "max_sim_time": float((sim_min + sim_max) / 2.0),
        "max_sim_time_min": float(sim_min),
        "max_sim_time_max": float(sim_max),
    })
    sampler = ev.RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=int(world_seed))
    env = ev.make_env(cfg, sampler, "structured")
    agent = TopsisAgent(int(cfg.gym.max_techs))
    agent.attach_env(env)
    np.random.seed(int(world_seed))
    random.seed(int(world_seed))
    t0 = time.time()
    agent.on_episode_start()
    with ev.quiet():
        obs, _ = env.reset(seed=int(world_seed))
    sums = {n: [0, 0.0, 0.0] for n in V5_COMPONENTS}  # count, sum, sum of squares
    n_steps = 0
    while True:
        action = agent.select_action(obs, deterministic=True)
        with ev.quiet():
            obs, _, term, trunc, _ = env.step(action)
        n_steps += 1
        br = env._last_reward_breakdown
        for name in V5_COMPONENTS:
            if name in TERMINAL or name not in br:
                continue
            v = float(br[name])
            s = sums[name]
            s[0] += 1
            s[1] += v
            s[2] += v * v
        if term or trunc:
            terminal = {n: float(br[n]) for n in TERMINAL if n in br}
            break
    return {
        "world": int(world_seed), "sums": sums, "terminal": terminal, "n_steps": n_steps,
        "n_techs": len(env.dispatcher.techs), "n_machines": len(env._factory_machines()),
        "sim": float(env._sim_time()), "wall": time.time() - t0,
    }


def say(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%FT%TZ')} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--world-seed", type=int, default=90_000)
    ap.add_argument("--sim-min", type=float, default=200_000.0)
    ap.add_argument("--sim-max", type=float, default=350_000.0)
    ap.add_argument("--steps-cap", type=int, default=1_000_000)
    ap.add_argument("--out", default="run_configs/agents/seqbase_mpc.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.worlds, args.workers = 2, 2
        args.sim_min = args.sim_max = 8_000.0
    from agents.baselines.sequential import V5_COMPONENTS

    tasks = [(args.world_seed + i, args.sim_min, args.sim_max, args.steps_cap) for i in range(args.worlds)]
    say(f"SEQBASE CALIBRATION: {args.worlds} worlds, {args.workers} workers, "
        f"horizons U({args.sim_min:.0f},{args.sim_max:.0f}), seeds {args.world_seed}..")
    pooled = {n: [0, 0.0, 0.0] for n in V5_COMPONENTS}
    terminal = {n: [] for n in TERMINAL}
    worlds = []
    with Pool(args.workers, initializer=_worker_init, maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_world, tasks):
            for n, (c, s1, s2) in res["sums"].items():
                p = pooled[n]
                p[0] += c
                p[1] += s1
                p[2] += s2
            for n, v in res["terminal"].items():
                terminal[n].append(v)
            worlds.append({k: res[k] for k in ("world", "n_steps", "n_techs", "n_machines", "sim", "wall")})
            say(f"  world {res['world']}: {res['n_steps']} decisions, {res['n_techs']} techs, "
                f"{res['n_machines']} machines, {res['wall']:.0f}s")
    sigma, stats = {}, {}
    for n in V5_COMPONENTS:
        if n in TERMINAL:
            vals = np.asarray(terminal[n], dtype=float)
            sd = float(vals.std()) if len(vals) > 1 else 1.0
            stats[n] = {"n": int(len(vals)), "mean": float(vals.mean()) if len(vals) else 0.0, "std": sd}
        else:
            c, s1, s2 = pooled[n]
            mean = s1 / c if c else 0.0
            sd = float(np.sqrt(max(s2 / c - mean * mean, 0.0))) if c else 1.0
            stats[n] = {"n": int(c), "mean": mean, "std": sd}
        sigma[n] = sd if sd > 1e-8 else 1.0
        say(f"  {n:28s} n={stats[n]['n']:>9d} mean={stats[n]['mean']:+.5g} std={stats[n]['std']:.5g}")
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.is_file() else {}
    data["sigma"] = sigma
    data["calibration"] = {
        "policy": "topsis", "env_config": str(TRAIN_CFG.relative_to(ROOT)),
        "worlds": args.worlds, "world_seed": args.world_seed, "sim_min": args.sim_min,
        "sim_max": args.sim_max, "stats": stats, "per_world": sorted(worlds, key=lambda w: w["world"]),
        "note": "std of raw v5 components pooled over all decisions of all worlds "
                "(episode-end terms: std across episodes); zero std -> 1.0",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    say(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
