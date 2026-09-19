"""Light grid tuner for the rolling-horizon look-ahead dispatcher.

Tunes only the look-ahead depth ``horizon_k`` and the terminal-value weight
``terminal_weight`` of ``RollingHorizonMPCAgent``
(``agents.baselines.sequential``).  Every other planner parameter and the
calibrated reward scales stay as in the parameter JSON (``--params``,
written first by ``scripts/calibrate_seqbase.py``).

Protocol (the same as ``scripts/tune_evo_topsis.py``):

* worlds from the TRAINING distribution of HTT-RL (``train_multiscale_v5``
  sampler, horizons U(sim_min, sim_max)); world seeds are disjoint from the
  benchmark seeds (eval seed 20260722);
* common random numbers: every candidate runs on the same worlds, through
  the benchmark harness's own ``run_episode`` (KPI parity);
* fitness = mean signed relative improvement over the Topsis rule on four
  KPIs POOLED over the worlds: total products (up), mean MTTR (down),
  total disruptions per 10^3 total products (down) and total final fleet
  knowledge (up).  Pooling keeps a world with a collapsed line (a few
  products, so a huge disruption ratio) from dominating the score; the
  per-world mean of ``scripts/tune_evo_topsis.py`` is also recorded;
* stage 1 scores the full grid on ``--worlds`` worlds; stage 2 re-runs the
  ``--top`` best grid points, the greedy (K=0, no terminal value) and the
  reference on ``--val-worlds`` fresh worlds; the best stage-2 grid point
  is written to ``--out`` (``horizon_k``, ``terminal_weight`` and a
  ``tuning`` record).

Usage::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= uv run --no-sync python scripts/tune_seqbase.py \\
        --workers 12 --out run_configs/agents/seqbase_mpc.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
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
KPI_KEYS = ("products", "mttr", "disr", "know")


def _worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def evaluate(task: tuple) -> dict:
    """One episode of one candidate on one training world -> headline KPIs."""
    kind, k, w, world_seed, sim_min, sim_max, steps_cap, params_path = task
    import eval_human_vs_performance as ev
    from agents import TopsisAgent
    from agents.baselines.sequential import load_params, GreedyTrainingRewardAgent, RollingHorizonMPCAgent

    cfg = ev.KATAConfig(**json.loads(TRAIN_CFG.read_text()))
    cfg.gym = cfg.gym.model_copy(update={
        "max_episode_steps": int(steps_cap),
        "max_sim_time": float((sim_min + sim_max) / 2.0),
        "max_sim_time_min": float(sim_min),
        "max_sim_time_max": float(sim_max),
    })
    sampler = ev.RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=int(world_seed))
    env = ev.make_env(cfg, sampler, "structured")
    n_act = int(cfg.gym.max_techs)
    params = load_params(params_path)
    if kind == "topsis":
        agent = TopsisAgent(n_act)
    elif kind == "greedy":
        agent = GreedyTrainingRewardAgent(n_act, params=params)
    else:
        agent = RollingHorizonMPCAgent(n_act, params=params, horizon_k=int(k), terminal_weight=float(w))
    agent.attach_env(env)
    t0 = time.time()
    with ev.quiet():
        kpis, records = ev.run_episode(agent, env, seed=int(world_seed), record_every=10 ** 9)
    products = float(kpis["finished_products"])
    return {
        "kind": kind, "k": k, "w": w, "world": int(world_seed),
        "products": products,
        "mttr": float(kpis["mttr"]),
        "disr": float(kpis["ill_technician_count"]) / max(products, 1.0) * 1000.0,
        "know": float(records[-1]["fleet_knowledge"]),
        "n_steps": int(kpis["n_steps"]),
        "sim": float(kpis["final_sim_time"]),
        "n_techs": len(env.dispatcher.techs),
        "planner_ms": float(kpis.get("planner_ms_median", float("nan"))),
        "wall": time.time() - t0,
    }


def fitness(cand: list[dict], ref: list[dict]) -> float:
    """Mean signed relative improvement over the reference on the four
    headline KPIs pooled over the worlds (higher is better)."""
    for c, r in zip(cand, ref):
        assert c["world"] == r["world"]

    def pooled(rows):
        prod = sum(x["products"] for x in rows)
        disr = sum(x["disr"] * x["products"] / 1000.0 for x in rows)
        return {"products": prod, "mttr": float(np.mean([x["mttr"] for x in rows])),
                "disr": disr / max(prod, 1.0) * 1000.0, "know": sum(x["know"] for x in rows)}

    c, r = pooled(cand), pooled(ref)
    return float(np.mean([
        (c["products"] - r["products"]) / max(r["products"], 1.0),
        (r["mttr"] - c["mttr"]) / max(r["mttr"], 1e-9),
        (r["disr"] - c["disr"]) / max(r["disr"], 1e-9),
        (c["know"] - r["know"]) / max(r["know"], 1e-9),
    ]))


def fitness_per_world(cand: list[dict], ref: list[dict]) -> float:
    """Per-world mean of the relative improvements (the Evo-Topsis tuner's
    fitness), recorded for comparison."""
    vals = []
    for c, r in zip(cand, ref):
        assert c["world"] == r["world"]
        vals.append(float(np.mean([
            (c["products"] - r["products"]) / max(r["products"], 1.0),
            (r["mttr"] - c["mttr"]) / max(r["mttr"], 1e-9),
            (r["disr"] - c["disr"]) / max(r["disr"], 1e-9),
            (c["know"] - r["know"]) / max(r["know"], 1e-9),
        ])))
    return float(np.mean(vals))


def say(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%FT%TZ')} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", default="1,2,3", help="grid of horizon_k")
    ap.add_argument("--w", default="0,0.5,1,2", help="grid of terminal_weight")
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--val-worlds", type=int, default=16)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--world-seed", type=int, default=70_000, help="stage-1 worlds; stage 2 uses +10000")
    ap.add_argument("--sim-min", type=float, default=200_000.0)
    ap.add_argument("--sim-max", type=float, default=350_000.0)
    ap.add_argument("--steps-cap", type=int, default=1_000_000)
    ap.add_argument("--params", default="run_configs/agents/seqbase_mpc.json")
    ap.add_argument("--out", default="run_configs/agents/seqbase_mpc.json")
    ap.add_argument("--history", default="reports/seqbase_tune_history.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.k, args.w, args.worlds, args.val_worlds, args.top = "1,2", "0,1", 1, 1, 1
        args.sim_min = args.sim_max = 6_000.0
        args.workers = min(args.workers, 4)
    grid = [(int(k), float(w)) for k in args.k.split(",") for w in args.w.split(",")]
    params_path = str((ROOT / args.params) if not Path(args.params).is_absolute() else args.params)

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    hist = open(args.history, "w", newline="")
    hw = csv.writer(hist)
    cols = ["stage", "kind", "k", "w", "world", *KPI_KEYS, "n_steps", "sim", "n_techs", "planner_ms", "wall"]
    hw.writerow(cols)

    def run_stage(pool, cands, worlds, stage):
        tasks = [(kind, k, w, ws, args.sim_min, args.sim_max, args.steps_cap, params_path)
                 for (kind, k, w) in cands for ws in worlds]
        out, t0 = {}, time.time()
        for i, r in enumerate(pool.imap_unordered(evaluate, tasks, chunksize=1)):
            out[(r["kind"], r["k"], r["w"], r["world"])] = r
            hw.writerow([stage] + [r[c] if not isinstance(r[c], float) else f"{r[c]:.4f}" for c in cols[1:]])
            hist.flush()
            if (i + 1) % 8 == 0 or i + 1 == len(tasks):
                say(f"  [{stage}] {i + 1}/{len(tasks)} episodes ({time.time() - t0:.0f}s)")
        return {(kind, k, w): [out[(kind, k, w, ws)] for ws in worlds] for (kind, k, w) in cands}

    def table(res, stage):
        ref = res[("topsis", None, None)]
        rows = []
        for key, rs in res.items():
            if key[0] == "topsis":
                continue
            rows.append({"kind": key[0], "horizon_k": key[1], "terminal_weight": key[2],
                         "fitness": fitness(rs, ref),
                         "fitness_per_world": fitness_per_world(rs, ref),
                         **{m: float(np.mean([r[m] for r in rs])) for m in KPI_KEYS},
                         "planner_ms": float(np.nanmedian([r["planner_ms"] for r in rs]))})
        rows.sort(key=lambda x: -x["fitness"])
        say(f"{stage}: reference topsis " + " ".join(f"{m}={np.mean([r[m] for r in ref]):.1f}" for m in KPI_KEYS)
            + f" | fleets {sorted({r['n_techs'] for r in ref})}")
        for x in rows:
            say(f"  {x['kind']:6s} K={x['horizon_k']} w={x['terminal_weight']} fitness {x['fitness']:+.4f} "
                f"(per-world mean {x['fitness_per_world']:+.4f}) | "
                + " ".join(f"{m}={x[m]:.1f}" for m in KPI_KEYS) + f" | planner {x['planner_ms']:.2f} ms")
        return rows

    ref_c = ("topsis", None, None)
    greedy_c = ("greedy", 0, 0.0)
    worlds1 = [args.world_seed + i for i in range(args.worlds)]
    worlds2 = [args.world_seed + 10_000 + i for i in range(args.val_worlds)]
    say(f"SEQBASE TUNE: grid K={args.k} w={args.w} ({len(grid)} points), worlds {args.worlds}+{args.val_worlds}, "
        f"workers {args.workers}, horizons U({args.sim_min:.0f},{args.sim_max:.0f}), params {params_path}")
    with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
        res1 = run_stage(pool, [ref_c, greedy_c] + [("mpc", k, w) for k, w in grid], worlds1, "stage1")
        rows1 = table(res1, "stage1")
        finalists = [(x["kind"], x["horizon_k"], x["terminal_weight"]) for x in rows1 if x["kind"] == "mpc"][:args.top]
        res2 = run_stage(pool, [ref_c, greedy_c] + finalists, worlds2, "stage2")
        rows2 = table(res2, "stage2")
    best = next(x for x in rows2 if x["kind"] == "mpc")
    say(f"chosen: horizon_k={best['horizon_k']} terminal_weight={best['terminal_weight']} "
        f"(stage-2 fitness {best['fitness']:+.4f})")

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.is_file() else {}
    data["horizon_k"] = int(best["horizon_k"])
    data["terminal_weight"] = float(best["terminal_weight"])
    data["tuning"] = {
        "reference": "topsis", "env_config": str(TRAIN_CFG.relative_to(ROOT)),
        "fitness": "mean relative improvement over topsis of the KPIs pooled over the worlds: "
                   "total products, mean MTTR, total disruptions per 1e3 total products, "
                   "total final fleet knowledge",
        "grid": {"horizon_k": sorted({k for k, _ in grid}), "terminal_weight": sorted({w for _, w in grid})},
        "stage1_worlds": worlds1, "stage2_worlds": worlds2,
        "sim_min": args.sim_min, "sim_max": args.sim_max,
        "stage1": rows1, "stage2": rows2,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(data, indent=2) + "\n")
    say(f"wrote {out}")
    say("SEQBASE TUNE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
