"""GA tuner for the Evo-TOPSIS dispatching rule (the benchmark's
metaheuristic / simheuristic baseline).

A genetic algorithm searches the signed 5-weight vector of
``EvoTopsisAgent`` (criteria: empirical repair time, fatigue, workload,
key experience, total experience; sign = direction, magnitude =
importance).  Fitness is *simulation-based*: each candidate runs one
full episode per world on the same multiscale training distribution
HTT-RL trained on (``train_multiscale_v5``, horizons U(sim_min,
sim_max)), through the benchmark harness's own ``run_episode`` (KPI
parity with the paper's tables), and is scored as the mean relative
improvement over the untuned rule (Emp-TOPSIS) on the four headline
KPIs -- products (up), MTTR (down), disruptions per 10^3 products
(down), final fleet knowledge (up) -- averaged over the generation's
worlds.  Common random numbers: every candidate of a generation sees
the same worlds; the world set rotates each generation; the final pick
is validated on fresh worlds.

Outputs: the weights JSON consumed by ``EvoTopsisAgent.load_weights``
(``--out``), a per-episode history CSV (``--history``), and a log on
stdout.  ``--smoke`` runs a tiny configuration end to end.
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
CRITERIA = ("repair_time", "fatigue", "workload", "key_experience", "total_experience")
REFERENCE = (0.5, 0.3, 0.2, 0.0, 0.0)  # Emp-TOPSIS
KPI_KEYS = ("products", "mttr", "disr", "know")


def _worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def evaluate(task: tuple) -> dict:
    """One episode of one candidate on one world -> headline KPIs."""
    weights, world_seed, sim_min, sim_max, steps_cap = task
    import eval_human_vs_performance as ev  # harness runner: KPI parity
    from agents.baselines.heuristics import EvoTopsisAgent

    cfg = ev.KATAConfig(**json.loads(TRAIN_CFG.read_text()))
    cfg.gym = cfg.gym.model_copy(update={
        "max_episode_steps": int(steps_cap),
        "max_sim_time": float((sim_min + sim_max) / 2.0),
        "max_sim_time_min": float(sim_min),
        "max_sim_time_max": float(sim_max),
    })
    # Fresh sampler per (candidate, world): identical seed -> identical
    # draw sequence -> every candidate faces byte-identical worlds.
    sampler = ev.RandomScenarioSampler(cfg, cfg.randomized_scenario, seed=int(world_seed))
    env = ev.make_env(cfg, sampler, "structured")
    agent = EvoTopsisAgent(int(cfg.gym.max_techs), weights=weights)
    agent.attach_env(env)
    t0 = time.time()
    kpis, records = ev.run_episode(agent, env, seed=int(world_seed), record_every=10 ** 9)
    products = float(kpis["finished_products"])
    ill = float(kpis["ill_technician_count"])
    return {
        "world": int(world_seed),
        "products": products,
        "mttr": float(kpis["mttr"]),
        "disr": ill / max(products, 1.0) * 1000.0,
        "know": float(records[-1]["fleet_knowledge"]),
        "n_steps": int(kpis["n_steps"]),
        "sim": float(kpis["final_sim_time"]),
        "n_techs": int(len(env.dispatcher.techs)) if getattr(env, "dispatcher", None) else -1,
        "wall": time.time() - t0,
    }


def fitness(cand: list[dict], ref: list[dict]) -> float:
    """Mean over worlds of the mean signed relative improvement over the
    reference rule on the four headline KPIs (higher is better)."""
    vals = []
    for c, r in zip(cand, ref):
        assert c["world"] == r["world"]
        terms = [
            (c["products"] - r["products"]) / max(r["products"], 1.0),
            (r["mttr"] - c["mttr"]) / max(r["mttr"], 1e-9),
            (r["disr"] - c["disr"]) / max(r["disr"], 1e-9),
            (c["know"] - r["know"]) / max(r["know"], 1e-9),
        ]
        vals.append(float(np.mean(terms)))
    return float(np.mean(vals))


def say(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%FT%TZ')} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--worlds", type=int, default=6, help="worlds per generation (common random numbers)")
    ap.add_argument("--val-worlds", type=int, default=12, help="fresh worlds for the final selection")
    ap.add_argument("--top", type=int, default=5, help="finalists validated from the last generation")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--world-seed", type=int, default=50_000, help="first world seed (validation uses +10000)")
    ap.add_argument("--sim-min", type=float, default=200_000.0)
    ap.add_argument("--sim-max", type=float, default=350_000.0)
    ap.add_argument("--steps-cap", type=int, default=1_000_000)
    ap.add_argument("--out", default="run_configs/agents/evo_topsis_weights.json")
    ap.add_argument("--history", default="reports/evo_topsis_tune_history.csv")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    args = ap.parse_args()
    if args.smoke:
        args.pop, args.gens, args.worlds, args.val_worlds, args.top = 3, 1, 1, 1, 2
        args.sim_min = args.sim_max = 15_000.0
        args.steps_cap = 2_000
        args.workers = min(args.workers, 3)

    rng = np.random.default_rng(args.seed)
    ref = np.asarray(REFERENCE)
    n_local = max(0, args.pop // 2 - 1)
    pop = [ref.copy()]
    pop += [np.clip(ref + rng.normal(0.0, 0.3, len(CRITERIA)), -1, 1) for _ in range(n_local)]
    pop += [rng.uniform(-1, 1, len(CRITERIA)) for _ in range(args.pop - len(pop))]

    Path(args.history).parent.mkdir(parents=True, exist_ok=True)
    hist = open(args.history, "w", newline="")
    hw = csv.writer(hist)
    hw.writerow(["phase", "gen", "cand", *CRITERIA, "world", *KPI_KEYS, "n_steps", "sim", "n_techs", "wall"])

    def run_batch(pool, weights_list, worlds, phase, gen):
        tasks = [(tuple(float(x) for x in w), ws, args.sim_min, args.sim_max, args.steps_cap)
                 for w in weights_list for ws in worlds]
        out, t0 = [], time.time()
        for i, r in enumerate(pool.imap(evaluate, tasks, chunksize=1)):
            out.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                say(f"  [{phase} g{gen}] {i + 1}/{len(tasks)} episodes ({time.time() - t0:.0f}s)")
        W = len(worlds)
        grouped = [out[k * W:(k + 1) * W] for k in range(len(weights_list))]
        for k, (w, rows) in enumerate(zip(weights_list, grouped)):
            for r in rows:
                hw.writerow([phase, gen, k, *[f"{x:.4f}" for x in w], r["world"],
                             *[f"{r[m]:.4f}" for m in KPI_KEYS], r["n_steps"], f"{r['sim']:.0f}",
                             r["n_techs"], f"{r['wall']:.1f}"])
        hist.flush()
        return grouped

    def tournament(fits, k=3):
        idx = rng.choice(len(fits), size=k, replace=False)
        return int(idx[int(np.argmax(np.asarray(fits)[idx]))])

    say(f"EVO-TOPSIS GA: pop {args.pop}, gens {args.gens}, worlds/gen {args.worlds}, "
        f"val worlds {args.val_worlds}, workers {args.workers}, horizons U({args.sim_min:.0f},{args.sim_max:.0f})")
    generations = []
    with Pool(args.workers, initializer=_worker_init, maxtasksperchild=8) as pool:
        fits = None
        for g in range(args.gens):
            worlds = [args.world_seed + g * args.worlds + i for i in range(args.worlds)]
            grouped = run_batch(pool, pop + [ref], worlds, "train", g)
            ref_rows = grouped[-1]
            fits = [fitness(rows, ref_rows) for rows in grouped[:-1]]
            order = list(np.argsort(-np.asarray(fits)))
            b = order[0]
            generations.append({"gen": g, "worlds": worlds, "best_fit": fits[b], "mean_fit": float(np.mean(fits)),
                                "best_weights": [float(x) for x in pop[b]],
                                "ref_kpis": {m: float(np.mean([r[m] for r in ref_rows])) for m in KPI_KEYS},
                                "best_kpis": {m: float(np.mean([r[m] for r in grouped[b]])) for m in KPI_KEYS}})
            say(f"gen {g}: best {fits[b]:+.4f} mean {np.mean(fits):+.4f} | best w = "
                + " ".join(f"{x:+.2f}" for x in pop[b])
                + " | ref " + " ".join(f"{m}={generations[-1]['ref_kpis'][m]:.1f}" for m in KPI_KEYS)
                + " | best " + " ".join(f"{m}={generations[-1]['best_kpis'][m]:.1f}" for m in KPI_KEYS)
                + f" | fleets {sorted({r['n_techs'] for r in ref_rows})}")
            if g == args.gens - 1:
                break
            elites = [pop[i].copy() for i in order[:args.elite]]
            sigma = max(0.05, 0.2 * (1.0 - g / max(args.gens - 1, 1)))
            children = []
            while len(children) < args.pop - len(elites):
                a, b2 = pop[tournament(fits)], pop[tournament(fits)]
                lo, hi = np.minimum(a, b2), np.maximum(a, b2)
                span = hi - lo
                child = rng.uniform(lo - 0.3 * span, hi + 0.3 * span)
                mask = rng.random(len(CRITERIA)) < 0.5
                child = child + mask * rng.normal(0.0, sigma, len(CRITERIA))
                children.append(np.clip(child, -1, 1))
            pop = elites + children
        # ---- final selection on fresh worlds
        finalists = [pop[i].copy() for i in order[:args.top]]
        val_worlds = [args.world_seed + 10_000 + i for i in range(args.val_worlds)]
        grouped = run_batch(pool, finalists + [ref], val_worlds, "val", args.gens)
        ref_rows = grouped[-1]
        val_fits = [fitness(rows, ref_rows) for rows in grouped[:-1]]
    hist.close()
    best = int(np.argmax(val_fits))
    chosen = finalists[best]
    val_kpis = {m: float(np.mean([r[m] for r in grouped[best]])) for m in KPI_KEYS}
    ref_kpis = {m: float(np.mean([r[m] for r in ref_rows])) for m in KPI_KEYS}
    for k, (w, f) in enumerate(zip(finalists, val_fits)):
        say(f"finalist {k}: val {f:+.4f} (train {fits[order[k]]:+.4f}) w = " + " ".join(f"{x:+.3f}" for x in w))
    say(f"CHOSEN finalist {best}: val fitness {val_fits[best]:+.4f}; val KPIs "
        + " ".join(f"{m}={val_kpis[m]:.1f}" for m in KPI_KEYS)
        + " vs ref " + " ".join(f"{m}={ref_kpis[m]:.1f}" for m in KPI_KEYS))
    result = {
        "criteria": list(CRITERIA),
        "weights": [float(x) for x in chosen],
        "semantics": "magnitude = TOPSIS importance; sign > 0 cost (lower better), < 0 benefit (higher better)",
        "reference_weights": list(REFERENCE),
        "fitness_train": float(fits[order[best]]),
        "fitness_val": float(val_fits[best]),
        "val_kpis": val_kpis,
        "ref_val_kpis": ref_kpis,
        "finalists": [{"weights": [float(x) for x in w], "fitness_val": float(f)} for w, f in zip(finalists, val_fits)],
        "tuner": {"pop": args.pop, "gens": args.gens, "worlds": args.worlds, "val_worlds": args.val_worlds,
                  "seed": args.seed, "world_seed": args.world_seed, "sim_min": args.sim_min,
                  "sim_max": args.sim_max, "steps_cap": args.steps_cap, "env_config": str(TRAIN_CFG.relative_to(ROOT)),
                  "fitness": "mean relative improvement over Emp-TOPSIS on products, MTTR, disr/1k, final knowledge"},
        "generations": generations,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
