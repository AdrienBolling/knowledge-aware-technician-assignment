"""Sanity checks for vectorised parallel KataEnvs.

Verifies, empirically, that parallel SimPy simulators are fully
isolated:

1. **Solo == vector**: an env stepped alone and the same-seeded env-0
   inside an async 4-worker pool produce bit-identical trajectories
   (fingerprint over rewards and observation checksums).
2. **Twins**: two identically-seeded workers inside one pool produce
   identical fingerprints (no cross-talk), while differently-seeded
   workers produce different ones (no accidental seed sharing).
3. **Scaling**: wall-clock for N parallel episodes vs N sequential.

Usage::

    uv run python scripts/sanity_vec_env.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")

import gymnasium as gym
import numpy as np

from experiment.vec_env import _WorkerFactory, unbatch_obs

ENV_CFG = "run_configs/benchmark_suite/baseline.json"
VOCAB_JSON = "run_configs/vocab/set_vocab.json"
SIM, STEPS = 4_000.0, 400
RESET_SEED = 977


def _cfg_json(sim=None, steps=None) -> str:
    sim = SIM if sim is None else sim
    steps = STEPS if steps is None else steps
    cfg = json.loads(Path(ENV_CFG).read_text())
    cfg.setdefault("gym", {})
    cfg["gym"]["max_sim_time"] = sim
    cfg["gym"]["max_episode_steps"] = steps
    return json.dumps(cfg)


def _vocab() -> dict[str, int] | None:
    p = Path(VOCAB_JSON)
    if not p.is_file():
        return None
    from kata.core.tokenizer import StateTokenizer

    return StateTokenizer.from_json(p, seq_length=64).get_vocab()


def fingerprint_solo(worker_idx: int, base_seed: int, seed: int) -> tuple:
    """Run one scripted episode on a standalone worker env."""
    factory = _WorkerFactory(_cfg_json(), "set_transformer", worker_idx, base_seed, _vocab())
    env = factory()
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.n
    total_r, obs_sum, steps = 0.0, 0.0, 0
    while True:
        a = steps % n_act
        obs, r, term, trunc, info = env.step(a)
        total_r += float(r)
        obs_sum += float(np.sum(obs["env_cont_values"]))
        steps += 1
        if term or trunc:
            break
    fp = (steps, round(total_r, 6), round(obs_sum, 3))
    env.close()
    return fp


def fingerprints_vector(worker_idxs: list[int], base_seed: int, seeds: list[int]) -> list[tuple]:
    """Run scripted episodes on an async pool; fingerprint each env's first episode."""
    fns = [
        _WorkerFactory(_cfg_json(), "set_transformer", w, base_seed, _vocab())
        for w in worker_idxs
    ]
    venv = gym.vector.AsyncVectorEnv(fns, shared_memory=False)
    n = len(worker_idxs)
    obs, _ = venv.reset(seed=seeds)
    n_act = int(venv.single_action_space.n)
    total_r = np.zeros(n)
    obs_sum = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    finished = np.zeros(n, dtype=bool)
    t = 0
    while not finished.all():
        actions = np.full(n, t % n_act, dtype=np.int64)
        obs, r, term, trunc, _info = venv.step(actions)
        per_env = unbatch_obs(obs, n)
        for i in range(n):
            if finished[i]:
                continue
            total_r[i] += float(r[i])
            obs_sum[i] += float(np.sum(per_env[i]["env_cont_values"]))
            steps[i] += 1
            if term[i] or trunc[i]:
                finished[i] = True
        t += 1
    venv.close()
    return [
        (int(steps[i]), round(float(total_r[i]), 6), round(float(obs_sum[i]), 3))
        for i in range(n)
    ]


def main() -> int:
    ok = True
    base_seed = 42

    print("== 1. solo vs vector env-0 (same worker seed, same reset seed) ==")
    solo = fingerprint_solo(0, base_seed, RESET_SEED)
    vec = fingerprints_vector([0, 1, 2, 3], base_seed, [RESET_SEED, 11, 12, 13])
    print(f"   solo      : {solo}")
    print(f"   vector[0] : {vec[0]}")
    match = solo == vec[0]
    ok &= match
    print(f"   -> {'IDENTICAL' if match else 'MISMATCH (cross-talk or nondeterminism!)'}")

    print("== 2a. twins: two workers, same seeds -> must be identical ==")
    twins = fingerprints_vector([0, 0], base_seed, [RESET_SEED, RESET_SEED])
    same = twins[0] == twins[1]
    ok &= same
    print(f"   {twins[0]} vs {twins[1]} -> {'IDENTICAL' if same else 'MISMATCH'}")

    print("== 2b. different seeds -> must differ ==")
    differ = len({str(f) for f in vec}) == len(vec)
    ok &= differ
    print(f"   fingerprints: {vec}")
    print(f"   -> {'ALL DISTINCT' if differ else 'UNEXPECTED DUPLICATES'}")

    print("== 3. wall-clock scaling (4 episodes at 60k sim time, informational) ==")
    global SIM, STEPS
    SIM, STEPS = 60_000.0, 6_000
    t0 = time.time()
    for i in range(4):
        fingerprint_solo(i, base_seed, RESET_SEED + i)
    seq = time.time() - t0
    t0 = time.time()
    fingerprints_vector([0, 1, 2, 3], base_seed, [RESET_SEED + i for i in range(4)])
    par = time.time() - t0
    print(f"   sequential: {seq:.1f}s   async pool: {par:.1f}s   speedup: {seq / par:.2f}x")

    print(f"\nRESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
