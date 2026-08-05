"""Sanity checks for vectorised parallel KataEnvs.

Verifies, empirically, that parallel SimPy simulators are fully
isolated:

1. **Solo == vector**: an env stepped alone and the same-seeded env-0
   inside an async 4-worker pool produce bit-identical trajectories
   (fingerprint over rewards and observation checksums).
2. **Twins**: two identically-seeded workers inside one pool produce
   identical fingerprints (no cross-talk), while differently-seeded
   workers produce different ones (no accidental seed sharing).

Checks 1 and 2 fingerprint the first *two* episodes of every env
separately, stepping through the NEXT_STEP autoreset: the post-autoreset
episode is only reproducible because ``SeededResetWrapper`` derives a
seed for unseeded resets instead of forwarding ``None``.
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


def fingerprint_solo(
    worker_idx: int, base_seed: int, seed: int, episodes: int = 2
) -> list[tuple]:
    """Run scripted episodes on a standalone worker env; fingerprint each.

    Only the first reset carries ``seed``; the later ones are bare, which
    is exactly what the async worker does on autoreset.  The action
    schedule follows the pool's *global* step counter (one step slot is
    burnt by the autoreset itself) so the scripted actions line up with
    :func:`fingerprints_vector`.
    """
    factory = _WorkerFactory(_cfg_json(), "set_transformer", worker_idx, base_seed, _vocab())
    env = factory()
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.n
    fps: list[tuple] = []
    t = 0
    for ep in range(episodes):
        if ep > 0:
            t += 1  # the autoreset consumes a step slot in the pooled run
            env.reset()
        total_r, obs_sum, steps = 0.0, 0.0, 0
        while True:
            obs, r, term, trunc, info = env.step(t % n_act)
            total_r += float(r)
            obs_sum += float(np.sum(obs["env_cont_values"]))
            steps += 1
            t += 1
            if term or trunc:
                break
        fps.append((steps, round(total_r, 6), round(obs_sum, 3)))
    env.close()
    return fps


def fingerprints_vector(
    worker_idxs: list[int], base_seed: int, seeds: list[int], episodes: int = 2
) -> list[list[tuple]]:
    """Run scripted episodes on an async pool; fingerprint each env's first
    ``episodes`` episodes separately."""
    fns = [
        _WorkerFactory(_cfg_json(), "set_transformer", w, base_seed, _vocab())
        for w in worker_idxs
    ]
    venv = gym.vector.AsyncVectorEnv(
        fns,
        shared_memory=False,
        context="fork",
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    n = len(worker_idxs)
    obs, _ = venv.reset(seed=seeds)
    n_act = int(venv.single_action_space.n)
    total_r = np.zeros((n, episodes))
    obs_sum = np.zeros((n, episodes))
    steps = np.zeros((n, episodes), dtype=int)
    ep = np.zeros(n, dtype=int)
    prev_done = np.zeros(n, dtype=bool)
    t = 0
    while (ep < episodes).any():
        actions = np.full(n, t % n_act, dtype=np.int64)
        obs, r, term, trunc, _info = venv.step(actions)
        per_env = unbatch_obs(obs, n)
        for i in range(n):
            if prev_done[i]:
                # NEXT_STEP autoreset: the action was ignored and this obs
                # is the next episode's start — nothing to record.
                prev_done[i] = False
                continue
            if ep[i] >= episodes:
                continue
            e = ep[i]
            total_r[i, e] += float(r[i])
            obs_sum[i, e] += float(np.sum(per_env[i]["env_cont_values"]))
            steps[i, e] += 1
            if term[i] or trunc[i]:
                prev_done[i] = True
                ep[i] += 1
        t += 1
    venv.close()
    return [
        [
            (
                int(steps[i, e]),
                round(float(total_r[i, e]), 6),
                round(float(obs_sum[i, e]), 3),
            )
            for e in range(episodes)
        ]
        for i in range(n)
    ]


def main() -> int:
    ok = True
    base_seed = 42

    print("== 1. solo vs vector env-0 (same worker seed, same reset seed) ==")
    solo = fingerprint_solo(0, base_seed, RESET_SEED)
    vec = fingerprints_vector([0, 1, 2, 3], base_seed, [RESET_SEED, 11, 12, 13])
    for e in range(len(solo)):
        print(f"   episode {e + 1}: solo {solo[e]} | vector[0] {vec[0][e]}")
    match = solo == vec[0]
    ok &= match
    print(f"   -> {'IDENTICAL' if match else 'MISMATCH (cross-talk or nondeterminism!)'}")

    print("== 2a. twins: two workers, same seeds -> must be identical ==")
    twins = fingerprints_vector([0, 0], base_seed, [RESET_SEED, RESET_SEED])
    same = twins[0] == twins[1]
    ok &= same
    for e in range(len(twins[0])):
        print(f"   episode {e + 1}: {twins[0][e]} vs {twins[1][e]}")
    print(f"   -> {'IDENTICAL' if same else 'MISMATCH'}")

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
        fingerprint_solo(i, base_seed, RESET_SEED + i, episodes=1)
    seq = time.time() - t0
    t0 = time.time()
    fingerprints_vector(
        [0, 1, 2, 3], base_seed, [RESET_SEED + i for i in range(4)], episodes=1
    )
    par = time.time() - t0
    print(f"   sequential: {seq:.1f}s   async pool: {par:.1f}s   speedup: {seq / par:.2f}x")

    print(f"\nRESULT: {'ALL CHECKS PASSED' if ok else 'FAILURES DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
