"""Benchmark the human vs performance SetTransformer checkpoints.

Runs both trained checkpoints plus every heuristic baseline on one
benchmark-suite scenario, recording *time-stamped* per-step series so the
results can be sliced into within-episode horizon windows (short /
medium / long) afterwards.  Modeled on the per-scenario notebooks in
``benchmarks/`` (same fixed-eval-scenario construction, same seeds), but
extended with sim-time stamps, cumulative throughput, availability, and
fleet knowledge / fatigue series at every decision step.

Usage (from the repo root)::

    uv run python scripts/eval_human_vs_performance.py --scenario baseline
    uv run python scripts/eval_human_vs_performance.py --scenario all

Artefacts land in ``reports/hvp_eval/<scenario>/``:

* ``episodes.csv``  — one row per (agent, episode) with end-of-episode KPIs
* ``steps.csv.gz``  — long-format per-step series for every agent/episode
* ``manifest.json`` — full provenance
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from kata.core.config import KATAConfig
from kata.core.tokenizer import StateTokenizer
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder
from kata.EntityFactories import RandomScenarioSampler
from experiment.config import AgentConfig
from agents import (
    GreedyRewardAgent,
    LeastBusyAgent,
    LeastFatiguedAgent,
    OptimalAssignmentAgent,
    RandomAgent,
    RoundRobinAgent,
    SetTransformerAgent,
    ShortestProcessingTimeAgent,
    ShortestQueueAgent,
    TopsisAgent,
    TrainWeakestAgent,
)

HEURISTICS = {
    "random": RandomAgent,
    "round_robin": RoundRobinAgent,
    "least_busy": LeastBusyAgent,
    "least_fatigued": LeastFatiguedAgent,
    "shortest_queue": ShortestQueueAgent,
    # Skill / optimisation / multi-criteria / reward-greedy / upskilling
    # baselines drawn from the nearest works in the survey taxonomy; all
    # read the env's decision-support API (expected repair times, skill
    # match, workload counts, or the counterfactual per-assignment reward).
    "shortest_processing": ShortestProcessingTimeAgent,
    "optimal_assignment": OptimalAssignmentAgent,
    "topsis": TopsisAgent,
    "greedy_reward": GreedyRewardAgent,
    "train_weakest": TrainWeakestAgent,
}

CHECKPOINTS = {
    "human": Path("checkpoints/human_set_transformer_best.pt"),
    "performance": Path("checkpoints/performance_set_transformer_best.pt"),
    # Second-generation human-centric agent (long-horizon gamma/lambda +
    # PopArt + GRU, trained with the opt-in improvement stack).  Its
    # architecture is read from the checkpoint's own ``improvements`` dict
    # by build_agents, so no separate agent config is needed here.
    "hc_v2": Path("checkpoints/hc_v2/set_transformer_best.pt"),
}

AGENT_CONFIG = Path("run_configs/agents/set_transformer.json")

# Horizon profiles mirror benchmarks/_generate.py (Part 1 numbers).
SCENARIOS = {
    "baseline": dict(
        cfg="run_configs/benchmark_suite/baseline.json",
        n_eps=5, sim=200_000.0, steps=20_000,
    ),
    "small_scale": dict(
        cfg="run_configs/benchmark_suite/small_scale.json",
        n_eps=3, sim=200_000.0, steps=20_000,
    ),
    "massive_scale": dict(
        cfg="run_configs/benchmark_suite/massive_scale.json",
        n_eps=1, sim=100_000.0, steps=10_000,
    ),
    # Very-long horizon study: same industrial layout, 50x the horizon.
    # One episode per agent; step cap sized to stay non-binding.
    "very_long": dict(
        cfg="run_configs/benchmark_suite/massive_scale.json",
        n_eps=1, sim=5_000_000.0, steps=1_500_000,
    ),
}

EVAL_SEED = 4321


@contextmanager
def quiet():
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def build_scenario(scenario: str, *, n_eps_override: int | None = None,
                   sim_override: float | None = None,
                   steps_override: int | None = None):
    prof = SCENARIOS[scenario]
    env_cfg = KATAConfig(**json.loads(Path(prof["cfg"]).read_text()))
    sim = sim_override if sim_override is not None else prof["sim"]
    steps = steps_override if steps_override is not None else prof["steps"]
    n_eps = n_eps_override if n_eps_override is not None else prof["n_eps"]
    env_cfg.gym = env_cfg.gym.model_copy(update={
        "max_episode_steps": steps,
        "max_sim_time": sim,
    })
    rcfg = env_cfg.randomized_scenario
    assert rcfg.enabled, f"{scenario}: expected randomized_scenario.enabled"
    sampler = RandomScenarioSampler(env_cfg, rcfg, seed=EVAL_SEED)
    fixed_eval_cfg = sampler.sample_config()

    def scenario_factory(c=fixed_eval_cfg):
        return ScenarioBuilder(c).build()

    n_techs = rcfg.n_technicians
    machine_types = sampler.all_machine_types()
    component_types = sampler.all_component_types()
    return env_cfg, scenario_factory, n_techs, machine_types, component_types, n_eps


def load_set_tokenizer(checkpoint: Path, env_cfg) -> StateTokenizer:
    embedded = SetTransformerAgent.peek_vocab(checkpoint)
    if embedded:
        tok = StateTokenizer(seq_length=env_cfg.gym.tokenizer_seq_length)
        tok.load_vocab(embedded)
        tok.freeze()
        return tok
    vocab_path = getattr(env_cfg.gym, "set_vocab_path", None)
    if vocab_path and Path(vocab_path).is_file():
        return StateTokenizer.from_json(
            Path(vocab_path), seq_length=env_cfg.gym.tokenizer_seq_length
        )
    raise RuntimeError(f"no vocab for checkpoint {checkpoint}")


def make_env(env_cfg, scenario_factory, representation, tokenizer=None) -> KataEnv:
    gym_cfg = env_cfg.gym.model_copy(
        update={"observation_representation": representation}
    )
    with quiet():
        return KataEnv(
            scenario_factory=scenario_factory,
            config=gym_cfg,
            tokenizer=tokenizer,
        )


def peek_improvements(checkpoint: Path) -> dict:
    """Return the opt-in improvement config embedded in a checkpoint.

    Checkpoints saved by the improved PPO agent carry an
    ``improvements`` dict (``rnn_type``, ``rnn_hidden``, ``use_popart``,
    ...); historical checkpoints have none.  We read it so the eval-time
    agent is built with the *same architecture* it was trained with,
    rather than a fixed config that would only partially load a
    GRU/PopArt checkpoint.
    """
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return dict(ck.get("improvements") or {}) if isinstance(ck, dict) else {}


def build_agents(env_cfg, scenario_factory, n_techs):
    """Return {label: (agent, env)} for the trained checkpoints + 5 heuristics.

    Each trained agent is instantiated with the architecture recorded in
    its own checkpoint (plain set-transformer, or the GRU/PopArt variant),
    so heterogeneous checkpoints can be benchmarked side by side.
    """
    agents: dict[str, tuple] = {}
    agent_cfg = AgentConfig(**json.loads(AGENT_CONFIG.read_text()))
    for label, ckpt in CHECKPOINTS.items():
        if not ckpt.is_file():
            print(f"  (skipping {label}: no checkpoint at {ckpt})", flush=True)
            continue
        tok = load_set_tokenizer(ckpt, env_cfg)
        params = dict(agent_cfg.params)
        params["n_actions"] = int(env_cfg.gym.max_techs)
        params.setdefault("max_techs", int(env_cfg.gym.max_techs))
        params.setdefault("max_machines", int(env_cfg.gym.max_machines))
        params.setdefault("env_length", int(env_cfg.gym.set_env_length))
        params.setdefault("sim_time_scale", float(env_cfg.gym.max_sim_time))
        params.setdefault("vocab_size", tok.vocab_size)
        # Match the checkpoint's trained architecture (opt-in improvements).
        imp = peek_improvements(ckpt)
        rnn_type = str(imp.get("rnn_type", "none") or "none")
        if rnn_type != "none":
            params["rnn_type"] = rnn_type
            params["rnn_hidden"] = int(imp.get("rnn_hidden", 128))
        if imp.get("use_popart"):
            params["use_popart"] = True
            params["normalize_rewards"] = False  # mutually exclusive
        agent = SetTransformerAgent(**params)
        agent.load(ckpt)
        env = make_env(env_cfg, scenario_factory, "set", tokenizer=tok)
        agents[label] = (agent, env)
    for name, cls in HEURISTICS.items():
        env = make_env(env_cfg, scenario_factory, "structured")
        agent = cls(n_actions=n_techs)
        # Skill/optimisation baselines read the env's decision-support API
        # (expected repair times, batch cost matrix); harmless no-op for the
        # obs-only rules.
        agent.attach_env(env)
        agents[name] = (agent, env)
    return agents


def fleet_fatigue(env) -> tuple[float, float]:
    f = np.array(
        [float(getattr(t, "fatigue", 0.0)) for t in env.dispatcher.techs],
        dtype=np.float64,
    )
    return (float(f.mean()), float(f.std())) if f.size else (0.0, 0.0)


def finished_products(env) -> int:
    sinks = getattr(env.dispatcher, "sinks", []) or []
    return sum(int(getattr(s, "completed", 0)) for s in sinks)


def run_episode(agent, env, *, seed: int, deterministic: bool = True,
                record_every: int = 1):
    """One rollout; returns (kpis_dict, step_records_list).

    ``record_every`` subsamples the per-step series (every N-th decision
    plus the terminal one); episode KPIs are unaffected.  Use for very
    long episodes where every-step records would dominate memory.
    """
    np.random.seed(seed)  # heuristic tiebreaks / RandomAgent
    agent.on_episode_start()
    with quiet():
        obs, _ = env.reset(seed=seed)
    if hasattr(env, "freeze_reward_normalizer"):
        env.freeze_reward_normalizer()

    records = []
    ep_reward, n_steps = 0.0, 0
    final_info: dict = {}
    while True:
        action = agent.select_action(obs, deterministic=deterministic)
        with quiet():
            obs, reward, term, trunc, info = env.step(action)
        ep_reward += float(reward)
        n_steps += 1
        if (term or trunc) or (n_steps % record_every == 0):
            m = info.get("metrics", {})
            fat_mean, fat_std = fleet_fatigue(env)
            records.append({
                "step": n_steps,
                "sim_time": float(info.get("sim_time", 0.0)),
                "mttr_rolling": float(m.get("mttr_rolling", np.nan)),
                "repair_quality": float(m.get("repair_quality", np.nan)),
                "repair_time_delta_per": float(m.get("repair_time_delta_per", np.nan)),
                "queue_size": int(info.get("pending_queue_size", 0)),
                "finished_products": finished_products(env),
                "fleet_availability": float(env._fleet_availability_raw()),
                "fleet_knowledge": float(env._fleet_mean_knowledge_volume()),
                "fatigue_mean": fat_mean,
                "fatigue_std": fat_std,
            })
        if term or trunc:
            final_info = info
            break
    agent.on_episode_end(ep_reward)

    kpis = {
        k: float(v)
        for k, v in final_info.get("metrics", {}).items()
        if "/" not in k
    }
    kpis["episode_reward"] = ep_reward
    kpis["n_steps"] = n_steps
    kpis["final_sim_time"] = float(final_info.get("sim_time", 0.0))
    return kpis, records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    choices=[*SCENARIOS.keys(), "all"])
    ap.add_argument("--n-eps", type=int, default=None)
    ap.add_argument("--sim", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--agents", default="all",
                    help="comma-separated agent keys to run "
                         "(human,performance,random,round_robin,least_busy,"
                         "least_fatigued,shortest_queue), 'trained', "
                         "'heuristics', or 'all'")
    ap.add_argument("--merge", action="store_true",
                    help="merge results into existing artifacts in the "
                         "output dir: rows of the re-run agents are "
                         "replaced, all other agents' rows are kept")
    ap.add_argument("--record-every", type=int, default=1)
    ap.add_argument("--out-root", default="reports/hvp_eval")
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for scenario in names:
        t_scenario = time.time()
        print(f"\n================ {scenario} ================", flush=True)
        env_cfg, factory, n_techs, mtypes, ctypes, n_eps = build_scenario(
            scenario, n_eps_override=args.n_eps,
            sim_override=args.sim, steps_override=args.steps,
        )
        agents = build_agents(env_cfg, factory, n_techs)
        if args.agents != "all":
            if args.agents == "trained":
                keep = set(CHECKPOINTS)
            elif args.agents == "heuristics":
                keep = set(HEURISTICS)
            else:
                keep = {k.strip() for k in args.agents.split(",")}
            unknown = keep - set(agents)
            if unknown:
                raise SystemExit(f"unknown agent keys: {sorted(unknown)}")
            agents = {k: v for k, v in agents.items() if k in keep}
        print(f"agents: {list(agents)} | n_techs={n_techs} "
              f"| horizon sim={env_cfg.gym.max_sim_time:.0f} "
              f"steps={env_cfg.gym.max_episode_steps} | episodes={n_eps}",
              flush=True)

        out_dir = Path(args.out_root) / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_rows, step_frames = [], []
        for label, (agent, env) in agents.items():
            for ep in range(n_eps):
                seed = EVAL_SEED * 100 + ep
                t0 = time.time()
                kpis, records = run_episode(agent, env, seed=seed,
                                            record_every=args.record_every)
                dt = time.time() - t0
                ep_rows.append({"agent": label, "episode": ep,
                                "wall_s": round(dt, 1), **kpis})
                df = pd.DataFrame(records)
                df.insert(0, "agent", label)
                df.insert(1, "episode", ep)
                step_frames.append(df)
                print(f"  {label:<14s} ep{ep}  steps={kpis['n_steps']:>6d}  "
                      f"finished={kpis.get('finished_products', float('nan')):>7.0f}  "
                      f"mttr={kpis.get('mttr', float('nan')):>7.2f}  "
                      f"avail={kpis.get('fleet_availability_rate', float('nan')):.3f}  "
                      f"[{dt:.1f}s]", flush=True)

        ep_df = pd.DataFrame(ep_rows)
        steps_df = pd.concat(step_frames, ignore_index=True)
        if args.merge and (out_dir / "episodes.csv").is_file():
            rerun = set(ep_df["agent"])
            old_ep = pd.read_csv(out_dir / "episodes.csv")
            ep_df = pd.concat(
                [old_ep[~old_ep["agent"].isin(rerun)], ep_df],
                ignore_index=True,
            )
            old_steps = pd.read_csv(out_dir / "steps.csv.gz")
            steps_df = pd.concat(
                [old_steps[~old_steps["agent"].isin(rerun)], steps_df],
                ignore_index=True,
            )
            print(f"  [merge] replaced rows for {sorted(rerun)}; "
                  f"kept {sorted(set(ep_df['agent']) - rerun)}", flush=True)
        ep_df.to_csv(out_dir / "episodes.csv", index=False)
        steps_df.to_csv(
            out_dir / "steps.csv.gz", index=False, compression="gzip"
        )
        manifest = {
            "scenario": scenario,
            "env_config": SCENARIOS[scenario]["cfg"],
            "agent_config": str(AGENT_CONFIG),
            "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
            "heuristics": list(HEURISTICS),
            "n_eval_episodes": n_eps,
            "max_eval_sim_time": env_cfg.gym.max_sim_time,
            "max_eval_steps": env_cfg.gym.max_episode_steps,
            "eval_seed": EVAL_SEED,
            "deterministic": True,
            "agents_run": list(agents),
            "merged": bool(args.merge),
            "record_every": args.record_every,
            "n_techs": n_techs,
            "machine_types": mtypes,
            "component_types": ctypes,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[{scenario}] done in {time.time() - t_scenario:.0f}s "
              f"-> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
