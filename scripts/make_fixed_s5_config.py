"""Write the S5 benchmark factory as a static training config.

The benchmark samples the S5 layout from eval seed 20260722 (the first draw of
RandomScenarioSampler in build_scenario).  Training envs sample new layouts,
so a fine-tune on that factory needs the draw frozen into a static config.
This script takes the sampled KATAConfig, disables randomized_scenario, sets
the training reward of the quality fine-tune (train_multiscale_v5.json with
repair_quality 2.5), pads the action space for vector envs, and writes
run_configs/benchmark_suite/lifecycle_fixed_s5.json.

--verify checks that the written config builds the same factory and, for a
short deterministic rollout of the checkpoint, gives the same KPIs as the
benchmark harness.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")

import eval_human_vs_performance as ev  # noqa: E402
from live_finetune_s5 import training_reward  # noqa: E402

OUT = Path("run_configs/benchmark_suite/lifecycle_fixed_s5.json")


def machines_by_type(built):
    for part in built:
        for obj in (part, *vars(part).values()) if hasattr(part, "__dict__") else (part,):
            if hasattr(obj, "machines_by_type"):
                return {t: len(v) for t, v in obj.machines_by_type.items()}
    raise RuntimeError("machines_by_type not found in the built scenario")


def rollout_kpis(gym_cfg, factory, ckpt: Path, seed: int) -> dict:
    """Deterministic benchmark rollout of ``ckpt`` on ``factory`` under ``gym_cfg``."""
    cfg = SimpleNamespace(gym=gym_cfg)  # the harness helpers read only ``.gym``
    tok = ev.load_set_tokenizer(ckpt, cfg)
    env = ev.make_env(cfg, factory, "set", tokenizer=tok, legacy_obs=False)
    base = ev.AgentConfig(**json.loads(Path("run_configs/agents/set_transformer_v6.json").read_text()))
    params, _ = ev.set_transformer_params(base.params, cfg, tok, ckpt)
    agent = ev.SetTransformerAgent(**params)
    agent.load(str(ckpt))
    agent.net.eval()
    kpis, _ = ev.run_episode(agent, env, seed=seed, deterministic=True, record_every=10**9)
    return {k: kpis.get(k) for k in ("n_steps", "finished_products", "total_repairs",
                                     "ill_technician_count", "total_breakdowns")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-seed", type=int, default=20260722)
    ap.add_argument("--train-config", default="run_configs/benchmark_suite/train_multiscale_v5.json")
    ap.add_argument("--repair-quality-coef", type=float, default=2.5)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-steps", type=int, default=3000)
    ap.add_argument("--checkpoint", default="checkpoints/ft_quality/set_transformer_best.pt")
    args = ap.parse_args()

    ev.EVAL_SEED = int(args.eval_seed)
    env_cfg, factory, *_ = ev.build_scenario("lifecycle")
    fixed = factory.__defaults__[0]
    if not isinstance(fixed, ev.KATAConfig):
        raise SystemExit(f"unexpected sampled config type {type(fixed)}")
    gym = fixed.gym.model_copy(update={
        "reward": training_reward(args.train_config, args.repair_quality_coef),
        "pad_action_space_to_max_techs": True,
        "observation_representation": "set",
        "max_sim_time": float(env_cfg.gym.max_sim_time),
        "max_episode_steps": int(env_cfg.gym.max_episode_steps),
    })
    rs = fixed.randomized_scenario.model_copy(update={"enabled": False})
    static = fixed.model_copy(update={"gym": gym, "randomized_scenario": rs})
    Path(args.out).write_text(json.dumps(static.model_dump(mode="json"), indent=2) + "\n")
    n_events = len(static.gym.lifecycle_events or [])
    print(f"written {args.out}: horizon {static.gym.max_sim_time:.0f}, "
          f"{n_events} lifecycle events, randomized_scenario.enabled={rs.enabled}")

    if args.verify:
        loaded = ev.KATAConfig(**json.loads(Path(args.out).read_text()))
        a = machines_by_type(ev.ScenarioBuilder(fixed).build())
        b = machines_by_type(ev.ScenarioBuilder(loaded).build())
        print(f"layout equal: {a == b}  ({sum(a.values())} machines)")
        seed = ev.EVAL_SEED * 100
        ckpt = Path(args.checkpoint)
        bench_gym = env_cfg.gym.model_copy(update={"max_episode_steps": args.verify_steps})
        static_gym = loaded.gym.model_copy(update={"max_episode_steps": args.verify_steps})
        k_bench = rollout_kpis(bench_gym, factory, ckpt, seed)
        k_static = rollout_kpis(static_gym, lambda c=loaded: ev.ScenarioBuilder(c).build(),
                                ckpt, seed)
        print(f"benchmark env: {k_bench}")
        print(f"static config: {k_static}")
        ok = (a == b) and (k_bench == k_static)
        print("VERIFY OK" if ok else "VERIFY FAILED")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
