"""Live fine-tuning of HTT-RL during one S5 benchmark episode.

Deploys a checkpoint on the S5 lifecycle benchmark (fixed factory from eval
seed 20260722, episode seed 2026072200) and runs the benchmark episode with
``run_episode`` of scripts/eval_human_vs_performance.py, so the KPI accounting
is the benchmark's.  With ``--updates on`` the agent also stores every
transition and runs a PPO update every ``--rollout`` decisions (the value is
bootstrapped at rollout ends that are not terminal), so the policy keeps
learning while it dispatches.

The updates use the training reward of the quality fine-tune: the reward stack
of train_multiscale_v5.json with the repair-quality coefficient raised to 2.5,
with running component normalization (not frozen).  The reward does not change
the simulation, so the KPIs stay comparable with the benchmark row.  The PPO
settings follow the fine-tune recipe (scripts/dgy_v6_ft_queue.sh): lr 3e-5,
PopArt, gamma 0.9999 per time unit, lambda 0.98, 4 epochs, minibatch 256.

Runs of the study
  --updates on  --act stochastic     live fine-tuning (PPO samples its actions)
  --updates off --act stochastic     control: sampled actions, no learning
  --updates on  --act deterministic  live fine-tuning with argmax dispatching

Outputs: <out-root>/lifecycle/{episodes.csv, steps.csv.gz, updates.csv,
manifest.json} and <checkpoint-dir>/set_transformer_{uNNNN,final}.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("KATA_CONF_PATH", "/dev/null/__no_file__")

import eval_human_vs_performance as ev  # noqa: E402


def training_reward(train_config: str, repair_quality_coef: float):
    """Reward stack of the training config, with the quality lever applied."""
    train = ev.KATAConfig(**json.loads(Path(train_config).read_text()))
    reward = train.gym.reward
    rq = reward.repair_quality.model_copy(update={"coefficient": repair_quality_coef})
    return reward.model_copy(update={"repair_quality": rq})


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="agent label in episodes.csv")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/ft_quality/set_transformer_best.pt")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: checkpoints/<label>")
    ap.add_argument("--updates", choices=["on", "off"], default="on")
    ap.add_argument("--act", choices=["stochastic", "deterministic"], default="stochastic")
    ap.add_argument("--eval-seed", type=int, default=20260722)
    ap.add_argument("--agent-config", default="run_configs/agents/set_transformer_v6.json")
    ap.add_argument("--train-config", default="run_configs/benchmark_suite/train_multiscale_v5.json")
    ap.add_argument("--repair-quality-coef", type=float, default=2.5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--total-updates", type=int, default=300,
                    help="cosine schedule length (S5 has ~560k decisions = ~274 updates)")
    ap.add_argument("--warmup-updates", type=int, default=10)
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25, help="updates between checkpoints")
    ap.add_argument("--progress-every", type=int, default=50_000, help="decisions between log lines")
    ap.add_argument("--record-every", type=int, default=200)
    ap.add_argument("--sim", type=float, default=None, help="horizon override (smoke tests only)")
    ap.add_argument("--steps", type=int, default=None, help="step cap override (smoke tests only)")
    args = ap.parse_args()

    live = args.updates == "on"
    ckpt = Path(args.checkpoint)
    ck_dir = Path(args.checkpoint_dir or f"checkpoints/{args.label}")
    out_dir = Path(args.out_root) / "lifecycle"
    out_dir.mkdir(parents=True, exist_ok=True)
    if live:
        ck_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark factory and horizon; training reward on top.
    ev.EVAL_SEED = int(args.eval_seed)
    env_cfg, factory, n_techs, _, _, _ = ev.build_scenario(
        "lifecycle", sim_override=args.sim, steps_override=args.steps)
    env_cfg = env_cfg.model_copy(update={"gym": env_cfg.gym.model_copy(update={
        "reward": training_reward(args.train_config, args.repair_quality_coef)})})
    tok = ev.load_set_tokenizer(ckpt, env_cfg)
    env = ev.make_env(env_cfg, factory, "set", tokenizer=tok, legacy_obs=False)

    # Agent: harness architecture injection, then the fine-tune PPO recipe.
    base = ev.AgentConfig(**json.loads(Path(args.agent_config).read_text()))
    params, imp = ev.set_transformer_params(base.params, env_cfg, tok, ckpt)
    if "slot_role_binding" not in imp:
        raise SystemExit(f"{ckpt}: pre-v6 checkpoint (legacy observation) is not supported")
    params.update({
        "use_popart": True, "normalize_rewards": False, "rnn_type": "none",
        "gamma": 0.9999, "time_based_discount": True,
        "rollout_steps": int(args.rollout), "lr": float(args.lr),
        "total_updates": int(args.total_updates),
        "warmup_updates": int(args.warmup_updates),
    })
    torch.manual_seed(int(args.torch_seed))
    agent = ev.SetTransformerAgent(**params)
    agent.load(str(ckpt))
    agent.net.eval()  # dropout is 0 in the v6 config; updates do not toggle the mode

    counter = {"steps": 0, "updates": 0}
    upd_rows: list[dict] = []
    t_start = time.time()

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{args.label}] {msg}",
              flush=True)

    def on_step(prev_obs, action, reward, obs, term, trunc, info):
        counter["steps"] += 1
        done = bool(term or trunc)
        if counter["steps"] % args.progress_every == 0:
            log(f"step {counter['steps']} sim {float(info.get('sim_time', 0.0)):.0f} "
                f"products {ev.finished_products(env)} updates {counter['updates']} "
                f"elapsed {(time.time() - t_start) / 3600:.2f} h")
        if not live:
            return
        agent.observe_transition(prev_obs, action, reward, obs, term, trunc, info)
        if counter["steps"] % args.rollout != 0 and not done:
            return
        t0 = time.time()
        stats = agent.update()
        counter["updates"] += 1
        row = {"update": counter["updates"], "step": counter["steps"],
               "sim_time": float(info.get("sim_time", 0.0)),
               "update_s": round(time.time() - t0, 2),
               "elapsed_h": round((time.time() - t_start) / 3600, 3)}
        for k, v in (stats or {}).items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        upd_rows.append(row)
        if counter["updates"] % args.save_every == 0 or done:
            pd.DataFrame(upd_rows).to_csv(out_dir / "updates.csv", index=False)
            agent.save(str(ck_dir / f"set_transformer_u{counter['updates']:04d}.pt"))
            log(f"update {counter['updates']} step {counter['steps']} "
                f"lr {row.get('lr', float('nan')):.2e} kl {row.get('approx_kl', float('nan')):.4f} "
                f"entropy {row.get('entropy', float('nan')):.3f} vf {row.get('vf_loss', float('nan')):.4f}")

    seed = ev.EVAL_SEED * 100 + 0
    log(f"start: updates={args.updates} act={args.act} seed={seed} ckpt={ckpt} "
        f"horizon={env_cfg.gym.max_sim_time:.0f} lr={args.lr} rollout={args.rollout}")
    kpis, records = ev.run_episode(
        agent, env, seed=seed, deterministic=(args.act == "deterministic"),
        record_every=int(args.record_every), on_step=on_step,
        freeze_normalizer=not live)
    wall = time.time() - t_start
    if live:
        agent.save(str(ck_dir / "set_transformer_final.pt"))
        pd.DataFrame(upd_rows).to_csv(out_dir / "updates.csv", index=False)

    pd.DataFrame([{"agent": args.label, "episode": 0, "wall_s": round(wall, 1), **kpis}]
                 ).to_csv(out_dir / "episodes.csv", index=False)
    steps = pd.DataFrame(records)
    steps.insert(0, "agent", args.label)
    steps.insert(1, "episode", 0)
    steps.to_csv(out_dir / "steps.csv.gz", index=False, compression="gzip")
    try:
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {
        "scenario": "lifecycle", "label": args.label, "args": vars(args),
        "episode_seed": seed, "checkpoint_md5": md5(ckpt), "code_commit": commit,
        "n_updates": counter["updates"], "n_steps": kpis.get("n_steps"),
        "wall_s": round(wall, 1), "agent_params": {k: v for k, v in params.items()
                                                    if isinstance(v, (int, float, str, bool))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"done: products {kpis.get('finished_products')} mttr {kpis.get('mttr')} "
        f"updates {counter['updates']} wall {wall / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
