"""Behaviour-clone the set-transformer onto a strong heuristic teacher.

Warm-start for PPO: instead of learning from scratch (and losing to the
myopic baselines for most of training), clone the policy onto TOPSIS ---
the strongest multi-criteria dispatching rule --- then fine-tune with
PPO from that initialization.  PPO then only has to learn *when to
deviate* from the teacher (invest in a weak technician, spare a
fatigued one), which is exactly the long-horizon behaviour the thesis
is about.

Fairness note: the teacher is consulted only at data-collection time.
The cloned network conditions on the ordinary set observation --- the
same input the RL agent always sees (which already carries the ETA
signal TOPSIS ranks on) --- so the deployed policy reads no oracle
information at inference.

Usage (from the repo root)::

    uv run python scripts/warmstart_bc.py \
        --episodes 30 --out checkpoints/bc_topsis/set_transformer_bc.pt
    uv run python scripts/train_hc_improved.py \
        --init-checkpoint checkpoints/bc_topsis/set_transformer_bc.pt ...

The checkpoint is written with the agent's own ``save`` so the trainer's
``--init-checkpoint`` path loads it like any other checkpoint (vocab
embedded, architecture recorded in ``improvements``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Behaviour-clone the set-transformer onto TOPSIS."
    )
    ap.add_argument("--env-config",
                    default="run_configs/benchmark_suite/baseline.json",
                    help="training env config (same file the PPO run uses)")
    ap.add_argument("--agent-config",
                    default="run_configs/agents/set_transformer.json")
    ap.add_argument("--episodes", type=int, default=30,
                    help="teacher episodes to collect (fresh layout each)")
    ap.add_argument("--sim-time", type=float, default=100_000.0,
                    help="horizon per collection episode")
    ap.add_argument("--max-steps", type=int, default=5_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.1,
                    help="fraction of steps held out for agreement eval")
    ap.add_argument("--no-popart", action="store_true",
                    help="build the student without the PopArt head "
                         "(match the PPO run you plan to init)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out",
                    default="checkpoints/bc_topsis/set_transformer_bc.pt")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    import torch.nn.functional as F

    from agents import SetTransformerAgent, TopsisAgent
    from kata.EntityFactories import RandomScenarioSampler
    from kata.core.config import KATAConfig
    from kata.core.tokenizer import StateTokenizer
    from kata.env import KataEnv
    from kata.funcs import seed_numba_rng
    from kata.scenario import ScenarioBuilder

    rng = np.random.default_rng(args.seed)

    # ----- env: the PPO training distribution (rotating random layouts) --
    env_cfg = KATAConfig(**json.loads(Path(args.env_config).read_text()))
    gym_cfg = env_cfg.gym.model_copy(update={
        "observation_representation": "set",
        "max_sim_time": float(args.sim_time),
        "max_episode_steps": int(args.max_steps),
    })
    rcfg = env_cfg.randomized_scenario
    assert rcfg.enabled, "expected randomized_scenario.enabled"
    sampler = RandomScenarioSampler(env_cfg, rcfg, seed=args.seed)

    def factory():
        # Fresh layout per reset — mirrors training's scenario rotation.
        return ScenarioBuilder(sampler.sample_config()).build()

    # Tokenizer: canonical vocab file if configured, else the same
    # deterministic rebuild the trainer performs (pure function of the
    # template pools, so the BC checkpoint and the PPO run share ids).
    vocab_path = getattr(gym_cfg, "set_vocab_path", None)
    if vocab_path and Path(vocab_path).is_file():
        tok = StateTokenizer.from_json(
            Path(vocab_path), seq_length=gym_cfg.tokenizer_seq_length
        )
    else:
        tok = StateTokenizer.build_set_vocab(
            machine_types=sampler.all_machine_types(),
            component_types=sampler.all_component_types(),
            technician_templates=list(rcfg.technician_templates),
            seq_length=gym_cfg.tokenizer_seq_length,
        )
    env = KataEnv(scenario_factory=factory, config=gym_cfg, tokenizer=tok)

    # ----- student: same construction as the eval/training harness ------
    agent_data = json.loads(Path(args.agent_config).read_text())
    params = dict(agent_data["params"])
    params["n_actions"] = int(gym_cfg.max_techs)
    params.setdefault("max_techs", int(gym_cfg.max_techs))
    params.setdefault("max_machines", int(gym_cfg.max_machines))
    params.setdefault("env_length", int(gym_cfg.set_env_length))
    params.setdefault("sim_time_scale", float(gym_cfg.max_sim_time))
    params["vocab_size"] = tok.vocab_size
    if not args.no_popart:
        params["use_popart"] = True
        params["normalize_rewards"] = False
    student = SetTransformerAgent(**params)
    # Embed the vocab in the checkpoint (as the trainer does) so
    # eval-time loads don't depend on rebuilding it elsewhere.
    student.attach_vocab(tok.get_vocab())
    device = student.device

    # Size the teacher to the padded slot count: variable-fleet sampling
    # means per-episode fleets differ, and the padded action mask governs
    # which slots are real.
    n_techs = int(gym_cfg.max_techs)
    teacher = TopsisAgent(n_techs)
    teacher.attach_env(env)

    # ----- collect ------------------------------------------------------
    print(f"=== BC collection: {args.episodes} episodes x "
          f"{args.sim_time:,.0f} t.u., teacher={teacher.name} ===")
    dataset: list[tuple[dict, np.ndarray, int]] = []
    action_hist: Counter = Counter()
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed * 1_000 + ep
        np.random.seed(seed)
        seed_numba_rng(seed & 0xFFFFFFFF)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = teacher.select_action(obs, deterministic=True)
            dataset.append((
                student._extract_obs(obs),
                student._extract_action_mask(obs),
                int(action),
            ))
            action_hist[int(action)] += 1
            obs, _r, term, trunc, _info = env.step(action)
            done = bool(term or trunc)
        print(f"  ep{ep:03d}  steps so far {len(dataset):>7,d}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    print(f"collected {len(dataset):,d} decisions; teacher action "
          f"histogram {dict(sorted(action_hist.items()))}")

    # ----- split + group by shape signature (layouts vary) --------------
    idx = rng.permutation(len(dataset))
    n_val = max(1, int(len(dataset) * args.holdout))
    val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:].tolist()

    def groups_of(indices):
        by_shape: dict[tuple, list[int]] = defaultdict(list)
        for i in indices:
            ob = dataset[i][0]
            key = tuple((k, ob[k].shape) for k in sorted(ob))
            by_shape[key].append(i)
        return by_shape

    train_groups = groups_of(train_idx)
    val_groups = groups_of(sorted(val_idx))

    def batch(indices):
        obs0 = dataset[indices[0]][0]
        obs_batch = {
            k: torch.from_numpy(
                np.stack([dataset[i][0][k] for i in indices])
            ).to(device)
            for k in obs0
        }
        masks = torch.from_numpy(
            np.stack([dataset[i][1] for i in indices])
        ).to(device)
        actions = torch.tensor(
            [dataset[i][2] for i in indices], dtype=torch.long, device=device
        )
        return obs_batch, masks, actions

    def masked_logits(obs_batch, masks):
        logits, _value, _hidden = student.net(obs_batch, None)
        return logits.float().masked_fill(~masks.bool(), float("-inf"))

    @torch.no_grad()
    def agreement(groups) -> float:
        hits = total = 0
        student.net.eval()
        for indices in groups.values():
            for s in range(0, len(indices), args.batch_size):
                ob, mk, ac = batch(indices[s:s + args.batch_size])
                pred = masked_logits(ob, mk).argmax(dim=-1)
                hits += int((pred == ac).sum().item())
                total += len(ac)
        student.net.train()
        return hits / max(1, total)

    # ----- train --------------------------------------------------------
    opt = torch.optim.AdamW(student.net.parameters(), lr=args.lr)
    print(f"=== BC training: {args.epochs} epochs, "
          f"{len(train_idx):,d} train / {n_val:,d} val ===")
    for epoch in range(args.epochs):
        losses = []
        order = list(train_groups.values())
        rng.shuffle(order)
        for indices in order:
            indices = list(indices)
            rng.shuffle(indices)
            for s in range(0, len(indices), args.batch_size):
                ob, mk, ac = batch(indices[s:s + args.batch_size])
                loss = F.cross_entropy(masked_logits(ob, mk), ac)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        print(f"  epoch {epoch}: loss {np.mean(losses):.4f}  "
              f"val agreement {agreement(val_groups):.1%}", flush=True)

    final = agreement(val_groups)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    student.save(out)
    print(f"=== done: val agreement {final:.1%} -> {out} ===")
    print(f"next: uv run python scripts/train_hc_improved.py "
          f"--init-checkpoint {out} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
