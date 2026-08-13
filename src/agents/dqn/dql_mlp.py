"""Double DQN over a flattened ``set`` observation — traditional baseline.

This is the value-based member of the three plain-MLP baselines that
anchor the paper's architecture ablation.  It is deliberately the
*textbook* algorithm (Mnih et al., 2015 + van Hasselt et al., 2016):

* Double DQN target — the ONLINE net picks the next action, the TARGET
  net scores it;
* a UNIFORM replay buffer (``agents.networks.replay_buffer``);
* masked epsilon-greedy exploration, epsilon annealed linearly on
  environment steps;
* a hard target-network sync every ``target_update_freq`` gradient
  steps;
* a constant learning rate.

and deliberately *not* Rainbow: no dueling head, no prioritized replay,
no n-step returns, no noisy layers.  Two of those omissions are lessons
rather than taste:

``NoisyLinear`` is the D2 trap
    :class:`agents.dqn.rainbow.NoisyLinear` (rainbow.py:74-81) samples
    its perturbation whenever ``module.training`` is set, so a
    benchmark forward on a net left in train mode is stochastic even
    under ``deterministic=True``.  Plain ``nn.Linear`` heads plus an
    explicit eval-mode guard around action selection make deterministic
    acting actually deterministic.

Gradient cadence is per env step, not per episode
    :class:`agents.dqn.rainbow.RainbowDQNAgent` only learns inside
    :meth:`update`, and the serial training loop calls ``update()``
    once per episode (runner.py:1489) — a 600-episode run therefore
    gave Rainbow ~600 SGD steps in total.  Here the optimisation lives
    in :meth:`observe_transition`: one gradient step every
    ``train_freq`` env steps once the buffer holds ``min_replay_size``
    transitions, and :meth:`update` merely returns the cached metrics
    of the most recent gradient step.

Observation handling
--------------------

The env's ``set`` observation is flattened by
:class:`agents.networks.mlp_encoder.SetObsFlattener` (one-hot per
categorical slot position, symlog per wide-range scalar) and the
``max_techs`` action-mask bits are appended at a FIXED offset
(``self.mask_offset``).  The mask bits are legitimate input features —
availability is exactly what the Q-function must condition on — and
storing them inside the flat vector is what lets the replay buffer keep
its plain ``np.ndarray`` contract (replay_buffer.py:13-55) while the
Double-DQN target still masks the NEXT state with the NEXT state's own
mask.

The learning rate is constant on purpose: the cosine schedule of the
PPO agents is the defect-D1 class (a mis-sized ``total_updates`` drove
the LR to exactly 0 mid-run), and a baseline should not be able to fail
that way.

Private RNG streams
-------------------

Exploration and replay sampling draw from two generators the agent
OWNS (``self._explore_rng`` / ``self._replay_rng``), never from the
process-global ``random`` module.  That module is the simulator's
stream — ``kata.machine.Machine._breakdown_driver`` draws its failure
times from it and the runner re-seeds it per episode
(runner.py:1421-1428) — so an agent consuming it would make the WORLD
diverge as a function of its own internals: change the gradient
cadence (``train_freq``, ``min_replay_size``) and every subsequent
machine failure moves, which is neither a fair comparison against the
heuristics nor reproducible.  With private streams the seeded factory
is bit-identical whatever the learner does, epsilon decisions are
independent of the gradient cadence, and both generator states ride
in the checkpoint so a resumed run continues the same exploration
sequence.

Serial training only — this agent intentionally does not implement
``select_actions``/``env_id`` streams, so the runner's vectorised loop
will refuse it rather than silently mis-attribute transitions.
"""

from __future__ import annotations

import logging
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.base import Agent, resolve_device
from agents.networks.mlp_encoder import MLPQNetwork, SetObsFlattener
from agents.networks.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

# Canonical frozen set vocabulary (``GymEnvConfig.set_vocab_path``
# default, config.py:722).  The flattener's one-hot tables are a pure
# function of it, so the input width — and hence the network shape — is
# fixed the moment the agent is constructed, long before the runner
# attaches its tokenizer vocab.
_DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[3] / "run_configs" / "vocab" / "set_vocab.json"
)

_STORE_DTYPES = {"float16": np.float16, "float32": np.float32}


class DQLMLPAgent(Agent):
    """Double DQN with a plain MLP over the flattened set observation.

    Parameters
    ----------
    n_actions:
        Size of the (padded) discrete action space.  Must equal
        ``max_techs``: the Q head emits one value per technician slot
        and the action mask is padded to the same cap.
    vocab:
        Frozen set vocabulary used to build the flattener's one-hot
        tables — a ``{token: id}`` mapping, a path to a vocab JSON, or
        ``None`` for the canonical ``run_configs/vocab/set_vocab.json``.
    vocab_size:
        Ignored; accepted because the runner injects it for token-based
        agents.  A value inconsistent with ``vocab`` is logged as a
        warning (it signals vocabulary drift).
    max_techs / max_machines / tech_slot_length / machine_slot_length /
    env_length:
        Observation geometry — must match the env config that produces
        the observations.  The runner injects all but
        ``machine_slot_length`` (runner.py:344-359).
    sim_time_scale:
        Ignored; accepted for runner compatibility.  The flattener
        squashes sim time with a stateless ``symlog`` instead of a
        fitted scale, precisely so no normaliser state can drift
        between training and evaluation (defect D4).
    hidden_sizes:
        Widths of the Linear-LayerNorm-ReLU trunk.
    lr / gamma / batch_size / max_grad_norm:
        Optimisation and discounting.  ``gamma`` is per decision (this
        baseline stays on the classic per-decision MDP view).
    buffer_capacity / min_replay_size:
        Replay size and the warmup before learning starts
        (``min_replay_size`` is floored at ``batch_size``).  Memory is
        ``2 * obs_dim * sizeof(store_dtype) * capacity`` — with the
        default 30/100-slot geometry (``obs_dim`` 5234) that is ~1.0 GB
        at 50k transitions in float16 and ~10.5 GB at the shipped 500k
        (``run_configs/agents/dql_mlp.json``; fine on a 128 GB host).
        The shipped capacity is deliberately much larger than one
        scenario block: multiscale episodes run ~11.5k decisions and
        the scenario rotation dwells 5 episodes, so a 50k window only
        ever held the CURRENT factory — a uniform buffer that never
        mixes layouts turns the "off-policy" baseline into an
        on-distribution one.
    train_freq:
        Env steps between gradient steps (once warm).
    target_update_freq:
        GRADIENT steps between hard target-network syncs.
    epsilon_start / epsilon_end / epsilon_decay_steps:
        Linear epsilon schedule over ENV steps.
    store_dtype:
        ``"float16"`` (default) or ``"float32"`` storage precision for
        replayed observations; batches are cast back to float32.
    seed:
        Seeds torch's global RNG at construction and the agent's two
        PRIVATE streams — exploration (``seed``) and replay sampling
        (``seed + 1``, a distinct stream so the two never lock step).
        ``None`` leaves both unseeded.  Neither reads the process-global
        ``random`` module: that one belongs to the simulator (see the
        module docstring).
    device:
        ``"auto"`` / ``"cpu"`` / ``"cuda"`` …

    """

    def __init__(
        self,
        n_actions: int,                        # = max_techs (the slot cap)
        *,
        # Observation encoding
        vocab: Mapping[str, int] | str | Path | None = None,
        vocab_size: int | None = None,
        max_techs: int = 30,
        max_machines: int = 100,
        tech_slot_length: int = 16,
        machine_slot_length: int = 12,
        env_length: int = 16,
        sim_time_scale: float | None = None,
        # Network
        hidden_sizes: Sequence[int] = (512, 512),
        # Optimisation
        lr: float = 1e-4,
        gamma: float = 0.997,
        batch_size: int = 64,
        max_grad_norm: float = 10.0,
        # Replay + cadence
        buffer_capacity: int = 50_000,
        min_replay_size: int = 2_000,
        train_freq: int = 8,
        target_update_freq: int = 2_000,
        # Exploration
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 200_000,
        # Misc
        store_dtype: str = "float16",
        seed: int | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(n_actions, name="DQLMLP")

        if n_actions != max_techs:
            msg = (
                f"DQLMLPAgent expects n_actions == max_techs "
                f"(got n_actions={n_actions}, max_techs={max_techs}).  "
                "The Q head emits one value per technician slot."
            )
            raise ValueError(msg)
        if store_dtype not in _STORE_DTYPES:
            msg = (
                f"store_dtype must be one of {sorted(_STORE_DTYPES)} "
                f"(got {store_dtype!r})"
            )
            raise ValueError(msg)

        self.device = torch.device(resolve_device(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        # --- private RNG streams (see module docstring) ---------------
        # Everything stochastic the agent does reads these, so nothing
        # the learner does can perturb the simulator's global stream.
        self._explore_rng = (
            random.Random(int(seed)) if seed is not None else random.Random()
        )
        self._replay_rng = (
            random.Random(int(seed) + 1) if seed is not None else random.Random()
        )

        # --- observation encoding -------------------------------------
        # Vocabulary the one-hot tables are built from.  Any later vocab
        # (attach_vocab / checkpoint) must agree with it on every
        # ``KEY=VALUE`` id, otherwise the bins silently change meaning.
        src = _DEFAULT_VOCAB_PATH if vocab is None else vocab
        self._flattener_vocab: dict[str, int] = (
            {str(k): int(v) for k, v in src.items()}
            if isinstance(src, Mapping)
            else SetObsFlattener.load_vocab(src)
        )
        self._flattener = SetObsFlattener(
            self._flattener_vocab,
            max_techs=max_techs,
            max_machines=max_machines,
            tech_slot_len=tech_slot_length,
            machine_slot_len=machine_slot_length,
            env_len=env_length,
        ).to(self.device)
        self._vocab: dict[str, int] | None = None
        n_tokens = max(self._flattener_vocab.values()) + 1
        if vocab_size is not None and int(vocab_size) != n_tokens:
            logger.warning(
                "vocab_size=%d does not match the %d-token vocab the "
                "flattener was built from — check for vocabulary drift.",
                int(vocab_size), n_tokens,
            )

        self.mask_offset = int(self._flattener.out_dim)
        self.obs_dim = self.mask_offset + int(n_actions)
        self._store_dtype = str(store_dtype)
        self._store_np = _STORE_DTYPES[self._store_dtype]

        # --- networks -------------------------------------------------
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.online_net = MLPQNetwork(
            self.obs_dim, n_actions, self.hidden_sizes
        ).to(self.device)
        self.target_net = MLPQNetwork(
            self.obs_dim, n_actions, self.hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # never trains; frozen copy of the online net

        # Constant LR — no scheduler on purpose (see module docstring).
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=float(lr))
        self.lr = float(lr)

        # --- knobs ----------------------------------------------------
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        # A warmup shorter than one minibatch could never be sampled —
        # raise it rather than let the cadence silently skip steps.
        self.min_replay_size = max(int(min_replay_size), int(batch_size))
        self.train_freq = max(1, int(train_freq))
        self.target_update_freq = max(1, int(target_update_freq))
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.max_techs = int(max_techs)
        self.max_machines = int(max_machines)
        self.tech_slot_length = int(tech_slot_length)
        self.machine_slot_length = int(machine_slot_length)
        self.env_length = int(env_length)

        self.replay = ReplayBuffer(
            capacity=int(buffer_capacity), rng=self._replay_rng
        )
        self.buffer_capacity = int(buffer_capacity)

        # --- counters -------------------------------------------------
        self._env_steps = 0       # transitions observed (drives epsilon)
        self._train_ticks = 0     # env steps taken while the buffer is warm
        self._grad_steps = 0      # SGD steps taken
        self._target_syncs = 0    # hard target copies performed
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def net(self) -> nn.Module:
        """Alias for :attr:`online_net`.

        The benchmark harness resolves an agent's module as
        ``getattr(agent, "net", None) or getattr(agent, "online_net")``
        (eval_human_vs_performance.py:359-363) and the set-agent branch
        calls ``agent.net.eval()`` outright — expose both spellings so
        either path finds the same module.
        """
        return self.online_net

    @property
    def epsilon(self) -> float:
        """Current exploration rate — linear in ENV steps."""
        if self._env_steps >= self.epsilon_decay_steps:
            # Explicit endpoint: the interpolation would land a float
            # epsilon short of epsilon_end for the rest of training.
            return self.epsilon_end
        frac = self._env_steps / self.epsilon_decay_steps
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def gradient_steps(self) -> int:
        return self._grad_steps

    @property
    def target_syncs(self) -> int:
        return self._target_syncs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.online_net.parameters())

    @contextmanager
    def _eval_mode(self):
        """Run a forward with the online net in eval mode (defect D2).

        The trunk carries no dropout today, but acting must never depend
        on ``module.training`` — that is exactly how Rainbow's noisy
        heads made "deterministic" benchmark forwards stochastic.
        """
        was_training = self.online_net.training
        if was_training:
            self.online_net.eval()
        try:
            yield
        finally:
            if was_training:
                self.online_net.train()

    # ------------------------------------------------------------------
    # Observation → flat vector (with the action mask embedded)
    # ------------------------------------------------------------------
    def _flat_vector(self, obs: Mapping[str, Any], mask: np.ndarray) -> np.ndarray:
        """Flatten ``obs`` and append its ``n_actions`` mask bits.

        The mask lives at a FIXED offset (``self.mask_offset``) so the
        target computation can slice it back out of a replayed batch.
        """
        feats = self._flattener(obs).detach().cpu().numpy()
        flat = np.empty(self.obs_dim, dtype=np.float32)
        flat[: self.mask_offset] = feats
        flat[self.mask_offset :] = mask.astype(np.float32)
        return flat

    def _next_mask(self, next_obs: Mapping[str, Any], done: bool) -> np.ndarray:
        """Action mask of the successor state.

        A terminal successor may legitimately carry no valid action (or
        no mask at all); its bootstrap is zeroed by ``done`` anyway, so
        an all-zero mask is accepted there.  Mid-episode the flattener's
        strictness stands: a missing/empty mask is a bug, not a reason
        to fall back to all-ones.
        """
        if not done:
            return self._flattener.extract_action_mask(next_obs)
        try:
            return self._flattener.extract_action_mask(next_obs)
        except (KeyError, ValueError):
            return np.zeros(self.n_actions, dtype=bool)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def select_action(self, obs: dict[str, Any], *, deterministic: bool = False) -> int:
        """Masked epsilon-greedy (greedy when ``deterministic``).

        Invalid technician slots are excluded from BOTH branches: the
        exploration draw samples the valid set, and the greedy branch
        argmaxes over ``-inf``-masked Q-values.

        Both draws come from the agent's private ``_explore_rng``, not
        from the global ``random`` module the simulator draws its
        failures from (see the module docstring).
        """
        mask = self._flattener.extract_action_mask(obs)
        if not deterministic and self._explore_rng.random() < self.epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[self._explore_rng.randrange(valid.size)])

        flat = self._flat_vector(obs, mask)
        x = torch.from_numpy(flat).unsqueeze(0).to(self.device)
        with self._eval_mode(), torch.no_grad():
            q = self.online_net(x)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return int(q.masked_fill(~mask_t, float("-inf")).argmax(dim=-1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def observe_transition(
        self,
        obs: dict[str, Any],
        action: int,
        reward: float,
        next_obs: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Store the transition and, on cadence, take one gradient step.

        Learning is driven from here rather than from :meth:`update`
        because the serial training loop calls ``update()`` only once
        per episode — the cadence bug that starved Rainbow of gradient
        steps.  A step is taken every ``train_freq`` env steps that
        follow the buffer reaching ``min_replay_size``, i.e. after ``N``
        calls the agent has taken
        ``floor(max(0, N - min_replay_size) / train_freq)`` gradient
        steps.
        """
        _ = info
        done = bool(terminated or truncated)
        mask = self._flattener.extract_action_mask(obs)
        next_mask = self._next_mask(next_obs, done)

        # Warm BEFORE this push: learning starts on the step after the
        # buffer first holds ``min_replay_size`` transitions.
        warm = len(self.replay) >= self.min_replay_size
        self.replay.push(
            self._flat_vector(obs, mask).astype(self._store_np, copy=False),
            int(action),
            float(reward),
            self._flat_vector(next_obs, next_mask).astype(self._store_np, copy=False),
            done,
        )
        self._env_steps += 1

        if not warm:
            return
        self._train_ticks += 1
        if self._train_ticks % self.train_freq == 0:
            self._last_metrics = self._gradient_step()

    def _double_dqn_target(
        self,
        next_obs: torch.Tensor,
        next_mask: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Double-DQN bootstrap ``r + gamma * Q_target(s', argmax_a' Q_online(s', a'))``.

        The NEXT state's own action mask constrains BOTH the online
        argmax and, through it, the target gather — an unmasked argmax
        would happily bootstrap off a technician the successor state
        cannot assign.  Rows whose mask is empty (terminal successors)
        are un-masked before the argmax to avoid an all-``-inf`` row;
        their contribution is zeroed by ``done`` regardless.
        """
        with torch.no_grad():
            valid = next_mask.bool()
            valid = valid | (~valid.any(dim=1, keepdim=True))
            next_q_online = self.online_net(next_obs).masked_fill(
                ~valid, float("-inf")
            )
            best = next_q_online.argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, best).squeeze(1)
            return rewards + self.gamma * next_q * (1.0 - dones)

    def _gradient_step(self) -> dict[str, float]:
        """One Double-DQN SGD step on a uniform minibatch."""
        batch = self.replay.sample(self.batch_size)
        obs = batch["obs"].to(self.device, dtype=torch.float32)
        next_obs = batch["next_obs"].to(self.device, dtype=torch.float32)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch["done"].to(self.device)
        # The mask bits ride along inside the stored vector.
        next_mask = next_obs[:, self.mask_offset :] > 0.5

        q = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = self._double_dqn_target(next_obs, next_mask, rewards, dones)
        loss = F.smooth_l1_loss(q, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self._target_syncs += 1

        return {
            "loss": float(loss.item()),
            "mean_q": float(q.mean().item()),
            "mean_target": float(target.mean().item()),
            "epsilon": float(self.epsilon),
            "grad_steps": float(self._grad_steps),
            "target_syncs": float(self._target_syncs),
            "buffer_size": float(len(self.replay)),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def update(self, **kwargs: Any) -> dict[str, float]:
        """Return the cached metrics of the most recent gradient step.

        Off-policy learning already happened inside
        :meth:`observe_transition`; this hook exists so the runner's
        per-episode ``agent.update()`` call has something to log.
        """
        _ = kwargs
        return dict(self._last_metrics)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    @staticmethod
    def _categorical_ids(vocab: Mapping[str, int]) -> dict[str, int]:
        """The ``KEY=VALUE`` ids that define the one-hot bin layout."""
        return {str(k): int(v) for k, v in vocab.items() if "=" in str(k)}

    def _assert_vocab_compatible(self, vocab: Mapping[str, int], source: str) -> None:
        """Refuse a vocabulary that would re-index the one-hot bins.

        Vocab drift RE-INDEXES shared keys rather than appending, so a
        differing mapping does not merely add categories — it permutes
        the meaning of every bin the trunk has learned.  Fail loudly
        instead of silently scrambling features.
        """
        mine = self._categorical_ids(self._flattener_vocab)
        theirs = self._categorical_ids(vocab)
        if mine == theirs:
            return
        diff = sorted(
            k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k)
        )[:5]
        msg = (
            f"{source} vocabulary disagrees with the one the flattener was "
            f"built from (e.g. {diff}).  The one-hot bins are keyed by token "
            f"id, so loading it would change what every categorical feature "
            f"means.  Rebuild the agent with this vocabulary instead."
        )
        raise RuntimeError(msg)

    def attach_vocab(self, vocab: dict[str, int]) -> None:
        """Attach the tokenizer vocabulary so it travels with the weights.

        Called by the experiment runner right after construction.  The
        mapping must match the one the flattener was built from.
        """
        self._assert_vocab_compatible(vocab, "attached")
        self._vocab = dict(vocab)

    @staticmethod
    def peek_vocab(path: str | Path) -> dict[str, int] | None:
        """Return the vocabulary stored alongside a checkpoint, if any."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        v = ckpt.get("vocab") if isinstance(ckpt, dict) else None
        return dict(v) if isinstance(v, dict) else None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist nets, optimiser, counters, RNG states and the vocabulary.

        The two private generator states travel with the weights: the
        agent no longer inherits its randomness from the runner's
        per-episode global seeding, so a resume that did not restore
        them would silently re-run the same exploration prefix.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict[str, Any] = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # Schedule state: epsilon is a pure function of env_steps,
            # the cadence of everything else of the tick counters.
            "env_steps": self._env_steps,
            "train_ticks": self._train_ticks,
            "grad_steps": self._grad_steps,
            "target_syncs": self._target_syncs,
            "epsilon": float(self.epsilon),
            # Private RNG streams (exploration / replay sampling).
            "explore_rng_state": self._explore_rng.getstate(),
            "replay_rng_state": self._replay_rng.getstate(),
            "obs_dim": self.obs_dim,
            "mask_offset": self.mask_offset,
            "n_actions": self.n_actions,
            # Everything an eval-time loader needs to rebuild an
            # identically-shaped agent from the checkpoint alone.
            "improvements": {
                "algo": "double_dqn",
                "hidden_sizes": list(self.hidden_sizes),
                "dueling": False,
                "prioritized_replay": False,
                "noisy_nets": False,
                "n_step": 1,
                "gamma": self.gamma,
                "train_freq": self.train_freq,
                "target_update_freq": self.target_update_freq,
                "max_techs": self.max_techs,
                "max_machines": self.max_machines,
                "tech_slot_length": self.tech_slot_length,
                "machine_slot_length": self.machine_slot_length,
                "env_length": self.env_length,
                "store_dtype": self._store_dtype,
            },
        }
        if self._vocab is not None:
            ckpt["vocab"] = dict(self._vocab)
        torch.save(ckpt, path)

    def load(self, path: str | Path) -> None:
        """Restore nets, optimiser, counters, RNG states and vocabulary."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(
            ckpt.get("target_net", ckpt["online_net"])
        )
        self.target_net.eval()
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError):
                logger.warning("Could not restore optimizer state from %s", path)
        self._env_steps = int(ckpt.get("env_steps", 0))
        self._train_ticks = int(ckpt.get("train_ticks", 0))
        self._grad_steps = int(ckpt.get("grad_steps", 0))
        self._target_syncs = int(ckpt.get("target_syncs", 0))
        # Pre-RNG checkpoints simply keep the generators the constructor
        # seeded — the states are additive, never required.
        for state, rng in (
            (ckpt.get("explore_rng_state"), self._explore_rng),
            (ckpt.get("replay_rng_state"), self._replay_rng),
        ):
            if state is not None:
                try:
                    rng.setstate(tuple(state))
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Could not restore RNG state from %s (%s)", path, exc
                    )
        if "vocab" in ckpt:
            self._assert_vocab_compatible(ckpt["vocab"], "checkpoint")
            self._vocab = dict(ckpt["vocab"])


__all__ = ["DQLMLPAgent"]
