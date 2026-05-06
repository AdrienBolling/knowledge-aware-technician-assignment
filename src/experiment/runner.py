"""Experiment runner — trains and/or evaluates an agent on KataEnv.

Three experiment modes are supported:

- ``train``  — train an agent, optionally with inline eval checkpoints.
- ``eval``   — load a checkpoint and run evaluation episodes only.
- ``evaluated_training`` — train with interleaved full evaluation rounds,
  each logged as a separate W&B run sharing the same group.

Usage
-----
>>> from experiment import Experiment
>>> exp = Experiment.from_configs(
...     env_config="run_configs/complex_factory.json",
...     agent_config="run_configs/agents/grpo.json",
...     experiment_config="run_configs/experiments/default.json",
... )
>>> exp.run()
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from agents import (
    Agent,
    GRPOAgent,
    LeastBusyAgent,
    LeastFatiguedAgent,
    RainbowDQNAgent,
    RandomAgent,
    RoundRobinAgent,
    ShortestQueueAgent,
)
from experiment.config import (
    AgentConfig,
    ExperimentConfig,
    load_json,
)
from kata.core.config import KATAConfig
from kata.core.tokenizer import StateTokenizer
from kata.env import KataEnv
from kata.scenario import ScenarioBuilder

# ======================================================================
# Agent registry
# ======================================================================

_AGENT_REGISTRY: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "round_robin": RoundRobinAgent,
    "least_busy": LeastBusyAgent,
    "least_fatigued": LeastFatiguedAgent,
    "shortest_queue": ShortestQueueAgent,
    "rainbow_dqn": RainbowDQNAgent,
    "grpo": GRPOAgent,
}

_LEARNING_AGENTS = {"rainbow_dqn", "grpo"}
_TOKEN_AGENTS = {"rainbow_dqn", "grpo"}


# ======================================================================
# Experiment
# ======================================================================


class Experiment:
    """Encapsulates an agent + environment for training, evaluation, and logging.

    Parameters
    ----------
    env_config:
        Full KATA environment configuration.
    agent_config:
        Agent type selection and hyperparameters.
    experiment_config:
        Experiment mode, training/eval settings, checkpoint, and W&B config.
    quiet:
        When True, set experiment logger to WARNING level.
        Entity simulation logs use DEBUG level and are silent by default
        (the ``kata`` package logger uses a NullHandler).

    """

    def __init__(
        self,
        env_config: KATAConfig,
        agent_config: AgentConfig,
        experiment_config: ExperimentConfig,
        *,
        quiet: bool = False,
    ) -> None:
        self.env_cfg = env_config
        self.agent_cfg = agent_config
        self.exp_cfg = experiment_config
        self.quiet = quiet

        self._wandb_run: Any = None
        self._best_eval_return: float = -math.inf

        # -- Build environment --
        self.tokenizer: StateTokenizer | None = None
        self.env = self._build_env()
        self.eval_env = self._build_env()

        # -- Build agent --
        self.agent = self._build_agent()

        # -- Load checkpoint if specified --
        if self.agent_cfg.checkpoint:
            self.agent.load(self.agent_cfg.checkpoint)
            logger.info("Loaded checkpoint: %s", self.agent_cfg.checkpoint)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        env_config: str | Path,
        agent_config: str | Path,
        experiment_config: str | Path,
        *,
        quiet: bool = False,
    ) -> Experiment:
        """Construct an Experiment from three JSON file paths."""
        env_data = load_json(env_config)
        agent_data = load_json(agent_config)
        exp_data = load_json(experiment_config)

        # Set KATA_CONF_PATH to a dummy to avoid loading default config
        os.environ["KATA_CONF_PATH"] = "/dev/null/__no_file__"

        return cls(
            env_config=KATAConfig(**env_data),
            agent_config=AgentConfig(**agent_data),
            experiment_config=ExperimentConfig(**exp_data),
            quiet=quiet,
        )

    def _build_env(self) -> KataEnv:
        """Create a KataEnv with the right observation mode and tokenizer."""
        cfg = self.env_cfg
        gym_cfg = cfg.gym

        # Build tokenizer for token-based agents
        if self.agent_cfg.agent_type in _TOKEN_AGENTS and self.tokenizer is None:
            machine_types = sorted({m.machine_type for m in cfg.machines.values()})
            n_techs = len(cfg.technicians)
            self.tokenizer = StateTokenizer.build_vocab(
                machine_types=machine_types,
                n_technicians=n_techs,
                seq_length=gym_cfg.tokenizer_seq_length,
            )

        factory = lambda: ScenarioBuilder(cfg).build()
        return KataEnv(
            scenario_factory=factory,
            config=gym_cfg,
            tokenizer=self.tokenizer,
        )

    def _build_agent(self) -> Agent:
        """Instantiate the agent from the registry."""
        agent_type = self.agent_cfg.agent_type
        cls = _AGENT_REGISTRY[agent_type]
        params = dict(self.agent_cfg.params)

        n_actions = self.env.action_space.n
        params["n_actions"] = n_actions

        # Inject vocab_size for token-based agents
        if agent_type in _TOKEN_AGENTS and self.tokenizer is not None:
            params.setdefault("vocab_size", self.tokenizer.vocab_size)
            params.setdefault("max_seq_len", self.env_cfg.gym.tokenizer_seq_length)

        return cls(**params)

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------

    def _auto_tags(self, run_type: str) -> list[str]:
        """Build informative tags from the experiment configuration."""
        tags = list(self.exp_cfg.wandb.tags)

        tags.append(f"agent:{self.agent_cfg.agent_type}")
        tags.append(f"mode:{self.exp_cfg.mode}")
        tags.append(f"run:{run_type}")
        tags.append(f"obs:{self.env_cfg.gym.observation_representation}")
        tags.append(f"obs_mode:{self.env_cfg.gym.observation_mode}")

        if self.agent_cfg.agent_type in _LEARNING_AGENTS:
            tags.append("learning")
        else:
            tags.append("heuristic")

        if self.agent_cfg.checkpoint:
            tags.append("from_checkpoint")

        n_techs = len(self.env_cfg.technicians)
        n_machines = len(self.env_cfg.machines)
        tags.append(f"techs:{n_techs}")
        tags.append(f"machines:{n_machines}")

        # Reward components that are enabled
        reward_cfg = self.env_cfg.gym.reward
        enabled = [
            name
            for name in [
                "assignment",
                "wait_time",
                "queue_size",
                "busy_technician",
                "fatigue_cost",
                "knowledge_match",
                "workload_balance",
                "estimated_repair_time",
                "machine_criticality",
            ]
            if getattr(reward_cfg, name, None) and getattr(reward_cfg, name).enabled
        ]
        if enabled:
            tags.append(f"rewards:{len(enabled)}")

        return tags

    def _auto_group(self) -> str:
        """Generate a W&B group name for grouping related runs."""
        if self.exp_cfg.wandb.group:
            return self.exp_cfg.wandb.group
        return f"{self.agent_cfg.agent_type}_seed{self.exp_cfg.seed}"

    def _init_wandb(self, run_type: str) -> None:
        """Initialize a W&B run.

        Parameters
        ----------
        run_type:
            One of ``"train"``, ``"eval"``. Used for tagging and naming.

        """
        wcfg = self.exp_cfg.wandb
        if not wcfg.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            msg = "wandb is required for logging. Install with: pip install wandb"
            raise ImportError(msg) from exc

        config_dict = {
            "env": self.env_cfg.model_dump(mode="json"),
            "agent": self.agent_cfg.model_dump(mode="json"),
            "experiment": self.exp_cfg.model_dump(mode="json"),
            "run_type": run_type,
        }

        group = self._auto_group()
        tags = self._auto_tags(run_type)

        if wcfg.name:
            name = f"{wcfg.name}/{run_type}"
        else:
            name = f"{self.agent_cfg.agent_type}_{run_type}_s{self.exp_cfg.seed}"

        self._wandb_run = wandb.init(
            project=wcfg.project,
            entity=wcfg.entity,
            group=group,
            tags=tags,
            name=name,
            job_type=run_type,
            config=config_dict,
            reinit=True,
        )

    def _finish_wandb(self) -> None:
        """Finalize the current W&B run."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _log_wandb(self, data: dict[str, Any], step: int) -> None:
        """Log a dict to W&B if active."""
        if self._wandb_run is not None:
            import wandb

            wandb.log(data, step=step)

    def _log_wandb_plot(
        self,
        key: str,
        values: list[float],
        step: int,
    ) -> None:
        """Log a within-episode time-series as a wandb line plot."""
        if self._wandb_run is None:
            return
        import wandb

        table = wandb.Table(
            data=[[i, v] for i, v in enumerate(values)],
            columns=["step", "value"],
        )
        wandb.log(
            {key: wandb.plot.line(table, "step", "value", title=key)},
            step=step,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the experiment according to the configured mode.

        Returns
        -------
        dict
            Training history and/or evaluation results depending on mode.

        """
        mode = self.exp_cfg.mode
        if mode == "train":
            return self._run_train()
        if mode == "eval":
            return self._run_eval()
        if mode == "evaluated_training":
            return self._run_evaluated_training()
        msg = f"Unknown experiment mode: {mode!r}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Mode: train
    # ------------------------------------------------------------------

    def _run_train(self) -> dict[str, list[float]]:
        """Train the agent. Periodic eval is logged inline (same W&B run)."""
        self._init_wandb("train")
        try:
            history = self._train_loop()
        finally:
            self._finish_wandb()
        return history

    # ------------------------------------------------------------------
    # Mode: eval
    # ------------------------------------------------------------------

    def _run_eval(self) -> dict[str, Any]:
        """Evaluate the agent (no training). Requires a checkpoint for
        learning agents (heuristic agents work out of the box).
        """
        self._init_wandb("eval")
        try:
            results = self._eval_standalone()
        finally:
            self._finish_wandb()
        return results

    def _eval_standalone(self) -> dict[str, Any]:
        """Run n_episodes evaluation episodes and print/log results."""
        cfg = self.exp_cfg.eval
        n = cfg.n_episodes

        returns: list[float] = []
        all_metrics: dict[str, list[float]] = defaultdict(list)
        all_breakdown: dict[str, list[float]] = defaultdict(list)

        for i in range(n):
            seed = self.exp_cfg.seed + i
            ep_data = self._run_episode(training=False, seed=seed)
            returns.append(ep_data["return"])
            for k, v in ep_data.get("episode_metrics", {}).items():
                all_metrics[k].append(v)
            for k, v in ep_data.get("step_metrics_mean", {}).items():
                all_metrics[k].append(v)
            for k, v in ep_data.get("mean_reward_components", {}).items():
                all_breakdown[k].append(v)

            # Log each episode
            log_data: dict[str, Any] = {
                "eval/return": ep_data["return"],
                "eval/length": ep_data["length"],
                "eval/episode": i + 1,
            }
            for k, v in ep_data.get("episode_metrics", {}).items():
                log_data[f"eval/metrics/{k}"] = v
            for k, v in ep_data.get("step_metrics_mean", {}).items():
                log_data[f"eval/metrics/{k}"] = v
            for k, v in ep_data.get("mean_reward_components", {}).items():
                log_data[f"eval/reward/{k}"] = v
            self._log_wandb(log_data, step=i + 1)

            if (i + 1) % max(1, n // 5) == 0 or i == 0:
                logger.info(
                    f"  eval {i + 1:3d}/{n} | "
                    f"return={ep_data['return']:+8.2f} | "
                    f"steps={ep_data['length']:3d}"
                )

        # Summary
        result: dict[str, Any] = {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns)),
            "returns": returns,
        }
        for k, vals in all_metrics.items():
            result[f"metrics/{k}"] = float(np.mean(vals))
        for k, vals in all_breakdown.items():
            result[f"reward/{k}"] = float(np.mean(vals))

        # Log summary to wandb
        summary_data = {
            "eval/summary/return_mean": result["return_mean"],
            "eval/summary/return_std": result["return_std"],
            "eval/summary/return_min": result["return_min"],
            "eval/summary/return_max": result["return_max"],
        }
        for k, vals in all_metrics.items():
            summary_data[f"eval/summary/metrics/{k}"] = float(np.mean(vals))
        self._log_wandb(summary_data, step=n)

        logger.info(
            f"\nEvaluation complete ({n} episodes): "
            f"mean={result['return_mean']:+.2f}  "
            f"std={result['return_std']:.2f}  "
            f"min={result['return_min']:+.2f}  "
            f"max={result['return_max']:+.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Mode: evaluated_training
    # ------------------------------------------------------------------

    def _run_evaluated_training(self) -> dict[str, Any]:
        """Train with interleaved evaluation rounds.

        Training and evaluation are logged as **separate W&B runs** in the
        same group, making it easy to compare train vs eval curves in the
        W&B dashboard.
        """
        # Phase 1: training (with inline progress eval)
        self._init_wandb("train")
        try:
            history = self._train_loop()
        finally:
            self._finish_wandb()

        # Phase 2: full standalone evaluation
        self._init_wandb("eval")
        try:
            eval_results = self._eval_standalone()
        finally:
            self._finish_wandb()

        return {"training": history, "evaluation": eval_results}

    # ------------------------------------------------------------------
    # Core training loop (shared by train and evaluated_training)
    # ------------------------------------------------------------------

    def _train_loop(self) -> dict[str, list[float]]:
        """Run the training loop with optional inline eval."""
        cfg = self.exp_cfg
        is_learning = self.agent_cfg.agent_type in _LEARNING_AGENTS

        history: dict[str, list[float]] = {
            "return": [],
            "length": [],
            "loss": [],
            "entropy": [],
        }

        for ep in range(1, cfg.n_episodes + 1):
            ep_data = self._run_episode(training=True, seed=cfg.seed + ep)

            history["return"].append(ep_data["return"])
            history["length"].append(ep_data["length"])
            history["loss"].append(ep_data.get("loss", float("nan")))
            history["entropy"].append(ep_data.get("entropy", float("nan")))

            # -- W&B logging --
            if ep % cfg.wandb.log_interval == 0:
                log_data: dict[str, Any] = {
                    "train/return": ep_data["return"],
                    "train/length": ep_data["length"],
                    "train/episode": ep,
                }
                if is_learning:
                    log_data["train/loss"] = ep_data.get("loss", float("nan"))
                    log_data["train/entropy"] = ep_data.get("entropy", float("nan"))

                # Mean per-step reward components
                for k, v in ep_data.get("mean_reward_components", {}).items():
                    log_data[f"train/reward/{k}"] = v

                # Episode-level metrics (breakdowns, repairs, products)
                for k, v in ep_data.get("episode_metrics", {}).items():
                    log_data[f"train/metrics/{k}"] = v

                # Step-wise metrics: log episode mean
                for k, v in ep_data.get("step_metrics_mean", {}).items():
                    log_data[f"train/metrics/{k}"] = v

                # Rolling return average
                window = min(10, len(history["return"]))
                log_data["train/return_avg"] = float(
                    np.mean(history["return"][-window:])
                )

                self._log_wandb(log_data, step=ep)

                # Step-wise metrics: log within-episode evolution as plots
                for k, vals in ep_data.get("step_metrics_series", {}).items():
                    self._log_wandb_plot(
                        f"train/metrics/{k}_episode",
                        vals,
                        step=ep,
                    )

            # -- Stdout logging --
            if ep % cfg.log_interval == 0 or ep == 1:
                avg_r = float(np.mean(history["return"][-cfg.log_interval :]))
                parts = [
                    f"Episode {ep:4d}/{cfg.n_episodes}",
                    f"return={ep_data['return']:+8.2f}",
                    f"avg={avg_r:+8.2f}",
                    f"steps={ep_data['length']:3d}",
                ]
                if is_learning:
                    parts.append(f"loss={ep_data.get('loss', float('nan')):7.4f}")
                em = ep_data.get("episode_metrics", {})
                sm = ep_data.get("step_metrics_mean", {})
                if "repair_quality" in sm:
                    parts.append(f"quality={sm['repair_quality']:.3f}")
                if "finished_products" in em:
                    parts.append(f"products={em['finished_products']:.0f}")
                logger.info(" | ".join(parts))

            # -- Inline evaluation --
            if cfg.eval.enabled and ep % cfg.eval.interval == 0:
                eval_data = self._inline_eval(ep)
                self._log_wandb(eval_data, step=ep)
                mean_r = eval_data["eval/return_mean"]
                std_r = eval_data["eval/return_std"]
                logger.info(f"  [eval] mean={mean_r:+8.2f}  std={std_r:.2f}")

                if cfg.checkpoint.save_best and mean_r > self._best_eval_return:
                    self._best_eval_return = mean_r
                    self._save_checkpoint("best")

            # -- Periodic checkpoint --
            if cfg.checkpoint.enabled and ep % cfg.checkpoint.interval == 0:
                self._save_checkpoint(f"ep{ep:05d}")

        # Final checkpoint
        if cfg.checkpoint.enabled:
            self._save_checkpoint("final")

        logger.info(
            f"\nTraining complete. "
            f"Final avg return (last 10): {np.mean(history['return'][-10:]):+.2f}"
        )
        return history

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        *,
        training: bool,
        seed: int,
    ) -> dict[str, Any]:
        """Run a single episode, returning aggregated data.

        Returns
        -------
        dict with keys:
            return: float — episode return (sum of rewards)
            length: int — number of steps
            loss, entropy: float — agent update metrics (nan if not learning)
            mean_reward_components: dict — mean per-step value of each reward component
            step_metrics_mean: dict — mean of step-wise metrics over the episode
            step_metrics_series: dict[str, list[float]] — raw per-step values
            episode_metrics: dict — episode-level metrics (breakdowns, repairs, products)

        """
        env = self.env if training else self.eval_env
        agent = self.agent
        deterministic = not training

        obs, info = env.reset(seed=seed)

        agent.on_episode_start()

        ep_return = 0.0
        steps = 0
        ep_components: dict[str, float] = defaultdict(float)
        step_metrics_series: dict[str, list[float]] = defaultdict(list)
        last_info: dict[str, Any] = info

        done = False
        while not done:
            prev_obs = obs

            action = agent.select_action(obs, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)

            if training:
                agent.observe_transition(
                    prev_obs,
                    action,
                    reward,
                    obs,
                    terminated,
                    truncated,
                    info,
                )

            ep_return += reward
            steps += 1

            for k, v in info.get("reward_breakdown", {}).items():
                ep_components[k] += v

            # Collect step-wise metrics (repair_time_delta, repair_quality)
            for k, v in info.get("metrics", {}).items():
                step_metrics_series[k].append(v)

            last_info = info
            done = terminated or truncated

        agent.on_episode_end(ep_return)

        # Agent update (on-policy agents update at episode end)
        update_metrics: dict[str, float] = {}
        if training and self.agent_cfg.agent_type in _LEARNING_AGENTS:
            update_metrics = agent.update()

        # Separate step-wise metric aggregates from episode-level metrics.
        # Step-wise metrics (repair_time_delta, repair_quality) are collected
        # every step — we report their mean and keep the raw series.
        # Episode-level metrics (total_breakdowns, total_repairs,
        # finished_products) are emitted only on the final step by the env.
        step_metric_names = {"repair_time_delta", "repair_quality"}
        episode_metric_names = {
            "total_breakdowns",
            "total_repairs",
            "finished_products",
        }

        step_metrics_mean: dict[str, float] = {}
        for k, vals in step_metrics_series.items():
            if k in step_metric_names:
                step_metrics_mean[k] = float(np.mean(vals))

        # Episode metrics come from the last info dict (appended on termination)
        episode_metrics: dict[str, float] = {}
        final_metrics = last_info.get("metrics", {})
        for k in episode_metric_names:
            if k in final_metrics:
                episode_metrics[k] = final_metrics[k]

        return {
            "return": ep_return,
            "length": steps,
            "loss": update_metrics.get("loss", float("nan")),
            "entropy": update_metrics.get("entropy", float("nan")),
            "mean_reward_components": {
                k: v / max(steps, 1) for k, v in ep_components.items()
            },
            "step_metrics_mean": step_metrics_mean,
            "step_metrics_series": {
                k: v for k, v in step_metrics_series.items() if k in step_metric_names
            },
            "episode_metrics": episode_metrics,
            "update_metrics": update_metrics,
        }

    # ------------------------------------------------------------------
    # Inline evaluation (during training)
    # ------------------------------------------------------------------

    def _inline_eval(self, current_ep: int) -> dict[str, Any]:
        """Run a few evaluation episodes inline during training."""
        cfg = self.exp_cfg.eval
        returns: list[float] = []
        all_metrics: dict[str, list[float]] = defaultdict(list)
        all_breakdown: dict[str, list[float]] = defaultdict(list)

        for i in range(cfg.n_episodes):
            ep_data = self._run_episode(
                training=False,
                seed=10_000 + current_ep * 100 + i,
            )
            returns.append(ep_data["return"])
            for k, v in ep_data.get("episode_metrics", {}).items():
                all_metrics[k].append(v)
            for k, v in ep_data.get("step_metrics_mean", {}).items():
                all_metrics[k].append(v)
            for k, v in ep_data.get("mean_reward_components", {}).items():
                all_breakdown[k].append(v)

        result: dict[str, Any] = {
            "eval/return_mean": float(np.mean(returns)),
            "eval/return_std": float(np.std(returns)),
            "eval/return_min": float(np.min(returns)),
            "eval/return_max": float(np.max(returns)),
        }
        for k, vals in all_metrics.items():
            result[f"eval/metrics/{k}"] = float(np.mean(vals))
        for k, vals in all_breakdown.items():
            result[f"eval/reward/{k}"] = float(np.mean(vals))

        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> None:
        """Save agent state to disk and optionally to W&B."""
        ckpt_dir = Path(self.exp_cfg.checkpoint.dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{self.agent_cfg.agent_type}_{tag}.pt"

        self.agent.save(str(path))

        if self._wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=f"model-{tag}",
                type="model",
                description=f"Agent checkpoint ({tag})",
            )
            artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)
