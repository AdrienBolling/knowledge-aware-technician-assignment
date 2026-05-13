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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from agents import (
    Agent,
    GRPOAgent,
    LeastBusyAgent,
    LeastFatiguedAgent,
    PPOTransformerAgent,
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
from experiment.reports import ReportWriter
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
    "ppo_transformer": PPOTransformerAgent,
}

_LEARNING_AGENTS = {"rainbow_dqn", "grpo", "ppo_transformer"}
_TOKEN_AGENTS = {"rainbow_dqn", "grpo", "ppo_transformer"}


def _ticket_context(ticket: Any, sim_time: float) -> dict[str, Any]:
    """Build a flat dict describing the ticket being assigned at this step.

    Returns empty-string / -1 sentinels when no ticket is in flight so
    the resulting CSV column shapes are stable.
    """
    if ticket is None:
        return {
            "ticket_machine_id": -1,
            "ticket_machine_type": "",
            "ticket_machine_name": "",
            "ticket_component_type": "",
            "ticket_created_at": -1.0,
            "ticket_wait_time": -1.0,
        }
    machine = getattr(ticket, "machine", None)
    comp_info = (
        ticket.get_failed_component_info()
        if hasattr(ticket, "get_failed_component_info")
        else None
    )
    created_at = float(getattr(ticket, "created_at", -1.0))
    return {
        "ticket_machine_id": int(getattr(machine, "machine_id", -1)) if machine is not None else -1,
        "ticket_machine_type": str(getattr(machine, "mtype", "")),
        "ticket_machine_name": str(getattr(machine, "name", "") or ""),
        "ticket_component_type": str((comp_info or {}).get("component_type", "")),
        "ticket_created_at": created_at,
        "ticket_wait_time": (
            max(0.0, float(sim_time) - created_at) if created_at >= 0 else -1.0
        ),
    }


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
        # Train env follows the configured randomization.  Eval env
        # uses a fixed factory (sampled once with ``eval_seed``) unless
        # ``randomized_scenario.randomize_eval`` is True, so the eval
        # return curve is a clean policy-quality signal independent of
        # scenario luck.
        self.tokenizer: StateTokenizer | None = None
        self.env = self._build_env(randomize=True)
        rcfg = env_config.randomized_scenario
        self.eval_env = self._build_env(randomize=rcfg.randomize_eval)

        # -- Build agent --
        self.agent = self._build_agent()

        # -- Load checkpoint if specified --
        if self.agent_cfg.checkpoint:
            self.agent.load(self.agent_cfg.checkpoint)
            logger.info("Loaded checkpoint: %s", self.agent_cfg.checkpoint)

        # -- Report writer (CSV episode + step metrics, config.json) --
        self._report_writer = self._init_report_writer()

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

    def _build_env(self, *, randomize: bool = True) -> KataEnv:
        """Create a KataEnv with the right observation mode and tokenizer.

        Parameters
        ----------
        randomize:
            When True and randomisation is enabled in the config, the
            env's ``scenario_factory`` returns a freshly sampled
            scenario on every reset.  When False (but randomisation is
            still enabled in the config), the env binds to a *single*
            scenario sampled once at construction — used for the eval
            env so the eval curve isolates policy quality from
            scenario noise.  Has no effect when randomisation is
            disabled in the config (static scenario in both cases).
        """
        cfg = self.env_cfg
        gym_cfg = cfg.gym
        rcfg = cfg.randomized_scenario

        # -- Scenario factory: static, per-episode random, or fixed-but-sampled -----
        if rcfg.enabled and randomize:
            from kata.EntityFactories import RandomScenarioSampler

            sampler = RandomScenarioSampler(cfg, rcfg, seed=rcfg.seed)
            factory = sampler
            n_techs = rcfg.n_technicians
            machine_types = sampler.all_machine_types()
            component_types = sampler.all_component_types()
        elif rcfg.enabled and not randomize:
            from kata.EntityFactories import RandomScenarioSampler

            eval_seed = (
                rcfg.eval_seed
                if rcfg.eval_seed is not None
                else (rcfg.seed + 1 if rcfg.seed is not None else 1)
            )
            # Sample ONE scenario and reuse its config on every reset.
            _sampler = RandomScenarioSampler(cfg, rcfg, seed=eval_seed)
            fixed_eval_cfg = _sampler.sample_config()
            factory = lambda c=fixed_eval_cfg: ScenarioBuilder(c).build()
            n_techs = rcfg.n_technicians
            # Vocab still needs to cover the full pool so a tokenizer
            # shared with the train env doesn't surface UNKs.
            machine_types = _sampler.all_machine_types()
            component_types = _sampler.all_component_types()
        else:
            factory = lambda: ScenarioBuilder(cfg).build()
            n_techs = len(cfg.technicians)
            machine_types = sorted({m.machine_type for m in cfg.machines.values()})
            component_types = sorted(
                {
                    c.component_type
                    for m in cfg.machines.values()
                    for c in m.components.values()
                }
            )

        # Build a tokenizer covering ALL machine + component types that
        # could ever appear, so randomised episodes don't surface UNK
        # tokens.
        if self.agent_cfg.agent_type in _TOKEN_AGENTS and self.tokenizer is None:
            self.tokenizer = StateTokenizer.build_vocab(
                machine_types=machine_types,
                n_technicians=n_techs,
                seq_length=gym_cfg.tokenizer_seq_length,
                component_types=component_types,
            )

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
    # Report writer
    # ------------------------------------------------------------------

    def reconfigure_reports(self) -> None:
        """Rebuild the report writer after a config override (used by CLI)."""
        self._report_writer = self._init_report_writer()

    def _init_report_writer(self) -> ReportWriter | None:
        rcfg = self.exp_cfg.reports
        if not rcfg.enabled:
            return None
        exp_id = rcfg.exp_id
        if not exp_id:
            ts = time.strftime("%Y%m%d-%H%M%S")
            exp_id = (
                f"{self.agent_cfg.agent_type}_seed{self.exp_cfg.seed}_{ts}"
            )
        writer = ReportWriter(rcfg.dir, exp_id)
        # Persist the snapshot immediately so the exp_id -> config link is
        # in place even if training crashes before the first flush.
        writer.save_config(
            {
                "exp_id": writer.exp_id,
                "env": self.env_cfg.model_dump(mode="json"),
                "agent": self.agent_cfg.model_dump(mode="json"),
                "experiment": self.exp_cfg.model_dump(mode="json"),
            }
        )
        logger.info("Writing reports to: %s", writer.dir)
        return writer

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
                "fleet_knowledge",
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
        title: str | None = None,
        ylabel: str | None = None,
        ylim: tuple[float, float] | None = None,
        rolling_window: int | None = None,
    ) -> None:
        """Log a within-episode time-series as a matplotlib figure image.

        Uses ``wandb.Image`` rather than the interactive
        ``wandb.plot.line`` widget — image panels are reliably
        rendered in the W&B run page (whereas custom-chart panels
        sometimes silently fail to surface).

        When ``rolling_window`` is set, the raw values are drawn at
        ``alpha=0.5`` and a red trailing-mean overlay is rendered at
        ``alpha=1`` so the trend is legible even on noisy series.
        """
        if self._wandb_run is None:
            return
        if not values:
            return
        import matplotlib.pyplot as plt
        import wandb

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        xs = range(len(values))

        if rolling_window is not None and rolling_window > 1 and len(values) >= 2:
            w = min(int(rolling_window), len(values))
            # Trailing (causal) rolling mean — no lookahead.
            csum = [0.0]
            for v in values:
                csum.append(csum[-1] + float(v))
            trend = [
                (csum[i + 1] - csum[max(0, i + 1 - w)])
                / float(min(i + 1, w))
                for i in range(len(values))
            ]
            ax.plot(xs, values, linewidth=1.0, alpha=0.5, label="raw")
            ax.plot(
                xs,
                trend,
                linewidth=1.8,
                color="red",
                alpha=1.0,
                label=f"rolling mean (w={w})",
            )
            ax.legend(loc="best", fontsize=8)
        else:
            ax.plot(xs, values, linewidth=1.2)

        ax.set_xlabel("step within episode")
        ax.set_ylabel(ylabel if ylabel is not None else key.split("/")[-1])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title or key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        wandb.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    # Per-metric display hints used to give plots sensible axis labels and
    # ranges.  Anything not listed falls back to the metric name as the
    # y-axis label and an auto-scaled y-range.
    _PLOT_HINTS: dict[str, dict[str, Any]] = {
        "repair_time_delta": {
            "ylabel": "time saved vs base (sim units)",
            "rolling_window": 25,
        },
        "repair_time_delta_per": {
            "ylabel": "% time saved vs base",
            "ylim": (0.0, 100.0),
            "rolling_window": 25,
        },
        "repair_quality": {
            "ylabel": "knowledge match",
            "ylim": (0.0, 1.0),
        },
        "tech_knowledge": {
            "ylabel": "knowledge volume",
        },
        "tech_fatigue": {
            "ylabel": "fatigue",
            "ylim": (0.0, 1.0),
        },
        "tech_specialization": {
            "ylabel": "specialization index",
            "ylim": (0.0, 1.0),
        },
    }

    def _log_assignment_histogram(
        self,
        assignment_counts: dict[str, int],
        step: int,
        phase: str,
        episode_label: str,
    ) -> None:
        """Log a per-episode bar chart of assignments per technician."""
        if not assignment_counts:
            return
        # Sort by tech name for stable ordering across episodes
        items = sorted(assignment_counts.items())
        labels = [k for k, _ in items]
        values = [float(v) for _, v in items]
        self._log_wandb_bar_plot(
            f"{phase}/metrics/assignment_histogram_episode",
            labels=labels,
            values=values,
            step=step,
            title=f"assignments per technician ({episode_label})",
            ylabel="repairs assigned",
        )

    def _log_machine_stats_histograms(
        self,
        machine_stats: dict[str, dict[str, float]],
        step: int,
        phase: str,
        episode_label: str,
    ) -> None:
        """Log three per-episode bar charts: per-machine maintenance time,
        production time, and breakdown count.
        """
        if not machine_stats:
            return
        # Stable ordering: alphabetical by machine label
        items = sorted(machine_stats.items())
        labels = [k for k, _ in items]

        for key, ylabel, title_stub in (
            ("maintenance_time", "sim time spent broken/repairing", "maintenance time per machine"),
            ("production_time", "sim time spent processing", "production time per machine"),
            ("breakdowns", "breakdowns this episode", "breakdowns per machine"),
        ):
            values = [float(v.get(key, 0.0)) for _, v in items]
            self._log_wandb_bar_plot(
                f"{phase}/metrics/machine_{key}_episode",
                labels=labels,
                values=values,
                step=step,
                title=f"{title_stub} ({episode_label})",
                ylabel=ylabel,
            )

    def _log_step_series_plots(
        self,
        step_metrics_series: dict[str, list[float]],
        step: int,
        phase: str,
        episode_label: str,
    ) -> None:
        """Render the within-episode step series as matplotlib figures.

        Series whose key contains a ``/`` (e.g. ``tech_knowledge/expert_1``)
        are grouped under their prefix and rendered as one figure with a
        legend (one line per sub-key).  Plain scalar series are rendered
        as a single-line figure each.
        """
        if not step_metrics_series:
            return

        scalar_series: dict[str, list[float]] = {}
        grouped: dict[str, dict[str, list[float]]] = {}
        for name, values in step_metrics_series.items():
            if "/" in name:
                prefix, sub = name.split("/", 1)
                grouped.setdefault(prefix, {})[sub] = values
            else:
                scalar_series[name] = values

        for name, values in scalar_series.items():
            hints = self._PLOT_HINTS.get(name, {})
            self._log_wandb_plot(
                f"{phase}/metrics/{name}_episode",
                values,
                step=step,
                title=f"{name} ({episode_label})",
                ylabel=hints.get("ylabel"),
                ylim=hints.get("ylim"),
                rolling_window=hints.get("rolling_window"),
            )

        for prefix, subs in grouped.items():
            hints = self._PLOT_HINTS.get(prefix, {})
            self._log_wandb_grouped_plot(
                f"{phase}/metrics/{prefix}_episode",
                subs,
                step=step,
                title=f"{prefix} ({episode_label})",
                ylabel=hints.get("ylabel", prefix),
                ylim=hints.get("ylim"),
            )

    def _log_wandb_bar_plot(
        self,
        key: str,
        labels: list[str],
        values: list[float],
        step: int,
        title: str | None = None,
        ylabel: str | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Log a bar chart (e.g. per-episode assignment histogram) to wandb."""
        if self._wandb_run is None:
            return
        if not labels or not values:
            return
        import matplotlib.pyplot as plt
        import wandb

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        x = list(range(len(labels)))
        bars = ax.bar(x, values, color="C0", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel or "count")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title or key)
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate each bar with its value
        for rect, v in zip(bars, values, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{int(v)}" if float(v).is_integer() else f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        wandb.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def _log_wandb_grouped_plot(
        self,
        key: str,
        series: dict[str, list[float]],
        step: int,
        title: str | None = None,
        xlabel: str = "step within episode",
        ylabel: str | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        """Log multiple series as a single matplotlib figure with a legend.

        Used to fold per-entity series (e.g. one knowledge curve per
        technician) into one comparable plot per episode.
        """
        if self._wandb_run is None:
            return
        if not series:
            return
        import matplotlib.pyplot as plt
        import wandb

        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        for label, values in series.items():
            if not values:
                continue
            ax.plot(range(len(values)), values, linewidth=1.2, label=label)
        if any(series.values()):
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or key.split("/")[-1])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title or key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        wandb.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)

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
        try:
            if mode == "train":
                return self._run_train()
            if mode == "eval":
                return self._run_eval()
            if mode == "evaluated_training":
                return self._run_evaluated_training()
            msg = f"Unknown experiment mode: {mode!r}"
            raise ValueError(msg)
        finally:
            if self._report_writer is not None:
                paths = self._report_writer.flush()
                for kind, p in paths.items():
                    if p is not None:
                        logger.info("Report %s -> %s", kind, p)

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

        eval_bar = tqdm(
            range(n),
            desc=f"Evaluating {self.agent_cfg.agent_type}",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
        )
        for i in eval_bar:
            seed = self.exp_cfg.seed + i
            ep_data = self._run_episode(
                training=False, seed=seed, phase="eval", episode_idx=i + 1
            )
            eval_bar.set_postfix({"ret": f"{ep_data['return']:+.2f}"}, refresh=False)
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

            # Per-episode within-episode time-series (e.g. repair_time_delta)
            self._log_step_series_plots(
                ep_data.get("step_metrics_series", {}),
                step=i + 1,
                phase="eval",
                episode_label=f"eval episode {i + 1}",
            )
            # Per-episode assignment histogram
            self._log_assignment_histogram(
                ep_data.get("assignment_counts", {}),
                step=i + 1,
                phase="eval",
                episode_label=f"eval episode {i + 1}",
            )
            # Per-episode per-machine histograms
            self._log_machine_stats_histograms(
                ep_data.get("machine_stats", {}),
                step=i + 1,
                phase="eval",
                episode_label=f"eval episode {i + 1}",
            )

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

        # Progress bar always shown — even with --quiet — using sys.stderr
        # so it doesn't pollute stdout pipelines.  ``leave=True`` keeps the
        # final bar after training completes.
        progress = tqdm(
            range(1, cfg.n_episodes + 1),
            desc=f"Training {self.agent_cfg.agent_type}",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
        )
        for ep in progress:
            ep_data = self._run_episode(
                training=True, seed=cfg.seed + ep, phase="train", episode_idx=ep
            )

            history["return"].append(ep_data["return"])
            history["length"].append(ep_data["length"])
            history["loss"].append(ep_data.get("loss", float("nan")))
            history["entropy"].append(ep_data.get("entropy", float("nan")))

            # Update tqdm postfix with the most informative scalars
            window = min(10, len(history["return"]))
            postfix: dict[str, str] = {
                "ret": f"{ep_data['return']:+.2f}",
                "avg10": f"{float(np.mean(history['return'][-window:])):+.2f}",
            }
            if is_learning:
                loss_v = ep_data.get("loss", float("nan"))
                if not math.isnan(loss_v):
                    postfix["loss"] = f"{loss_v:.4f}"
            progress.set_postfix(postfix, refresh=False)

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

            # Step-wise metric series logged as a per-episode line plot
            # on EVERY episode end (independent of ``wandb.log_interval``)
            # so the user can inspect the within-episode evolution of
            # ``repair_time_delta`` etc. from the W&B UI for every run.
            self._log_step_series_plots(
                ep_data.get("step_metrics_series", {}),
                step=ep,
                phase="train",
                episode_label=f"episode {ep}",
            )
            # Per-episode assignment histogram (one bar per technician).
            self._log_assignment_histogram(
                ep_data.get("assignment_counts", {}),
                step=ep,
                phase="train",
                episode_label=f"episode {ep}",
            )
            # Per-episode per-machine histograms (maintenance / production /
            # breakdowns).
            self._log_machine_stats_histograms(
                ep_data.get("machine_stats", {}),
                step=ep,
                phase="train",
                episode_label=f"episode {ep}",
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
        phase: str | None = None,
        episode_idx: int | None = None,
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
        ep_wall_start = time.perf_counter()

        done = False
        while not done:
            prev_obs = obs

            # Snapshot the ticket *before* env.step pops it — used to
            # log "what was the agent deciding about" alongside the
            # chosen action and the resulting reward.
            current_ticket = getattr(env, "current_request", None)
            sim_time_before = float(getattr(env, "_sim_time", lambda: 0.0)())
            ticket_ctx = _ticket_context(current_ticket, sim_time_before)

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

            # Persist per-step row for the report CSV
            if (
                self._report_writer is not None
                and self.exp_cfg.reports.log_steps
                and phase is not None
                and episode_idx is not None
            ):
                step_row: dict[str, Any] = {
                    "phase": phase,
                    "episode": episode_idx,
                    "step": steps,
                    "seed": seed,
                    "sim_time": float(info.get("sim_time", sim_time_before)),
                    "pending_queue_size": int(info.get("pending_queue_size", 0)),
                    "has_open_ticket": bool(info.get("has_open_ticket", False)),
                    **ticket_ctx,
                    "action": int(action),
                    "reward": float(reward),
                }
                for k, v in info.get("metrics", {}).items():
                    step_row[k] = v
                for k, v in info.get("reward_breakdown", {}).items():
                    step_row[f"reward_{k}"] = v
                self._report_writer.add_step(step_row)

            last_info = info
            done = terminated or truncated

        agent.on_episode_end(ep_return)

        # Agent update (on-policy agents update at episode end)
        update_metrics: dict[str, float] = {}
        if training and self.agent_cfg.agent_type in _LEARNING_AGENTS:
            update_metrics = agent.update()

        # Separate step-wise metric aggregates from episode-level metrics.
        # Step-wise metrics (``repair_time_delta``, ``repair_quality``,
        # ``tech_knowledge/<tech_name>`` …) are collected every step;
        # episode-level metrics (``total_breakdowns``, ``total_repairs``,
        # …) are emitted only once on the terminal step.  The metric
        # *registries* are the source of truth: any name (or
        # ``<name>/<sub>`` prefix) coming from a ``StepMetric`` is
        # treated as a step series; everything from an ``EpisodeMetric``
        # is treated as a single end-of-episode value.
        from kata.metrics import EPISODE_METRICS, STEP_METRICS

        step_metric_prefixes = tuple(m.name for m in STEP_METRICS)
        episode_metric_names = {m.name for m in EPISODE_METRICS}

        def _is_step_series(key: str) -> bool:
            return any(
                key == p or key.startswith(p + "/") for p in step_metric_prefixes
            )

        step_metrics_mean: dict[str, float] = {}
        for k, vals in step_metrics_series.items():
            if _is_step_series(k):
                step_metrics_mean[k] = float(np.mean(vals))

        # Episode metrics come from the last info dict (appended on termination)
        episode_metrics: dict[str, float] = {}
        final_metrics = last_info.get("metrics", {})
        for k in episode_metric_names:
            if k in final_metrics:
                episode_metrics[k] = final_metrics[k]

        # Per-tech assignment counts at episode end (always present in info)
        assignment_counts = dict(last_info.get("assignment_counts", {}))
        machine_stats = dict(last_info.get("machine_stats", {}))
        mean_reward_components = {
            k: v / max(steps, 1) for k, v in ep_components.items()
        }

        # Persist per-episode row for the report CSV
        if (
            self._report_writer is not None
            and phase is not None
            and episode_idx is not None
        ):
            wall_clock = float(time.perf_counter() - ep_wall_start)
            unique_techs_used = int(
                sum(1 for c in assignment_counts.values() if int(c) > 0)
            )
            ep_row: dict[str, Any] = {
                "phase": phase,
                "episode": episode_idx,
                "seed": seed,
                "return": float(ep_return),
                "length": int(steps),
                "sim_time_end": float(last_info.get("sim_time", 0.0)),
                "wall_clock_seconds": wall_clock,
                "unique_techs_used": unique_techs_used,
                "loss": float(update_metrics.get("loss", float("nan"))),
                "entropy": float(update_metrics.get("entropy", float("nan"))),
            }
            # All agent-update metrics (pg_loss, vf_loss, approx_kl,
            # clip_fraction, lr, early_stop, rollout_size, …)
            for k, v in update_metrics.items():
                if k in ("loss", "entropy"):
                    continue  # already top-level
                try:
                    ep_row[f"update/{k}"] = float(v)
                except (TypeError, ValueError):
                    ep_row[f"update/{k}"] = v
            for k, v in episode_metrics.items():
                ep_row[f"metric/{k}"] = float(v)
            for k, v in step_metrics_mean.items():
                ep_row[f"step_mean/{k}"] = float(v)
            # Sum and mean of each reward component over the episode
            for k, v in ep_components.items():
                ep_row[f"reward_sum/{k}"] = float(v)
            for k, v in mean_reward_components.items():
                ep_row[f"reward_mean/{k}"] = float(v)
            for tech_name, count in assignment_counts.items():
                ep_row[f"assignments/{tech_name}"] = int(count)
            for label, s in machine_stats.items():
                ep_row[f"machine_maintenance/{label}"] = float(s.get("maintenance_time", 0.0))
                ep_row[f"machine_production/{label}"] = float(s.get("production_time", 0.0))
                ep_row[f"machine_breakdowns/{label}"] = int(s.get("breakdowns", 0))
            self._report_writer.add_episode(ep_row)

        return {
            "return": ep_return,
            "length": steps,
            "loss": update_metrics.get("loss", float("nan")),
            "entropy": update_metrics.get("entropy", float("nan")),
            "mean_reward_components": mean_reward_components,
            "step_metrics_mean": step_metrics_mean,
            "step_metrics_series": {
                k: v for k, v in step_metrics_series.items() if _is_step_series(k)
            },
            "episode_metrics": episode_metrics,
            "assignment_counts": assignment_counts,
            "machine_stats": machine_stats,
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
                phase="inline_eval",
                episode_idx=current_ep,
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
