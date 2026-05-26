"""Rendering utilities for KataEnv.

Three rendering modes are supported:

- ``"cli"`` — compact, coloured text table printed to stdout.
- ``"dict"`` — structured Python dict of the full environment state.
- ``"visual"`` — interactive Plotly figure (returned as ``plotly.graph_objects.Figure``).

All renderers accept a ``KataEnv`` instance and read its public/internal
state without mutating it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_len(store: Any) -> int:
    """Return item count for a SimPy Store or Buffer, 0 on failure."""
    if store is None:
        return 0
    if hasattr(store, "__len__"):
        return len(store)
    items = getattr(store, "items", None)
    if items is not None:
        return len(items)
    return 0


def _collect_state(env: Any) -> dict[str, Any]:
    """Gather the full observable state from a KataEnv into a plain dict."""
    sim_time = env._sim_time()
    step = env.episode_step

    # -- Current request --
    req = env.current_request
    request_info: dict[str, Any] | None = None
    if req is not None:
        machine = getattr(req, "machine", None)
        request_info = {
            "created_at": getattr(req, "created_at", None),
            "waiting_time": sim_time - float(getattr(req, "created_at", sim_time)),
            "base_repair_time": float(req.get_repair_time())
            if hasattr(req, "get_repair_time")
            else None,
            "machine_id": getattr(machine, "machine_id", None),
            "machine_type": getattr(machine, "mtype", None),
            "component": req.get_failed_component_info()
            if hasattr(req, "get_failed_component_info")
            else None,
        }

    # -- Technicians --
    techs_state = []
    for t in env.dispatcher.techs if env.dispatcher else []:
        techs_state.append(
            {
                "id": t.id,
                "name": getattr(t, "name", f"tech_{t.id}"),
                "busy": bool(getattr(t, "busy", False)),
                "fatigue": round(float(getattr(t, "fatigue", 0.0)), 4),
                "in_disruption": bool(getattr(t, "_in_disruption", False)),
                "fatigue_multiplier": round(float(t.get_fatigue_multiplier()), 4)
                if hasattr(t, "get_fatigue_multiplier")
                else None,
                "knowledge_multiplier": (
                    round(float(t.get_knowledge_multiplier(req)), 4)
                    if req is not None and hasattr(t, "get_knowledge_multiplier")
                    else None
                ),
            }
        )

    # -- Machines --
    machines_state = []
    for m in env._factory_machines():
        in_buf = getattr(m, "input_buffer", None)
        out_buf = getattr(m, "output_buffer", None)
        machines_state.append(
            {
                "id": getattr(m, "machine_id", None),
                "type": getattr(m, "mtype", None),
                "broken": bool(getattr(m, "broken", False)),
                "processing": bool(getattr(m, "is_processing", False)),
                "total_processed": int(getattr(m, "total_processed", 0)),
                "input_buffer_level": _safe_len(in_buf),
                "input_buffer_capacity": int(getattr(in_buf, "capacity", 0)),
                "output_buffer_level": _safe_len(out_buf),
            }
        )

    # -- Sinks --
    sinks = getattr(env.dispatcher, "sinks", []) if env.dispatcher else []
    finished = sum(int(getattr(s, "completed", 0)) for s in sinks)

    # -- Counters --
    return {
        "sim_time": round(sim_time, 2),
        "episode_step": step,
        "queue_size": env._queue_size(),
        "finished_products": finished,
        "breakdowns": getattr(env, "_breakdown_counter", 0),
        "repairs": getattr(env, "_repair_counter", 0),
        "current_request": request_info,
        "technicians": techs_state,
        "machines": machines_state,
        "last_reward_breakdown": dict(getattr(env, "_last_reward_breakdown", {})),
        "last_metrics": dict(getattr(env, "_last_step_metrics", {})),
    }


# ======================================================================
# CLI renderer
# ======================================================================

# ANSI colour helpers (degrade gracefully if terminal doesn't support them)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"


def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Render a horizontal bar for a 0-1 value."""
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def render_cli(env: Any) -> None:
    """Print a compact, coloured summary of the environment state to stdout."""
    state = _collect_state(env)

    lines: list[str] = []
    lines.append(f"{_BOLD}{'─' * 72}{_RESET}")
    lines.append(
        f"{_BOLD}  KataEnv{_RESET}  "
        f"t={_CYAN}{state['sim_time']}{_RESET}  "
        f"step={_CYAN}{state['episode_step']}{_RESET}  "
        f"queue={_YELLOW}{state['queue_size']}{_RESET}  "
        f"products={_GREEN}{state['finished_products']}{_RESET}  "
        f"breakdowns={_RED}{state['breakdowns']}{_RESET}  "
        f"repairs={_BLUE}{state['repairs']}{_RESET}"
    )

    # -- Current ticket --
    req = state["current_request"]
    if req is not None:
        comp_str = ""
        if req.get("component"):
            c = req["component"]
            comp_str = f"  component={c['component_type']}({c['component_id']})"
        lines.append(
            f"  {_BOLD}Ticket{_RESET}: machine={_MAGENTA}{req['machine_type']}"
            f"#{req['machine_id']}{_RESET}  "
            f"wait={_YELLOW}{req['waiting_time']:.1f}{_RESET}  "
            f"base_repair={req['base_repair_time']:.1f}{comp_str}"
        )
    else:
        lines.append(f"  {_DIM}No open ticket{_RESET}")

    # -- Technicians --
    lines.append(f"  {_BOLD}Technicians{_RESET}")
    for t in state["technicians"]:
        status = f"{_RED}BUSY{_RESET}" if t["busy"] else f"{_GREEN}IDLE{_RESET}"
        fatigue_pct = t["fatigue"] * 100
        fatigue_bar = _bar(t["fatigue"], width=10)
        fatigue_color = (
            _RED if t["fatigue"] > 0.7 else (_YELLOW if t["fatigue"] > 0.3 else _GREEN)
        )

        km_str = ""
        if t["knowledge_multiplier"] is not None:
            km = t["knowledge_multiplier"]
            # Lower multiplier = more knowledgeable = green
            km_color = _GREEN if km < 0.4 else (_YELLOW if km < 0.7 else _RED)
            km_str = f"  km={km_color}{km:.2f}{_RESET}"

        lines.append(
            f"    [{status}] {t['name']:12s}  "
            f"fatigue={fatigue_color}{fatigue_bar} {fatigue_pct:5.1f}%{_RESET}"
            f"{km_str}"
        )

    # -- Machines --
    lines.append(f"  {_BOLD}Machines{_RESET}")
    for m in state["machines"]:
        if m["broken"]:
            mstatus = f"{_RED}BROKEN{_RESET}"
        elif m["processing"]:
            mstatus = f"{_BLUE}WORKING{_RESET}"
        else:
            mstatus = f"{_DIM}IDLE{_RESET}"
        buf_str = f"in={m['input_buffer_level']}/{m['input_buffer_capacity']} out={m['output_buffer_level']}"
        lines.append(
            f"    [{mstatus:>17s}] {m['type']:10s}#{m['id']:<5d}  "
            f"processed={m['total_processed']:4d}  {buf_str}"
        )

    lines.append(f"{_BOLD}{'─' * 72}{_RESET}")
    logger.info("\n".join(lines))


# ======================================================================
# Dict renderer
# ======================================================================


def render_dict(env: Any) -> dict[str, Any]:
    """Return a structured dict snapshot of the full environment state."""
    return _collect_state(env)


# ======================================================================
# Visual (Plotly) renderer
# ======================================================================


def render_visual(env: Any) -> Any:
    """Return a Plotly Figure with a dashboard-style view of the environment.

    The figure uses ``plotly.graph_objects`` and ``plotly.subplots``.
    Plotly is imported lazily so the rest of the library works without it.

    Returns
    -------
    plotly.graph_objects.Figure

    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        msg = (
            "The 'visual' render mode requires plotly. "
            "Install it with:  pip install plotly"
        )
        raise ImportError(msg) from exc

    state = _collect_state(env)
    techs = state["technicians"]
    machines = state["machines"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Technician Fatigue",
            "Machine Status & Throughput",
            "Buffer Levels",
            "Episode Summary",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # -- 1) Technician fatigue bars --
    tech_names = [t["name"] for t in techs]
    fatigues = [t["fatigue"] for t in techs]
    bar_colors = ["#e74c3c" if t["busy"] else "#2ecc71" for t in techs]
    fig.add_trace(
        go.Bar(
            x=tech_names,
            y=fatigues,
            marker_color=bar_colors,
            text=[f"{'BUSY' if t['busy'] else 'IDLE'}" for t in techs],
            textposition="outside",
            name="Fatigue",
            hovertemplate="%{x}<br>Fatigue: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.1], title_text="Fatigue", row=1, col=1)

    # -- 2) Machine throughput bars coloured by status --
    machine_labels = [f"{m['type']}#{m['id']}" for m in machines]
    processed = [m["total_processed"] for m in machines]
    m_colors = []
    for m in machines:
        if m["broken"]:
            m_colors.append("#e74c3c")
        elif m["processing"]:
            m_colors.append("#3498db")
        else:
            m_colors.append("#95a5a6")

    fig.add_trace(
        go.Bar(
            x=machine_labels,
            y=processed,
            marker_color=m_colors,
            text=[
                "BROKEN" if m["broken"] else ("WORKING" if m["processing"] else "IDLE")
                for m in machines
            ],
            textposition="outside",
            name="Processed",
            hovertemplate="%{x}<br>Processed: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Total Processed", row=1, col=2)

    # -- 3) Buffer levels (grouped bar: input vs output) --
    fig.add_trace(
        go.Bar(
            x=machine_labels,
            y=[m["input_buffer_level"] for m in machines],
            name="Input Buffer",
            marker_color="#f39c12",
            hovertemplate="%{x}<br>Input: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=machine_labels,
            y=[m["output_buffer_level"] for m in machines],
            name="Output Buffer",
            marker_color="#9b59b6",
            hovertemplate="%{x}<br>Output: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Items", row=2, col=1)

    # -- 4) Summary table --
    req = state["current_request"]
    ticket_str = "None"
    if req is not None:
        comp = req.get("component")
        comp_str = f" ({comp['component_type']})" if comp else ""
        ticket_str = f"{req['machine_type']}#{req['machine_id']}{comp_str} wait={req['waiting_time']:.1f}"

    summary_keys = [
        "Sim Time",
        "Step",
        "Queue",
        "Breakdowns",
        "Repairs",
        "Products",
        "Current Ticket",
    ]
    summary_vals = [
        str(state["sim_time"]),
        str(state["episode_step"]),
        str(state["queue_size"]),
        str(state["breakdowns"]),
        str(state["repairs"]),
        str(state["finished_products"]),
        ticket_str,
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Field", "Value"],
                fill_color="#2c3e50",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[summary_keys, summary_vals],
                fill_color=[["#ecf0f1"] * len(summary_keys)],
                align="left",
                font=dict(size=11),
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text=(
            f"<b>KataEnv Dashboard</b>  —  "
            f"t={state['sim_time']}  step={state['episode_step']}"
        ),
        title_x=0.5,
        height=600,
        width=1100,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5
        ),
        template="plotly_white",
        barmode="group",
    )

    return fig
