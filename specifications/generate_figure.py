"""Generate the KATA environment architecture diagram."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Colour palette ──────────────────────────────────────────────────────────
C_AGENT = "#2c3e50"  # dark blue-grey
C_GYM = "#2980b9"  # blue
C_SIM = "#27ae60"  # green
C_TECH = "#e67e22"  # orange
C_MACHINE = "#8e44ad"  # purple
C_PROD = "#16a085"  # teal
C_CONFIG = "#c0392b"  # red
C_ENCODE = "#f39c12"  # gold
C_LIGHT = "#ecf0f1"  # light grey
C_ARROW = "#7f8c8d"  # grey


def box(x, y, w, h, label, color, fontsize=9, sublabel=None, alpha=0.15):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2,
    )
    ax.add_patch(rect)
    # Border
    rect2 = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor="none",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect2)
    ax.text(
        x + w / 2,
        y + h - 0.3,
        label,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h - 0.65,
            sublabel,
            ha="center",
            va="top",
            fontsize=7,
            color="#555",
            style="italic",
        )


def smallbox(x, y, w, h, label, color, fontsize=7.5):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor="white",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
    )


def arrow(x1, y1, x2, y2, label=None, color=C_ARROW, style="-|>", lw=1.5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.15,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=color,
            style="italic",
        )


def curved_arrow(x1, y1, x2, y2, label=None, color=C_ARROW, rad=0.3):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.5,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + 0.2,
            my,
            label,
            ha="left",
            va="center",
            fontsize=6.5,
            color=color,
            style="italic",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════
ax.text(
    9,
    12.7,
    "KATA Environment Architecture",
    ha="center",
    va="top",
    fontsize=16,
    fontweight="bold",
    color=C_AGENT,
)
ax.text(
    9,
    12.2,
    "Knowledge-Aware Technician Assignment",
    ha="center",
    va="top",
    fontsize=10,
    color="#7f8c8d",
)

# ═══════════════════════════════════════════════════════════════════════════
# RL Agent (top)
# ═══════════════════════════════════════════════════════════════════════════
box(6.5, 11.0, 5, 1.0, "RL Agent (Transformer)", C_AGENT, fontsize=11)
smallbox(7.0, 11.1, 1.8, 0.5, "Policy", C_AGENT)
smallbox(9.2, 11.1, 1.8, 0.5, "Embedding", C_AGENT)

# ═══════════════════════════════════════════════════════════════════════════
# KataEnv (Gymnasium wrapper)
# ═══════════════════════════════════════════════════════════════════════════
box(1.0, 7.8, 16, 3.0, "KataEnv  (Gymnasium Environment)", C_GYM, fontsize=12)

# Observation pipeline
smallbox(1.5, 8.8, 2.5, 0.7, "Observation\nBuilder", C_GYM)
smallbox(4.3, 8.8, 2.2, 0.7, "Value\nBucketing", C_GYM)
smallbox(6.8, 8.8, 2.2, 0.7, "State\nTokenizer", C_GYM)

# Reward
smallbox(9.5, 8.8, 2.5, 0.7, "Reward\nComposer", C_GYM)

# Warmup / MCA
smallbox(12.5, 8.8, 2.0, 0.7, "MCA\nWarmup", C_ENCODE)
smallbox(14.8, 8.8, 1.8, 0.7, "MCA\nEncoder", C_ENCODE)

# Obs mode labels
ax.text(
    3.4,
    8.55,
    "structured | tokens | token_ids",
    ha="center",
    fontsize=6,
    color=C_GYM,
    style="italic",
)

# Token format example
ax.text(
    5.4,
    8.1,
    'key: "SIM_TIME"  val: "T_500_1K"',
    ha="center",
    fontsize=6,
    color="#555",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.2", facecolor=C_LIGHT, edgecolor="#bbb", lw=0.5),
)

# Reward components
ax.text(
    10.75,
    8.1,
    "assignment + wait_time\n+ queue_size + busy_tech",
    ha="center",
    fontsize=6,
    color="#555",
    bbox=dict(boxstyle="round,pad=0.2", facecolor=C_LIGHT, edgecolor="#bbb", lw=0.5),
)

# ═══════════════════════════════════════════════════════════════════════════
# Arrows: Agent <-> Env
# ═══════════════════════════════════════════════════════════════════════════
# obs arrow (left side)
ax.annotate(
    "",
    xy=(4.5, 10.8),
    xytext=(7.5, 11.0),
    arrowprops=dict(arrowstyle="-|>", color=C_GYM, lw=2),
)
ax.text(
    5.2,
    10.7,
    "obs (token_ids)",
    fontsize=7,
    color=C_GYM,
    fontweight="bold",
    ha="center",
)

# action arrow (center)
ax.annotate(
    "",
    xy=(9.0, 10.8),
    xytext=(9.0, 11.0),
    arrowprops=dict(arrowstyle="-|>", color=C_AGENT, lw=2),
)
ax.text(9.6, 10.85, "action", fontsize=7, color=C_AGENT, fontweight="bold", ha="center")

# reward arrow (right side)
ax.annotate(
    "",
    xy=(13.5, 10.8),
    xytext=(10.5, 11.0),
    arrowprops=dict(arrowstyle="-|>", color=C_GYM, lw=2),
)
ax.text(12.9, 10.7, "reward", fontsize=7, color=C_GYM, fontweight="bold", ha="center")

# Obs pipeline flow
arrow(4.0, 9.15, 4.3, 9.15, color=C_GYM, lw=1)
arrow(6.5, 9.15, 6.8, 9.15, color=C_GYM, lw=1)

# ═══════════════════════════════════════════════════════════════════════════
# SimPy Simulation layer
# ═══════════════════════════════════════════════════════════════════════════
box(1.0, 1.0, 16, 6.5, "SimPy Discrete-Event Simulation", C_SIM, fontsize=12)

# ── Dispatcher ──
box(
    5.5,
    5.5,
    5.5,
    1.8,
    "GymTechDispatcher",
    C_TECH,
    fontsize=10,
    sublabel="repair queue + job orchestration",
)
smallbox(5.8, 5.7, 2.0, 0.6, "Repair\nQueue", C_TECH)
smallbox(8.2, 5.7, 2.5, 0.6, "PreemptiveResource\n(per tech)", C_TECH)

# ── Technicians ──
box(11.5, 5.5, 5.0, 1.8, "GymTechnician (x N)", C_TECH, fontsize=10)
smallbox(11.8, 5.7, 1.5, 0.55, "Fatigue", C_TECH)
smallbox(13.5, 5.7, 1.7, 0.55, "Knowledge\nGrid", C_TECH)
smallbox(15.4, 5.7, 0.8, 0.55, "Busy", C_TECH)

# ── Machines ──
box(1.5, 3.0, 6.5, 2.2, "Machines", C_MACHINE, fontsize=10)
smallbox(1.8, 3.2, 2.8, 0.8, "Machine\n(simple)", C_MACHINE)
smallbox(4.9, 3.2, 2.8, 0.8, "ComplexMachine\n(components)", C_MACHINE)
ax.text(
    4.15,
    4.7,
    "Breakdown: Simple | Weibull",
    ha="center",
    fontsize=6.5,
    color=C_MACHINE,
    style="italic",
)

# ── Production network ──
box(8.5, 3.0, 8.0, 2.2, "Production Network", C_PROD, fontsize=10)
smallbox(8.8, 3.2, 1.5, 0.7, "Source", C_PROD)
smallbox(10.5, 3.2, 1.5, 0.7, "Router", C_PROD)
smallbox(12.2, 3.2, 1.6, 0.7, "Feeder", C_PROD)
smallbox(14.0, 3.2, 1.2, 0.7, "Buffer", C_PROD)
smallbox(15.4, 3.2, 0.8, 0.7, "Sink", C_PROD)

# Production flow arrows
arrow(10.3, 3.55, 10.5, 3.55, color=C_PROD)
arrow(12.0, 3.55, 12.2, 3.55, color=C_PROD)
arrow(13.8, 3.55, 14.0, 3.55, color=C_PROD)
arrow(15.2, 3.55, 15.4, 3.55, color=C_PROD)

# Product route label
ax.text(
    12.5,
    4.7,
    "Product route: CNC -> Assembly -> Sink",
    ha="center",
    fontsize=6.5,
    color=C_PROD,
    style="italic",
)

# ── Repair Request ──
smallbox(3.0, 1.3, 2.5, 0.7, "RepairRequest", C_MACHINE, fontsize=7)
ax.text(
    4.25,
    1.2,
    "machine_type, component_type\ncomponent_id, repair_time",
    ha="center",
    fontsize=5.5,
    color="#666",
)

# ── Encoder ──
smallbox(6.5, 1.3, 2.5, 0.7, "RequestEncoder", C_ENCODE, fontsize=7)
ax.text(7.75, 1.2, "Hash | Lookup | MCA", ha="center", fontsize=5.5, color="#666")

# ── Knowledge Grid ──
smallbox(10.0, 1.3, 2.5, 0.7, "KnowledgeGrid", C_TECH, fontsize=7)
ax.text(
    11.25,
    1.2,
    "(ongoing library)\nGaussian propagation",
    ha="center",
    fontsize=5.5,
    color="#666",
)

# ═══════════════════════════════════════════════════════════════════════════
# Config (right side)
# ═══════════════════════════════════════════════════════════════════════════
box(
    13.5,
    1.1,
    3.5,
    1.2,
    "KATAConfig",
    C_CONFIG,
    fontsize=9,
    sublabel="JSON | env vars | Python",
)

# ═══════════════════════════════════════════════════════════════════════════
# Connecting arrows
# ═══════════════════════════════════════════════════════════════════════════

# Env -> Dispatcher
arrow(8.0, 7.8, 8.0, 7.3, "action", C_TECH, lw=2)

# Dispatcher -> Env (repair queue -> obs)
arrow(6.5, 7.3, 3.5, 7.8, "repair requests", C_GYM, lw=1.5)

# Dispatcher <-> Technicians
arrow(11.0, 6.4, 11.5, 6.4, "assign", C_TECH)
arrow(11.5, 6.0, 11.0, 6.0, "status", C_TECH)

# Machine -> Dispatcher (breakdown)
curved_arrow(4.8, 5.0, 5.5, 5.9, "request_repair()", C_MACHINE, rad=-0.3)

# Dispatcher -> Machine (repair complete)
curved_arrow(6.5, 5.5, 5.5, 4.8, "repair()", C_MACHINE, rad=-0.3)

# Repair request -> Encoder -> Knowledge Grid
arrow(5.5, 1.65, 6.5, 1.65, color=C_ENCODE)
arrow(9.0, 1.65, 10.0, 1.65, "encode()", C_ENCODE)

# Knowledge Grid -> Technician
curved_arrow(11.25, 2.0, 14.0, 5.5, "knowledge\nupdate", C_TECH, rad=0.3)

# Machine -> RepairRequest
arrow(3.2, 3.0, 3.8, 2.0, color=C_MACHINE)

# Production -> Machines (products flow)
curved_arrow(8.8, 3.5, 7.8, 3.5, "products", C_PROD, rad=0.3)

# Config -> everything
arrow(15.25, 2.3, 15.25, 3.0, color=C_CONFIG, style="-|>", lw=1)
arrow(14.0, 2.3, 8.5, 5.5, color=C_CONFIG, style="-|>", lw=1)

# MCA warmup -> Encoder
curved_arrow(13.5, 8.8, 7.75, 2.0, "fit()", C_ENCODE, rad=0.4)

# ═══════════════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════════════
legend_patches = [
    mpatches.Patch(color=C_GYM, label="Gymnasium Layer"),
    mpatches.Patch(color=C_SIM, label="SimPy Simulation"),
    mpatches.Patch(color=C_TECH, label="Technicians & Dispatch"),
    mpatches.Patch(color=C_MACHINE, label="Machines & Breakdowns"),
    mpatches.Patch(color=C_PROD, label="Production Network"),
    mpatches.Patch(color=C_ENCODE, label="Encoding & MCA"),
    mpatches.Patch(color=C_CONFIG, label="Configuration"),
]
ax.legend(
    handles=legend_patches,
    loc="lower left",
    fontsize=7.5,
    ncol=4,
    framealpha=0.9,
    bbox_to_anchor=(0.0, -0.02),
)

plt.tight_layout()
plt.savefig(
    "specifications/architecture.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.savefig(
    "specifications/architecture.pdf",
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("Saved specifications/architecture.png and specifications/architecture.pdf")
