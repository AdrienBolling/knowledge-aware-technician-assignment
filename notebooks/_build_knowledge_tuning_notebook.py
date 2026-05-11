"""Builds notebooks/knowledge_curve_tuning.ipynb.

Run with::

    PYTHONPATH=src uv run python notebooks/_build_knowledge_tuning_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md(r"""# Tuning the knowledge → repair-time response

The knowledge multiplier used by `GymTechnician.compute_repair_time` was
recently changed from the unbounded $1/(1+k)$ to a **saturating
exponential** with two knobs:

$$
m_k(k;\,\text{min\_floor},\,\alpha)
\;=\; \text{min\_floor}\,+\,(1-\text{min\_floor})\cdot e^{-\alpha\, k}
$$

Two parameters live under `sim.repair`:

| field | symbol | meaning |
|-------|--------|---------|
| `min_repair_fraction` | `min_floor` | hard lower bound on the multiplier — the most a perfectly skilled tech can shave off |
| `knowledge_sensitivity` | `alpha` | how quickly knowledge translates into a speed-up |

Defaults: `min_floor = 0.3`, `alpha = 0.002` — chosen so that with
the shipped knowledge-grid settings (`learning_rate = 0.1`,
`propagation_sigma = 1.0`) the multiplier saturates at `min_floor`
after ~60–70 similar repairs.

This notebook plots the response under a sweep of values so you can
pick parameters that reflect your domain assumptions.""")


code("""import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ongoing import KnowledgeGrid

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

def m_k(k, min_floor, alpha):
    \"\"\"Saturating-exponential knowledge multiplier.\"\"\"
    return min_floor + (1.0 - min_floor) * np.exp(-alpha * np.asarray(k))""")


md(r"""## 1. Pure response curve $m_k(k)$

How the multiplier behaves as a function of raw knowledge $k$.  The
old formula $1/(1+k)$ is overlaid as a reference.""")


code("""# k spans 0 .. ~5000 with the shipped knowledge-grid defaults after ~80 repairs
k_grid = np.linspace(0, 5000, 1001)
old = 1.0 / (1.0 + k_grid)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Vary alpha (sensitivity) — fixed min_floor
for a in [0.0005, 0.001, 0.002, 0.005, 0.01]:
    axes[0].plot(k_grid, m_k(k_grid, 0.3, a), label=f"alpha={a}")
axes[0].plot(k_grid, old, "k--", lw=1, label="old: 1/(1+k)")
axes[0].set(xlabel="knowledge k", ylabel="multiplier m_k",
            title="Vary alpha  (min_floor = 0.3)")
axes[0].legend()

# Vary min_floor — fixed alpha = default 0.002
for f in [0.0, 0.1, 0.3, 0.5, 0.7]:
    axes[1].plot(k_grid, m_k(k_grid, f, 0.002), label=f"min_floor={f}")
axes[1].plot(k_grid, old, "k--", lw=1, label="old: 1/(1+k)")
axes[1].set(xlabel="knowledge k", ylabel="multiplier m_k",
            title="Vary min_floor  (alpha = 0.002)")
axes[1].legend()

fig.tight_layout()
plt.show()""")


md(r"""**Reading.**

* `alpha` controls *how aggressive* knowledge is at low-experience.
  At $\alpha=1$ the curve mirrors the old $1/(1+k)$ near $k=0$ but
  flattens to `min_floor` instead of zero.  Smaller $\alpha$ pulls the
  whole curve up — knowledge buys a smaller speed-up per repair.
* `min_floor` is a hard cap.  At `min_floor=0` the multiplier reaches
  zero asymptotically (still bounded but the speed-up is unlimited);
  at `min_floor=0.5` even infinite expertise still leaves the repair
  taking half its base time.""")


md("""## 2. Curve as a function of *similar repairs done*

The agent doesn't see $k$ directly — it sees "I just did the same
repair $n$ times".  Convert the response to that axis using a real
`KnowledgeGrid` (defaults: `learning_rate=0.1`, `propagation_sigma=1`).""")


code("""def repeat_and_record(n, *, learning_rate=0.1, propagation_sigma=1.0):
    grid = KnowledgeGrid(
        shape=(10, 10),
        propagation_sigma=propagation_sigma,
        transmission_factor=0.5,
        learning_rate=learning_rate,
        methods=["propagation"],
    )
    emb = np.array([50.0, 50.0])  # mid-grid embedding
    k = np.empty(n)
    for i in range(n):
        grid.add_ticket_knowledge(emb)
        k[i] = grid.get_knowledge(emb)
    return k

N = 100
k_curve = repeat_and_record(N)
n_axis = np.arange(1, N + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for a in [0.0005, 0.001, 0.002, 0.005, 0.01]:
    axes[0].plot(n_axis, m_k(k_curve, 0.3, a), marker="o", ms=3, label=f"alpha={a}")
axes[0].set(xlabel="# of similar repairs", ylabel="multiplier m_k",
            title="Effect of alpha  (min_floor = 0.3)")
axes[0].set_ylim(0, 1.05)
axes[0].legend()

for f in [0.0, 0.1, 0.3, 0.5, 0.7]:
    axes[1].plot(n_axis, m_k(k_curve, f, 0.002), marker="o", ms=3, label=f"min_floor={f}")
axes[1].set(xlabel="# of similar repairs", ylabel="multiplier m_k",
            title="Effect of min_floor  (alpha = 0.002)")
axes[1].set_ylim(0, 1.05)
axes[1].legend()

fig.tight_layout()
plt.show()""")


md("""## 3. Effective repair time at the simulator scale

Same plot as above but converted to *effective minutes* assuming a
heavy-machine repair with `base = 480` (the spindle in the
`cnc_weibull` template).""")


code("""BASE = 480.0  # base repair time in time units (minutes), spindle
fig, ax = plt.subplots()
for f, a, label in [
    (0.0, 1.0,    "old: 1/(1+k)  [reference]"),
    (0.3, 0.002,  "default (0.3, 0.002) — saturates ~65 repairs"),
    (0.3, 0.001,  "gentler (0.3, 0.001) — saturates ~120 repairs"),
    (0.3, 0.005,  "snappier (0.3, 0.005) — saturates ~30 repairs"),
    (0.5, 0.002,  "higher floor (0.5, 0.002)"),
]:
    if label.startswith("old"):
        eff = BASE / (1.0 + k_curve)
        ax.plot(n_axis, eff, "k--", lw=1, label=label)
    else:
        eff = BASE * m_k(k_curve, f, a)
        ax.plot(n_axis, eff, marker="o", ms=3, label=label)

ax.axhline(BASE, color="grey", lw=0.8, ls=":")
ax.set(xlabel="# of similar repairs", ylabel="effective repair time (minutes)",
       title=f"Effective repair time  (base = {BASE:.0f} min, spindle)")
ax.legend()
plt.show()""")


md(r"""## 4. How many similar repairs to reach a target speed-up?

For each `(min_floor, alpha)` candidate, count the smallest $n$ such
that $m_k(k_n) \le \tau$ for $\tau \in \{0.9, 0.7, 0.5\}$ (i.e. 10 %,
30 %, 50 % faster than base).  ``--`` means the floor is above the
target so it can never be reached.""")


code("""candidates = [
    ("default",        0.3, 0.002),
    ("snappier",       0.3, 0.005),
    ("gentler",        0.3, 0.001),
    ("very gentle",    0.3, 0.0005),
    ("higher floor",   0.5, 0.002),
    ("aggressive",     0.2, 0.01),
]
# 0.4 is just above the 0.3 floor — used to detect "near-floor"
targets = [0.9, 0.7, 0.5, 0.4]

# Use a longer learning curve to reach low targets
N_long = 200
k_long = repeat_and_record(N_long)

def first_n_to_reach(target, k_curve, min_floor, alpha):
    if target < min_floor:
        return None
    multiplier = m_k(k_curve, min_floor, alpha)
    hits = np.where(multiplier <= target)[0]
    return int(hits[0]) + 1 if len(hits) else None

print(f"{'preset':<14}{'min_floor':>10}{'alpha':>7} | "
      + "  ".join(f"<= {t}" for t in targets))
print("-" * 60)
for name, f, a in candidates:
    cells = []
    for t in targets:
        n = first_n_to_reach(t, k_long, f, a)
        cells.append(f"{n:>5}" if n is not None else "   --")
    print(f"{name:<14}{f:>10}{a:>7} |" + "".join(cells))""")


md("""## 5. Hands-on: try your own values

Edit the dictionary below and re-run the cell to see the resulting
curve overlaid on the default response.""")


code("""user_choice = {"min_floor": 0.3, "alpha": 0.002}  # <-- edit me

fig, ax = plt.subplots()
ax.plot(n_axis, m_k(k_curve, 0.3, 0.002), color="grey", ls="--",
        lw=1.0, label="default (0.3, 0.002)")
ax.plot(
    n_axis,
    m_k(k_curve, user_choice["min_floor"], user_choice["alpha"]),
    marker="o", ms=3, color="C3",
    label=f"your choice ({user_choice['min_floor']}, {user_choice['alpha']})",
)
ax.set(xlabel="# of similar repairs", ylabel="multiplier m_k",
       title="Compare your candidate against the shipped default")
ax.set_ylim(0, 1.05)
ax.legend()
plt.show()

# Numerical summary
chosen = m_k(k_curve, user_choice["min_floor"], user_choice["alpha"])
print(f"After {N} repairs:")
print(f"  multiplier   = {chosen[-1]:.3f}")
print(f"  effective at base=100  -> {100 * chosen[-1]:.1f} time units")
print(f"  effective at base={BASE:.0f}  -> {BASE * chosen[-1]:.1f} time units")
print(f"  asymptotic floor       -> multiplier = {user_choice['min_floor']:.3f}")""")


md(r"""## Picking values — quick guide

* **Start by deciding the asymptotic floor (`min_floor`).**  This is
  domain knowledge: how fast can a perfect technician *ever* get?
  Half-speed (`0.5`) is conservative, 30 % (`0.3`) implies real
  expertise can shave off 70 %.

* **Then pick the slope (`alpha`)** based on how many repairs of one
  kind a tech sees before becoming "expert".  Use the table in
  Section 4: an `alpha` that reaches the desired speed-up around
  10–20 repairs is usually a reasonable starting point in this codebase.

* **Sanity check Section 3** with a representative `BASE` value (e.g.
  the spindle's 480 min) to confirm absolute numbers feel right.

* **Be aware of the encoder issue** flagged in
  `knowledge_impact.ipynb`: the `HashEncoder` currently maps every
  ticket to grid cell `(0, 0)`, so in the running simulator knowledge
  accumulates on a *single* cell across all repair types.  Until that
  is fixed, choose parameters expecting the curve to be sampled at
  large `k` quickly.

To apply your choice, edit `run_configs/<name>.json`:

```json
{
  "sim": {
    "repair": {
      "knowledge_enabled": true,
      "min_repair_fraction": 0.4,
      "knowledge_sensitivity": 0.15
    }
  }
}
```
""")


# ---------------------------------------------------------------------------
# Build & write the notebook
# ---------------------------------------------------------------------------

nb_cells: list[dict] = []
for cell_type, source in CELLS:
    cell: dict = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb_cells.append(cell)

notebook = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parent / "knowledge_curve_tuning.ipynb"
target.write_text(json.dumps(notebook, indent=1))
print(f"wrote {target}")
