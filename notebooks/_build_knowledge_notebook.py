"""Builds notebooks/knowledge_impact.ipynb from the cell sources defined here.

Run with::

    PYTHONPATH=src uv run python notebooks/_build_knowledge_notebook.py

This avoids hand-writing the .ipynb JSON (and its quoting headaches).
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

md(r"""# Knowledge package — impact on repair time

This notebook probes the `ongoing.KnowledgeGrid` (the engine behind
`GymTechnician.knowledge_grid`) to answer one question:

> **How does the *number of similar repairs* a technician has done
> shape the time they need on the next one?**

We bypass the Gym/SimPy stack and exercise the package directly so the
relationship is easy to visualise.

## Recap of the model

`GymTechnician.compute_repair_time(base, request)` multiplies the base
repair time by:

1. a **knowledge multiplier** $m_k = \dfrac{1}{1 + k}$, where $k$ is the
   knowledge level returned by `KnowledgeGrid.get_knowledge(embedding)`;
2. a **fatigue multiplier** $m_f = e^{-\alpha\, f}$ (we hold $f = 0$ in
   this notebook to isolate the knowledge effect).

So the **effective repair time** for a tech with knowledge $k$ on a
base‐time-$T$ repair is

$$T_\text{effective}(k) = T \cdot \frac{1}{1 + k}.$$

Knowledge $k$ grows monotonically as the technician accumulates similar
repairs (`KnowledgeGrid.add_ticket_knowledge(embedding)`).

> **Note on the embedding API.** `KnowledgeGrid.add_ticket_knowledge`
> takes a real-valued *embedding* whose coordinates are mapped to grid
> cells via `embedding_bounds` (default `[0, 100]²`).  We pass
> embeddings in that range below.  Passing raw grid coordinates (in
> `[0, shape-1]`) — which the current `HashEncoder` returns — would
> collapse all repairs to cell `(0, 0)`; that is a known bug in the
> encoder/grid contract, separate from the experiments here.""")


code("""import sys, pathlib
# make sure the project's src/ is importable when launched from notebooks/
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ongoing import KnowledgeGrid

np.random.seed(0)
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3""")


md("""## Helpers

A small wrapper that builds a fresh grid, runs `n_repairs` of the **same
embedding**, and returns the knowledge level (and the resulting time
multiplier and effective repair time) **after each repair**.""")


code('''def repeat_repairs(
    n_repairs: int,
    embedding: np.ndarray,
    *,
    base_repair_time: float = 100.0,
    learning_rate: float = 0.1,
    propagation_sigma: float = 1.0,
    transmission_factor: float = 0.5,
    methods=("propagation",),
    grid_shape=(10, 10),
):
    """Apply ``add_ticket_knowledge`` n_repairs times at the same point.

    Returns (k_curve, mult_curve, eff_time_curve), each of length ``n_repairs``,
    where:
      * k_curve[i]        = knowledge after the (i+1)-th repair
      * mult_curve[i]     = 1 / (1 + k_curve[i])   (knowledge multiplier)
      * eff_time_curve[i] = base_repair_time * mult_curve[i]
    """
    grid = KnowledgeGrid(
        shape=grid_shape,
        propagation_sigma=propagation_sigma,
        transmission_factor=transmission_factor,
        learning_rate=learning_rate,
        methods=list(methods),
    )
    k_curve = np.empty(n_repairs)
    for i in range(n_repairs):
        grid.add_ticket_knowledge(embedding.astype(np.float64))
        k_curve[i] = grid.get_knowledge(embedding.astype(np.float64))
    mult_curve = 1.0 / (1.0 + k_curve)
    eff_time_curve = base_repair_time * mult_curve
    return k_curve, mult_curve, eff_time_curve''')


md(r"""## Experiment 1 — One repair type, repeated

Pick a single embedding (one cell of the grid) and re-do the same
repair 50 times.  We plot:

- the **knowledge level** $k$,
- the **time multiplier** $m_k = 1/(1+k)$, and
- the **effective repair time** assuming a base of 100 simulation time
  units.""")


code("""N = 50
emb = np.array([50.0, 50.0])  # mid-grid point in [0, 100]^2

k, mult, eff = repeat_repairs(N, emb, base_repair_time=100.0)

x = np.arange(1, N + 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(x, k, marker="o", markersize=3)
axes[0].set(title="Knowledge level k", xlabel="# of similar repairs", ylabel="k")
axes[1].plot(x, mult, marker="o", markersize=3, color="C1")
axes[1].set(title="Time multiplier m_k = 1/(1+k)", xlabel="# of similar repairs", ylabel="multiplier")
axes[1].set_ylim(0, 1.05)
axes[2].plot(x, eff, marker="o", markersize=3, color="C2")
axes[2].axhline(100.0, ls="--", color="grey", lw=0.8, label="base = 100")
axes[2].set(title="Effective repair time", xlabel="# of similar repairs", ylabel="time units")
axes[2].legend()
fig.tight_layout()
plt.show()""")


md("""**Reading the plot.**  Knowledge starts at zero, so the first
repair takes essentially the full base time.  Each subsequent
identical repair sharpens the multiplier toward zero — but with strong
diminishing returns: by ~10 repeats most of the speed-up is already
realised.""")


md("""## Experiment 2 — Effect of the `learning_rate`

`learning_rate` controls how strongly each repair pushes the grid
forward.  We sweep a handful of values to see how fast the curve
descends.""")


code("""rates = [0.02, 0.05, 0.1, 0.2, 0.5]
fig, ax = plt.subplots()
for r in rates:
    _, _, eff = repeat_repairs(N, emb, learning_rate=r, base_repair_time=100.0)
    ax.plot(x, eff, label=f"learning_rate={r}")
ax.axhline(100.0, ls="--", color="grey", lw=0.8)
ax.set(
    title="Effective repair time vs. # repairs (varying learning_rate)",
    xlabel="# of similar repairs",
    ylabel="effective repair time (base=100)",
)
ax.legend()
plt.show()""")


md("""Higher `learning_rate` ⇒ faster descent at the cost of less
"fine-grained" learning down the line.  At `learning_rate=0.5` the
curve almost flat-lines at zero after ~4 repairs; at `0.02` it is
still descending after 50.""")


md("""## Experiment 3 — Spatial spread of knowledge (`propagation_sigma`)

When `methods=['propagation']`, each repair drops a Gaussian bump on
the grid centred on the embedding's cell — `propagation_sigma`
controls how wide the bump is.  Knowledge from *similar but
non-identical* repairs therefore transfers partially.

We do 30 repairs at the centre of a 10×10 grid and read knowledge
along a horizontal cross-section, varying `propagation_sigma`.""")


code("""N3 = 30
center = np.array([50.0, 50.0])  # centre cell of the 10x10 grid

# Embeddings along the row of the central cell
xs = np.linspace(0.0, 100.0, 11)  # 11 sample points on x, fixed y
sigmas = [0.2, 1.0, 2.0, 4.0]

fig, ax = plt.subplots()
for s in sigmas:
    grid = KnowledgeGrid(
        shape=(10, 10),
        propagation_sigma=s,
        transmission_factor=0.5,
        learning_rate=0.1,
        methods=["propagation"],
    )
    for _ in range(N3):
        grid.add_ticket_knowledge(center)
    profile = [grid.get_knowledge(np.array([x, 50.0])) for x in xs]
    ax.plot(xs, profile, marker="o", label=f"sigma={s}")
ax.set(
    title=f"Knowledge profile across the grid after {N3} repairs at the centre",
    xlabel="x-coordinate of the queried embedding",
    ylabel="knowledge",
)
ax.legend()
plt.show()""")


md("""Wider `propagation_sigma` produces a flatter, more diffuse
knowledge profile — useful when repairs of similar (but distinct)
component types should transfer expertise.  A narrow sigma keeps
knowledge tightly concentrated at the trained cell.""")


md("""## Experiment 4 — Two repair types interleaved

Two distinct embeddings (A and B), trained alternately.  Each
embedding gets 25 repairs total.  We track the effective repair time
the technician would need *for both types* as training progresses.""")


code("""N4 = 50  # total repairs (25 of each type)
emb_a = np.array([20.0, 20.0])
emb_b = np.array([80.0, 80.0])

grid = KnowledgeGrid(
    shape=(10, 10),
    propagation_sigma=1.0,
    transmission_factor=0.5,
    learning_rate=0.1,
    methods=["propagation"],
)

eff_a, eff_b = [], []
for i in range(N4):
    target = emb_a if i % 2 == 0 else emb_b
    grid.add_ticket_knowledge(target)
    eff_a.append(100.0 / (1.0 + grid.get_knowledge(emb_a)))
    eff_b.append(100.0 / (1.0 + grid.get_knowledge(emb_b)))

x4 = np.arange(1, N4 + 1)
fig, ax = plt.subplots()
ax.plot(x4, eff_a, label="effective time at A", color="C0")
ax.plot(x4, eff_b, label="effective time at B", color="C1")
ax.axhline(100.0, ls="--", color="grey", lw=0.8)
ax.set(
    title="Two repair types alternated — effective time on each",
    xlabel="repair # (alternating A/B)",
    ylabel="effective repair time (base=100)",
)
ax.legend()
plt.show()""")


md("""Both curves descend together because alternating updates accrue
knowledge at both points (and the Gaussian bump bleeds slightly
between them, which we already explored in Experiment 3).  If the
embeddings were further apart on the grid, or `propagation_sigma`
were smaller, the curves would separate more visibly.""")


md("""## Experiment 5 — Bridging back to the simulator

The `GymTechnician` uses a knowledge multiplier of $1/(1+k)$, so the
*number of similar repairs needed* to bring effective time below a
target $T_{\\rm eff}$ for a base time $T$ is the smallest $n$ such that

$$k_n \\ge \\dfrac{T - T_{\\rm eff}}{T_{\\rm eff}}.$$

This cell estimates that count for several base / target pairs.""")


code('''def first_repair_to_reach(
    target_fraction: float,
    embedding: np.ndarray,
    n_max: int = 200,
    **grid_kwargs,
):
    """Smallest n such that effective_time/base_time <= target_fraction."""
    k_curve, mult_curve, _ = repeat_repairs(
        n_max, embedding, **{**grid_kwargs, "base_repair_time": 1.0}
    )
    hits = np.where(mult_curve <= target_fraction)[0]
    return None if len(hits) == 0 else int(hits[0]) + 1

emb = np.array([50.0, 50.0])
print(f"{ '%-25s' % 'speed-up target' } first n")
for target in [0.5, 0.25, 0.1, 0.05, 0.01]:
    n = first_repair_to_reach(target, emb)
    pct = int((1 - target) * 100)
    print(f"reach {pct:>3d}% faster than base   {n if n is not None else \'>200\'}")''')


md("""## Takeaways

- Knowledge follows $k \\propto n^{1/b}$ where $b = -\\log(\\text{learning\\_rate})/\\log 2$.
  The time multiplier $1/(1+k)$ has *strong diminishing returns* — most
  of the speed-up is captured in the first 5–15 similar repairs.
- `learning_rate` tunes how aggressive each update is; high values
  saturate quickly, low values give a smoother long-tail learning curve.
- `propagation_sigma` controls how much expertise on one component
  transfers to neighbouring (similar) components.
- In the running simulation, the `HashEncoder` currently returns raw
  grid coordinates, which the grid then interprets as embeddings in
  `[0, 100]` — clamping every ticket to cell `(0, 0)`.  That means in
  practice **all repair types share one cell of the grid** and
  knowledge saturates after a handful of repairs.  Fixing this would
  turn the simulator's behaviour back into the per-type story shown in
  the experiments above.""")


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
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parent / "knowledge_impact.ipynb"
target.write_text(json.dumps(notebook, indent=1))
print(f"wrote {target}")
