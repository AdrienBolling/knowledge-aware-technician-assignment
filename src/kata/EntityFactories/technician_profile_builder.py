"""Build pre-seeded ``ongoing.KnowledgeGrid`` artefacts for technician profiles.

A "profile" here is a (career-experience, expertise-shape) pair that lives
*outside* the simulator's runtime knobs.  ``TechnicianConfig`` only controls
the dynamics (how fast a tech learns / fatigues); it knows nothing about
*how much* expertise a tech walks into the episode with, or *where* on the
embedding grid that expertise sits.  This module fills that gap by
generating a saved knowledge grid that ``GymTechnician`` will load at every
``env.reset()`` via the ``initial_knowledge_grid_path`` field.

The build pipeline is deterministic: given a profile spec and a seed, the
output bytes are reproducible across machines.  The companion notebook
``notebooks/build_technician_profiles.ipynb`` simply imports this module
and calls :func:`build_default_profiles` --- the same function the CI uses
to regenerate the bundled artefacts when the spec changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from ongoing import KnowledgeGrid

# Default location of the bundled grids (resolved relative to the repo
# root or the package's ``src/kata/resources/`` directory).
DEFAULT_OUTPUT_DIR = Path("src/kata/resources/technician_profiles")


@dataclass(frozen=True)
class ProfileSpec:
    """Description of a single technician profile.

    Attributes
    ----------
    name:
        Filename stem; the grid is saved to ``<output_dir>/<name>.npz``.
    grid_shape:
        Knowledge-grid shape.  Must match the corresponding
        ``TechnicianConfig.knowledge_k_shape``.
    propagation_sigma, transmission_factor, learning_rate:
        Knowledge-grid hyperparameters; must match the corresponding
        ``TechnicianConfig.knowledge_*`` fields.
    n_tickets:
        How many synthetic experience events to seed the grid with.
        Larger = more accumulated career.
    embedding_sampler:
        Callable ``(rng, upper) -> np.ndarray`` returning one embedding
        per call, where ``upper`` is the per-dimension upper bound of
        the embedding space (default ``100``, matching the encoders'
        default range and ``KnowledgeGrid``'s default
        ``embedding_bounds``).  A uniform sampler produces a broad
        generalist; a Gaussian sampler around a single neighbourhood
        produces a specialist.
    embedding_upper:
        Upper bound of the embedding space (the lower bound is 0).
        Defaults to ``100.0`` so samples live in the same coordinate
        space that ``KnowledgeGrid.embedding_to_coords`` expects.  If
        you change this, also pass ``embedding_bounds`` to the grid
        constructor in :func:`build_one`.
    description:
        Short human-readable summary, written into a sidecar
        ``profiles.txt`` so the bundled directory is self-documenting.
    """

    name: str
    grid_shape: tuple[int, ...]
    propagation_sigma: float
    transmission_factor: float
    learning_rate: float
    n_tickets: int
    embedding_sampler: Callable[[np.random.Generator, float], np.ndarray]
    embedding_upper: float = 100.0
    description: str = ""
    methods: tuple[str, ...] = field(default_factory=lambda: ("propagation",))


# ---------------------------------------------------------------------------
# Embedding samplers --- shape the spatial distribution of seeded experience
# ---------------------------------------------------------------------------


def uniform_sampler(
    rng: np.random.Generator, upper: float
) -> np.ndarray:
    """Uniform across the whole embedding space --- the generalist's signature."""
    return rng.uniform(0.0, upper, size=2)


def gaussian_at(
    centre: tuple[float, float], sigma: float = 10.0
) -> Callable[[np.random.Generator, float], np.ndarray]:
    """Sampler concentrated around ``centre`` with std ``sigma``.

    ``centre`` and ``sigma`` are expressed in the *embedding-space*
    coordinate system (default range ``[0, 100]``), not in grid-cell
    units.  A ``sigma`` of ~10 on a 10x10 grid corresponds to roughly
    one grid cell of spread.
    """
    centre_arr = np.asarray(centre, dtype=np.float64)

    def sample(rng: np.random.Generator, upper: float) -> np.ndarray:
        pt = rng.normal(loc=centre_arr, scale=sigma)
        return np.clip(pt, 0.0, upper - 1e-6)

    return sample


def mixture(
    samplers: Sequence[Callable[[np.random.Generator, float], np.ndarray]],
    weights: Sequence[float],
) -> Callable[[np.random.Generator, float], np.ndarray]:
    """Mixture of several samplers --- e.g. a senior with deep + broad coverage."""
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    samplers_list = list(samplers)

    def sample(rng: np.random.Generator, upper: float) -> np.ndarray:
        choice = int(rng.choice(len(samplers_list), p=weights_arr))
        return samplers_list[choice](rng, upper)

    return sample


# ---------------------------------------------------------------------------
# Profile catalogue
# ---------------------------------------------------------------------------


def default_profiles() -> list[ProfileSpec]:
    """Bundled profile catalogue.

    Names match (and extend) the templates in
    ``src/kata/resources/templates/technician_templates.json``.  When
    adding a new specialist here, also add a corresponding template
    that points at the generated ``.npz`` file via
    ``initial_knowledge_grid_path``.
    """
    shape = (10, 10)
    # Centres for the two specialist profiles in the embedding-space
    # coordinate system (default range [0, 100]).  Bottom-left and
    # top-right corners chosen so the two specialists' expertise
    # neighbourhoods overlap minimally on the grid.
    motor_centre = (25.0, 25.0)
    electronics_centre = (75.0, 75.0)

    return [
        ProfileSpec(
            name="trainee",
            grid_shape=shape,
            propagation_sigma=0.3,
            transmission_factor=0.2,
            learning_rate=0.95,
            n_tickets=5,
            embedding_sampler=uniform_sampler,
            description=(
                "Fresh hire: ~5 lifetime repairs spread thinly across "
                "the entire grid.  Knowledge volume near zero everywhere."
            ),
        ),
        ProfileSpec(
            name="junior",
            grid_shape=shape,
            propagation_sigma=0.5,
            transmission_factor=0.3,
            learning_rate=0.85,
            n_tickets=40,
            embedding_sampler=uniform_sampler,
            description=(
                "A few months of broad on-the-job exposure: 40 repairs "
                "uniformly across component families, no specialty."
            ),
        ),
        ProfileSpec(
            name="generalist",
            grid_shape=shape,
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.7,
            n_tickets=200,
            embedding_sampler=uniform_sampler,
            description=(
                "Several years of broad maintenance work: 200 repairs "
                "uniform over the grid.  Workhorse, no deep specialty."
            ),
        ),
        ProfileSpec(
            name="senior",
            grid_shape=shape,
            propagation_sigma=1.5,
            transmission_factor=0.7,
            learning_rate=0.6,
            n_tickets=500,
            embedding_sampler=mixture(
                [
                    uniform_sampler,
                    gaussian_at(motor_centre, sigma=15.0),
                ],
                weights=[0.7, 0.3],
            ),
            description=(
                "Veteran with broad coverage and a soft mechanical "
                "specialty: 500 lifetime repairs, 70% uniform, 30% "
                "biased toward the mechanical cluster."
            ),
        ),
        ProfileSpec(
            name="expert",
            grid_shape=shape,
            propagation_sigma=1.5,
            transmission_factor=0.7,
            learning_rate=0.51,
            n_tickets=1500,
            embedding_sampler=uniform_sampler,
            description=(
                "Highly experienced generalist: 1500 lifetime repairs, "
                "broad coverage of the entire grid.  Knowledge ceiling "
                "approaches the asymptotic floor for every component "
                "family."
            ),
        ),
        ProfileSpec(
            name="motor_specialist",
            grid_shape=shape,
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.6,
            n_tickets=600,
            embedding_sampler=mixture(
                [
                    gaussian_at(motor_centre, sigma=10.0),
                    uniform_sampler,
                ],
                weights=[0.85, 0.15],
            ),
            description=(
                "Deep mechanical / rotating-parts specialist: 85% of "
                "600 lifetime repairs concentrated around the motor "
                "neighbourhood, 15% incidental exposure elsewhere."
            ),
        ),
        ProfileSpec(
            name="electronics_specialist",
            grid_shape=shape,
            propagation_sigma=1.0,
            transmission_factor=0.5,
            learning_rate=0.6,
            n_tickets=600,
            embedding_sampler=mixture(
                [
                    gaussian_at(electronics_centre, sigma=10.0),
                    uniform_sampler,
                ],
                weights=[0.85, 0.15],
            ),
            description=(
                "Deep electronics specialist: 85% of 600 lifetime "
                "repairs concentrated around the electronics / "
                "controller neighbourhood, 15% incidental exposure."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build_one(spec: ProfileSpec, *, seed: int = 0) -> KnowledgeGrid:
    """Construct a :class:`KnowledgeGrid` populated from ``spec``."""
    rng = np.random.default_rng(seed)
    grid = KnowledgeGrid(
        shape=spec.grid_shape,
        propagation_sigma=spec.propagation_sigma,
        transmission_factor=spec.transmission_factor,
        learning_rate=spec.learning_rate,
        methods=list(spec.methods),
    )
    for _ in range(spec.n_tickets):
        emb = spec.embedding_sampler(rng, spec.embedding_upper)
        grid.add_ticket_knowledge(np.asarray(emb, dtype=np.float64))
    return grid


def build_default_profiles(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    seed: int = 0,
    verbose: bool = True,
) -> list[Path]:
    """Build all bundled profiles and write ``.npz`` + sidecar ``profiles.txt``.

    Returns the list of paths written.  Designed to be called both from
    the companion notebook and from a CI smoke job that regenerates
    the bundled artefacts after a spec change.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sidecar_lines: list[str] = [
        "# Bundled technician profiles --- generated by "
        "kata.EntityFactories.technician_profile_builder.build_default_profiles",
        "# Each line: name | n_tickets | description",
        "",
    ]
    for spec in default_profiles():
        grid = build_one(spec, seed=seed)
        out = output_dir / f"{spec.name}.npz"
        grid.save(out)
        written.append(out)
        sidecar_lines.append(f"{spec.name} | {spec.n_tickets} | {spec.description}")
        if verbose:
            stats = (
                f"max_knowledge={grid.get_max_knowledge():.3f}  "
                f"max_experiences={grid.get_max_experiences():.1f}  "
                f"specialisation_index={grid.specialisation_index():.3f}"
            )
            print(f"wrote {out.name:<32s} {stats}")
    (output_dir / "profiles.txt").write_text("\n".join(sidecar_lines) + "\n")
    return written


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "ProfileSpec",
    "build_default_profiles",
    "build_one",
    "default_profiles",
    "gaussian_at",
    "mixture",
    "uniform_sampler",
]
