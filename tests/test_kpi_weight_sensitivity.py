"""KPI-weight sensitivity and Pareto analysis of the summary score.

Pins the properties the appendix relies on:

* equal weights reproduce ``summary_scores.score`` (the published score);
* a policy that another policy dominates is not in the Pareto set, and
  exactly equal KPI vectors are ties, not dominance;
* a different best value only rescales that KPI's gaps and adds a constant,
  so the ranks equal the ranks under reweighted KPIs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture(scope="module")
def kw():
    spec = importlib.util.spec_from_file_location("kpi_weight_sensitivity", SCRIPTS / "kpi_weight_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kpi_weight_sensitivity"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def metrics():
    # Columns: the four scored KPIs (products up, MTTR down, disruptions down, knowledge up).
    return pd.DataFrame(
        {"prod": [1000.0, 950.0, 990.0, 900.0, 990.0],
         "mttr_mean": [80.0, 85.0, 78.0, 90.0, 78.0],
         "disr": [500.0, 520.0, 560.0, 600.0, 560.0],
         "know": [60.0, 59.0, 58.0, 55.0, 58.0]},
        index=["a", "b", "c", "d", "e"])


def test_kpis_are_the_paper_variant(kw):
    assert kw.KPIS == ["prod", "mttr_mean", "disr", "know"]


def test_equal_weights_reproduce_summary_score(kw, metrics):
    field = list(metrics.index)
    G = kw.gap_matrix(metrics, field)
    eq = kw.weighted_scores(G.to_numpy(), np.full(4, 0.25))[0]
    np.testing.assert_allclose(eq, kw.ss.score(metrics, kw.KPIS, field).to_numpy())
    # Worked value: b has gaps 5, 8.97, 4 and 1.67 % (bests 1000, 78, 500, 60).
    assert eq[1] == pytest.approx((5.0 + 700.0 / 78.0 + 4.0 + 100.0 / 60.0) / 4)


def test_corner_weight_gives_the_single_kpi_gap(kw, metrics):
    G = kw.gap_matrix(metrics, list(metrics.index))
    corner = kw.weighted_scores(G.to_numpy(), np.array([0.0, 0.0, 1.0, 0.0]))[0]
    np.testing.assert_allclose(corner, G["disr"].to_numpy())


def test_weight_draws_are_seeded_and_on_the_simplex(kw):
    W = kw.draw_weights(500, seed=7)
    assert W.shape == (500, 4)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert (W >= 0).all()
    np.testing.assert_array_equal(W, kw.draw_weights(500, seed=7))


def test_min_ranks_share_ties(kw):
    np.testing.assert_array_equal(kw.min_ranks(np.array([[2.0, 1.0, 2.0, 3.0]]))[0], [2, 1, 2, 4])


def test_dominated_policy_is_not_in_the_pareto_set(kw, metrics):
    directions = [+1, -1, -1, +1]
    n_dom, by, ties = kw.dominators(metrics[kw.KPIS].to_numpy(), directions)
    names = list(metrics.index)
    # a is better than b on every KPI; c trades MTTR for disruptions and knowledge;
    # every other policy is better than d on every KPI.
    assert dict(zip(names, n_dom)) == {"a": 0, "b": 1, "c": 0, "d": 4, "e": 0}
    assert [names[j] for j in by[1]] == ["a"]
    assert [names[j] for j in by[3]] == ["a", "b", "c", "e"]
    # c and e have equal KPI vectors: a tie, and both stay in the Pareto set.
    assert [names[j] for j in ties[2]] == ["e"] and [names[j] for j in ties[4]] == ["c"]


def test_new_best_value_acts_as_a_reweighting(kw, metrics):
    """Bests over a sub-roster: gap' = (b/b') gap + const, so the ranks follow reweighted KPIs."""
    field, sub = list(metrics.index), ["b", "c", "d", "e"]
    G, G_sub = kw.gap_matrix(metrics, field).loc[sub], kw.gap_matrix(metrics, sub)
    ratio = (kw.bests(metrics, field) / kw.bests(metrics, sub)).to_numpy()
    const = (G_sub - G * ratio).to_numpy()
    np.testing.assert_allclose(const, const[0][None, :].repeat(len(sub), axis=0), atol=1e-9)
    W = kw.draw_weights(200, seed=3)
    W_implied = W * ratio
    np.testing.assert_array_equal(kw.min_ranks(kw.weighted_scores(G_sub.to_numpy(), W)),
                                  kw.min_ranks(kw.weighted_scores(G.to_numpy(), W_implied)))


REPO = SCRIPTS.parent
# tab:results_dist (one decimal): S1..S5 and overall.
PUBLISHED = {"hc_v6": [6.1, 6.1, 3.4, 3.7, 3.8, 4.6],
             "ft_quality": [10.3, 14.6, 0.5, 1.8, 0.9, 5.6],
             "topsis": [4.5, 7.5, 1.5, 3.9, 6.4, 4.8]}


@pytest.mark.skipif(not (REPO / "reports/hvp_eval_v6w/lifecycle/episodes.csv").exists(),
                    reason="benchmark reports are not in the repository")
def test_equal_weights_reproduce_the_published_table(kw, monkeypatch):
    monkeypatch.chdir(REPO)
    metrics = kw.ss.load_metrics()
    field = kw.ss.main_field(metrics)
    t, _ = kw.ranks_table(metrics, field, kw.SHOWN)
    for agent, row in PUBLISHED.items():
        np.testing.assert_allclose(t.loc[agent].round(1).to_numpy(), row)
