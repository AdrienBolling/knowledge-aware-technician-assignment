# CLAUDE.md — Knowledge-Aware Technician Assignment (FactoReal / KATA)

Research paper: **"Knowledge-Aware Technician Allocation: The Long-Term Impact of Technician Upskilling"**
(Bolling, Kubler, Ruiz Rodriguez, Le Traon — SnT). Target: *Journal of Manufacturing Systems*, elsarticle two-column.
Thesis: RL dispatching with human-centric reward (fleet knowledge growth + fatigue balance) beats production-only reward over long horizons.

## Paths (PROTECTED)

| What | Where |
|---|---|
| Manuscript (git **submodule**) | `paper/Manuscript.tex` (+ `paper/biblio.bib`, `paper/figures/`) |
| Simulator (code name `kata`) | `src/kata/` — `env.py` (Gym env), `core/config.py`, `metrics.py` |
| Experiments | `src/experiment/runner.py`, configs in `run_configs/benchmark_suite/` (7 configs) |
| Framework paper the survey builds on | `Human-resource-allocation-problem-in-the-Industry-4-0-A-reference-framework.pdf` (Grillo et al. 2022; bib key `grilloHumanResourceAllocation2022`) |
| Taxonomy-table audit trail | `analysis/taxonomy_row_verification.md` (per-paper reasons for every 2026-07-06 change) |
| Build | `cd paper && latexmk -pdf -interaction=nonstopmode Manuscript.tex` (2 passes to settle refs) |

## Manuscript status (2026-07-06)

| Section | State |
|---|---|
| §1 Intro, §2 Related Work (+ Table 1 taxonomy), §4 FactoReal, §5 MDP Implementation, §7 Results (7.1 setup, 7.2 scales, 7.3 objectives ablation; 6 tables + 2 figures from 2026-07-07 benchmarks), Appendix A | drafted prose |
| §3 HRAP-4.0 formalization | **notation table only — body is commented out/empty** |
| §6 Method (hybrid tokens, PLE/T2V/Fourier, Transformer, MLM, PPO), §7.4 RL ablation, §8 Discussion, §9 Conclusion | skeleton `\emph{}` placeholders |
| Abstract | red placeholder — benchmark numbers now available (see `reports/hvp_eval/`, memory note benchmark-results-hvp) |

## Key decisions

| Decision | Rationale |
|---|---|
| Table 1 codes follow Grillo denominations: `COST EXE RAT QUAL` (classic) / `BAL WELL` (4.0) / `TARDY AVAIL FLEX RISK MOD` (ours); constraints `ELIG RES` / `DISR REC`; Scope + Method blocks marked "ours" | "Go back to Grillo denomination when possible"; fine columns kept because C1–C4/H1–H5 cluster analysis depends on them |
| `PROF`/`PERM` omitted from Table 1; cardinality families untabulated | No surveyed work uses them; dispatching trivially assigns one worker per task (said in legend) |
| Row corrections applied conservatively (43 adds, 11 removals; felberbauer B→A, goncalves C→A) | Only abstract-verifiable changes; judgment calls flagged in `analysis/taxonomy_row_verification.md` |
| §5 documents the MDP **implementation** (state schema, event loop, action semantics, reward table `tab:reward_components`); §6 Method covers the learned encoders | §5 written from actual code (`env.py::_build_token_stream/_set_obs/_action_mask/_reward_for_assignment`) — keep in sync if obs schema or reward stack changes |
| §5 framing: event-driven POMDP with semi-MDP remark; γ discounts decision steps, not sim time | Matches implementation; soften if reviewers balk |
| Reward regimes (production-only vs human-centric subsets/coefficients) deferred to §7 Setup | Env capability vs experiment configuration split |

## Blockers / Warnings

- `zhangGraphNeuralNetworks2022` **citation mismatch**: bib entry is a GNN-book chapter, row is coded as crowdsourcing assignment paper — needs the intended reference.
- Placeholder cites still unresolved (intentional): `ref`, `ref2`, `Marcelo`, `MarceloESWA`, `Chica2017` (intro).
- Legacy ECAI tail (appendix, marked "remove before submission"): 4 missing `ecai-template/fig/*.png`, stray `\\` after itemize (~l.1878), refs to `Sec:Exp:*` — the 5 remaining build errors are all from this; PDF still builds.
- `refcheck` package prints label names in margins — drop before submission.
- Method figure caption is placeholder "Overview of the proposed method" (`fig:method_overview`).
- Notation audit done, **fixes not yet applied**: broken Acronyms block in `tab:hrap` (copy-paste of POMDP rows), table vs body conflicts (`t^c_base` vs `c_time(τ)`, `\star` vs `*`, `m_F` vs `m_f`, `m_k` arg order, `Q_t` vs `\mathcal{Q}(t)`, `\mathcal{T}` kernel-vs-ticket-space collision), collisions λ_c/λ_i, β (3 uses), s (scale vs state).
- Full-text double-checks pending on Table 1: ali TEMP, alvarez SIM/ML, caricato MKS/CMP/TEMP, suer MKS, mcdonald FORG, staruch CAP; henao is a retail case study kept in block A.

## Next steps

1. Apply the notation fix package (rebuild Acronyms block, align table to body conventions, add missing symbols, collision renames ω_c/s_V/[0,G]²).
2. Rework the **human-factors block** of Table 1 (Skill representation / Skill dynamics) — agreed next task; same verification pipeline reusable.
3. Write §3 formalization body (currently just the notation table).
4. Method §6, Setup §7, Results §8 remain to be written; abstract numbers pending benchmarks.

## Session notes (2026-07-06)

- Code side (uncommitted): signed RepairTimeDelta, `mttr_rolling`/`mtbf`/`workload_balance` metrics, 3-mode technician-count scenario sampler + `episodes_per_scenario`, knowledge-sensitivity retune 0.002→0.15 (old default was a bug), new tests.
- Paper submodule work also uncommitted: Grillo realignment, §5, bib repairs (`grillo2022hrap40`→`grilloHumanResourceAllocation2022`, added `kuhn1955hungarian`/`bouajaja2017survey`, tien `collaborator`→`author`), `bm` package, `\graphicspath`, duplicate-label fixes.
