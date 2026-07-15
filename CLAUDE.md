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
- ~~Placeholder cites~~ RESOLVED 2026-07-07: `ref`→bouajaja2017survey, `Marcelo,ref2`→ruizrodriguez2022a+joo+calzavara, `MarceloESWA`→ruiz-rodriguez_dynamic_2024, `Chica2017`→chica2017simheuristics (new entry). **Author check**: intended self-cites for `Marcelo/ref2`? (`bollingKnowledgeawareLearningfocusedMultiObjective` exists in bib but lacks year/venue.)
- ~~Legacy ECAI tail~~ RESOLVED 2026-07-07: `refcheck.sty` defines `\comment`, so `\begin{comment}` blocks were silently RENDERING (6pp of legacy in every PDF + the 5 build errors). Both legacy blocks **deleted** (recoverable from git, commit 50c180d). Build now 0 errors / 0 undefined / 0 multiply-defined, 27pp.
- `refcheck` package prints label names in margins — drop before submission (also the \comment definition source).
- Masked-heuristics re-run pending (`scripts/rerun_masked_heuristics.sh`); §6 prose to update after (esp. 6.2.1 "fatigue–exhaustion spiral" — wrong mechanism, it's disruption-blindness).
- Notation audit **mostly applied 2026-07-07**: Acronyms block rebuilt (15 real acronyms), `t^c_base`→`c_time(τ)`, `c^⋆`→`c^*`, `m_F`+wrong formula→`m_f` (lin/exp), `m_k(k_i,τ)`→`m_k(τ,i)`, `φ:𝒯→[0,S]²` collision removed (φ prose + grid shape G), disruption-cell comma typo, ρ_i added (lr renamed — "diminishing-returns", lower ρ = faster learner). **Still open**: λ_c (reward coeff) vs λ_i (Jaber), β triple-use (Weibull/disruptions), `s` saturation scale vs state in §5 reward caption, `Q_t` vs `\mathcal{Q}(t)`.
- Full-text double-checks pending on Table 1: ali TEMP, alvarez SIM/ML, caricato MKS/CMP/TEMP, suer MKS, mcdonald FORG, staruch CAP; henao is a retail case study kept in block A.

## Next steps

1. Apply the notation fix package (rebuild Acronyms block, align table to body conventions, add missing symbols, collision renames ω_c/s_V/[0,G]²).
2. Rework the **human-factors block** of Table 1 (Skill representation / Skill dynamics) — agreed next task; same verification pipeline reusable.
3. Write §3 formalization body (currently just the notation table).
4. Method §6, Setup §7, Results §8 remain to be written; abstract numbers pending benchmarks.

## Session notes (2026-07-07, part 2)

- Manuscript builds **0 errors / 0 warnings** (28pp). §4 reworked (agent-vs-dispatcher fix, design-properties paragraph, rich figure caption); Appendix A: six-entity fix, φ hash-encoder explanation, ρ semantics + archetype-axis rationale + specialists documented, disruption trigger taxonomy + fatigue→exhaustion loop, defaults (f=0.3, α=0.15, Δ_travel=15), reward-stack paragraph replaced by pointer to §5.4, sampler paragraph updated to 3-mode + episodes_per_scenario, Bernoulli per-cycle clarification, eval-protocol cross-ref. §6.1 now states unit coefficients + normalization.

## Session notes (2026-07-06)

- Code side (uncommitted): signed RepairTimeDelta, `mttr_rolling`/`mtbf`/`workload_balance` metrics, 3-mode technician-count scenario sampler + `episodes_per_scenario`, knowledge-sensitivity retune 0.002→0.15 (old default was a bug), new tests.
- Paper submodule work also uncommitted: Grillo realignment, §5, bib repairs (`grillo2022hrap40`→`grilloHumanResourceAllocation2022`, added `kuhn1955hungarian`/`bouajaja2017survey`, tien `collaborator`→`author`), `bm` package, `\graphicspath`, duplicate-label fixes.
