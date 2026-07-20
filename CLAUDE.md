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
| Table 1 common block (2026-07-15, strict-Grillo rework): objectives `PROF COST EXE RAT PERM QUAL` (classic, Grillo order) / `BAL WELL` (4.0) / `TARDY AVAIL FLEX RISK MOD` (ours); constraints `TPP PPT RES` (classic) / `DISR REC` (4.0); Scope + Method "ours" | User: use strictly Grillo's classic categories; legend descriptions are shortened Grillo quotes |
| `PROF`/`PERM` columns kept **empty** (0 marks); `ELIG` dropped; `TPP`/`PPT` marked for the 32 works with an explicit assignment decision (11 model/predict-only rows unmarked — list in audit trail) | Empty columns = a finding (drift from classic objectives); ELIG duplicated Grillo's qualified variants inside the cardinality families; Grillo §3.2: TPP/PPT/RES "mandatory" for an HRAP model. Grand total 385→434, This-work row 17→19 |
| Row corrections applied conservatively (43 adds, 11 removals; felberbauer B→A, goncalves C→A) | Only abstract-verifiable changes; judgment calls flagged in `analysis/taxonomy_row_verification.md` |
| Table 1 / §2 restructure (2026-07-20): Scope block → **Context**, trimmed to the 3 descriptors the method uses (`HET STO HIST`); Industrial-dynamics + Broader-literature blocks **merged** into one `B. Broader literature` (17 rows); This-work row `D`→`C`; §2 split into 2.1 Common Parameters + 2.2 Human Factors, with a corpus-selection paragraph (43 works, 2 criteria = blocks A/B) | User request; grand total 434→405; `\resizebox` 0.88\textwidth to keep the float one page |
| §5 documents the MDP **implementation** (state schema, event loop, action semantics, reward table `tab:reward_components`); §6 Method covers the learned encoders | §5 written from actual code (`env.py::_build_token_stream/_set_obs/_action_mask/_reward_for_assignment`) — keep in sync if obs schema or reward stack changes |
| §5 framing: event-driven POMDP with semi-MDP remark; γ discounts decision steps, not sim time | Matches implementation; soften if reviewers balk |
| Reward regimes (production-only vs human-centric subsets/coefficients) deferred to §7 Setup | Env capability vs experiment configuration split |

## Blockers / Warnings

- `zhangGraphNeuralNetworks2022` **citation mismatch**: bib entry is a GNN-book chapter, row is coded as crowdsourcing assignment paper — needs the intended reference.
- ~~Placeholder cites~~ RESOLVED 2026-07-07: `ref`→bouajaja2017survey, `Marcelo,ref2`→ruizrodriguez2022a+joo+calzavara, `MarceloESWA`→ruiz-rodriguez_dynamic_2024, `Chica2017`→chica2017simheuristics (new entry). **Author check**: intended self-cites for `Marcelo/ref2`? (`bollingKnowledgeawareLearningfocusedMultiObjective` exists in bib but lacks year/venue.)
- ~~Legacy ECAI tail~~ RESOLVED 2026-07-07: `refcheck.sty` defines `\comment`, so `\begin{comment}` blocks were silently RENDERING (6pp of legacy in every PDF + the 5 build errors). Both legacy blocks **deleted** (recoverable from git, commit 50c180d). Build now 0 errors / 0 undefined / 0 multiply-defined, 27pp.
- `refcheck` package prints label names in margins — drop before submission (also the \comment definition source).
- ~~Masked-heuristics re-run~~ DONE 2026-07-15 (all 4 scenarios) and §6 fully refreshed. Prior headline: HC-RL is the only agent to beat informed heuristics (industrial rank 1.83). Co-author merge dfb4f08 integrated (HTT-RL stays contribution #2).
- **HEADLINE THREAT (2026-07-20)** — 5 strong baselines added (see memory `benchmark-results-hvp`). At industrial 100k, **greedy_reward (2217) and TOPSIS (2202) BEAT HC-v1 (2082)** on throughput+MTTR. `greedy_reward` = myopic argmax of HC-RL's OWN reward → "it's the reward, not the RL" at 100k, so the §6 claim "HC-RL is the only agent to beat informed heuristics" is now **FALSE at 100k** and needs revising. Caveat: greedy_reward/TOPSIS/SPT/Hungarian are *informed* (use env ground-truth repair/reward estimates). The 5M eval was **killed 2026-07-20 pm** (it measured bug-trained checkpoints — see GAE bullet; HC-v1's completed 5M episode survives only in `reports/hvp_eval_v2/run.log`: 81,516 finished / MTTR 63.6); the decisive test is now the post-fix retrain + 5M ladder re-run.
- **GAE off-by-one FOUND + FIXED (2026-07-20 pm)** — `ppo_set_transformer.py::_compute_gae` consumed `dones[t+1]` instead of `dones[t]`: cross-episode advantage/value-target leakage whenever a rollout buffer held interior episode boundaries (vec training = HC-v2 only), and terminal credit severed at n−2 in **all** trainings — with `terminal_finished_products`/`terminal_fleet_knowledge` ENABLED in `baseline.json`, the terminal bonuses were effectively inert beyond the final action for HC-v1/PO too. Fixed + `tests/test_gae.py` (5 tests). The 100k headline threat may be partly this bug's artifact. **Retrain running** since 2026-07-20 12:53Z on serval: `checkpoints/hc_v2_gaefix` = GAE fix + PopArt, no GRU, γ0.997/λ0.97, vec6, HC-v1's exact reward (`baseline.json`) — a clean infra ablation; log `reports/train_hc_v2_gaefix.log` (marker `DONE_TRAIN_GAEFIX`), wandb `tfvgwmpo`, ETA ~3–4 d.
- **HC-v2 is NOT a valid negative ablation** (2026-07-20 pm review): its regression (industrial quality 0.32 vs HC-v1 0.69) is confounded by (a) the GAE bug fully expressed in vec training, (b) a reward change (`baseline_crit.json` adds criticality+downtime), (c) zero-BPTT GRU + an inline-eval RNN-state clobber (`runner.py::_inline_eval` advances training worker 0's hidden state — still unfixed, moot for non-GRU runs). Do **not** write it into §7.4 as evidence against the improvements; keep HC-v1 as the paper agent meanwhile.
- **§7 prose contradicts its own tables** (2026-07-20 review; fixes deferred at user request): (1) l.1784 "best value of *every* directional KPI" — HC is 6th/7 on workload balance (its own rank 1.83 proves it); (2) l.~1936 "best MTTR of the entire benchmark (61.6)" — LF 56.8 / SQ 59.7 in the same table; (3) l.~1938 "highest availability of all agents in every window" — RR 0.852 > HC 0.848. Should-fix list: baseline-MTTR exception to the "all scales" claim, "advantage grows" (it shrinks 34%→23%), abstract "fatigue-related absences" (metric counts all disruptions) and "several multiples" (it's 1.8×), LF/SQ 0.01 MTTR overclaim, "time-averaged" captions (per-decision means), `IllTechnicianCount` docstring (sums events, not techs). Everything else — all 8 tables + ~40 inline numbers — verified exact vs the CSVs.
- Notation audit **mostly applied 2026-07-07**: Acronyms block rebuilt (15 real acronyms), `t^c_base`→`c_time(τ)`, `c^⋆`→`c^*`, `m_F`+wrong formula→`m_f` (lin/exp), `m_k(k_i,τ)`→`m_k(τ,i)`, `φ:𝒯→[0,S]²` collision removed (φ prose + grid shape G), disruption-cell comma typo, ρ_i added (lr renamed — "diminishing-returns", lower ρ = faster learner). **Still open**: λ_c (reward coeff) vs λ_i (Jaber), β triple-use (Weibull/disruptions), `s` saturation scale vs state in §5 reward caption, `Q_t` vs `\mathcal{Q}(t)`.
- Full-text double-checks pending on Table 1: ali TEMP, alvarez SIM/ML, caricato MKS/CMP/TEMP, suer MKS, mcdonald FORG, staruch CAP; henao is a retail case study kept in block A.

## Next steps

1. **Wait for `hc_v2_gaefix`** (wandb `tfvgwmpo`, ~3–4 d) → benchmark all scales with the full 14-heuristic ladder (`--agents ... --merge`; †/‡ oracle markers now emitted by `analyze_hvp_results.py`) → **re-run the 5M ladder** with all agents + baselines. Decisive for the thesis: does restoring terminal credit let HC-RL beat greedy_reward/TOPSIS?
2. **Fix §6/§7 false claims + abstract** (list in Blockers) and wire the generated `table_footnote.tex` († informed / ‡ reward-oracle) under the result tables.
3. **§7.4 RL ablation (blank)**: train `PPOTransformerAgent`/`RainbowDQNAgent` on the HC reward as RL anchors (repo already has them); `chen2024` ADP/CFA (cost-function approx + implicit cross-training) is the one distinct literature RL method worth porting. If retrying the GRU axis: fix the `_inline_eval` RNN clobber first and add real TBPTT.
4. Notation fix package (λ_c vs λ_i, β triple-use, `s` scale vs state, Q_t vs 𝒬(t)); write §3 formalization body.

## Session notes (2026-07-20, part 2 — review, GAE fix, retrain)

- **Full audit of prior (Opus-session) work** at user request: benchmark numbers re-pulled from serval CSVs (exact), Table 1 recounted cell-by-cell incl. parent-diff invariant (only the 9 dropped columns changed; 405 ✓), §7's 8 tables byte-identical to generated artifacts + ~40 inline numbers recomputed. Real defects found: the GAE bug, 3 false §7 sentences, the `_inline_eval` RNN clobber, HC-v2 reward confound (all in Blockers). Reward-probe/baseline/PopArt/GRU-eval/vec-seeding code verified correct.
- **4 more baselines** (same session, committed): `EmpiricalSPTAgent`/`EmpiricalTopsisAgent` — *honest-information* twins learning per-(tech, failure-key) means from the new env `repair_log()` (dispatcher now passes `tech` to `_on_repair_completed`, 3-arg backward-compat; `failure_key()` = `"mtype:component_type"`); `BatchMILPAgent` — batch assignment under balance cap ceil(K/n_avail), solved exactly via Hungarian on capacity-replicated columns (transportation reduction), 4×n_techs window; `ReserveSpecialistAgent` — weakest tech with ETA ≤ τ·best (τ=1.5; τ=1 ≡ SPT). Eval keys `empirical_spt, empirical_topsis, batch_milp, reserve_specialist`. `analyze_hvp_results.py` labels carry † (informed) / ‡ (reward oracle), unmarked = honest; per-scenario `table_footnote.tex` emitted. `tests/test_baselines.py` → 15; suite 454. Real-env smoke: 228 repairs logged/episode, estimator learned 46 pairs.
- **§2 polish** (paper submodule): title "+, and Methods" (Method block is inside the Common-parameters span), mapping-¶ reordered (adoption → findings → conventions → extensions → post-hoc note), §2.2 opener rewritten, 6 cluster-1 citations added, `~\cite` fix, stray tabular blank line removed. Build 0 err / 29pp. Red `\textcolor` markers left for author sign-off.
- **Ops**: serval is shared (ayoub's CPU training runs there — never broad-pkill python); `pkill -f`/`pgrep -f` patterns self-match the ssh remote shell and kill the session — use `[x]` bracket trick; a WireGuard drop presents as ssh timeout to 10.6.36.22.

## Session notes (2026-07-20)

- **HC-v2 benchmark + 5 new baselines** (serval-paris = paris-snt-unit, RTX 4080, ssh alias; out-root `reports/hvp_eval_v2/`, NOT v1 `reports/hvp_eval/`). Real HC-v2 ckpt was on serval `checkpoints/hc_v2/set_transformer_best.pt` (GRU+PopArt); local `set_transformer_best.pt` scp was a stale byte-dup of the PO ckpt. `scripts/eval_human_vs_performance.py` now **architecture-aware** (reads each ckpt's `improvements` dict → builds matching GRU/PopArt net). Industrial 100k (25k-cap re-run — default 10k cap TRUNCATES) finished: greedy_reward 2217 > topsis 2202 > **HC-v1 2082** > Hungarian 2069 > SPT 2049 > shortest_queue 2025 > … > hc_v2 1894 > PO 1863 > … > train_weakest 1614. See memory `benchmark-results-hvp`.
- **5 new baselines** (in `src/agents/baselines/heuristics.py`; parent commits 350b45c + bcefecb, now on `main` as ancestors of paper push fccb321): `ShortestProcessingTimeAgent` (SPT/skill-greedy), `OptimalAssignmentAgent` (Hungarian, scipy `linear_sum_assignment`), `TopsisAgent` (multi-criteria, Ferjani2017), `GreedyRewardAgent` (myopic argmax of HC reward), `TrainWeakestAgent` (naive upskill). Run: `--agents shortest_processing,optimal_assignment,topsis,greedy_reward,train_weakest --merge`.
- **Env decision-support API** (`src/kata/env.py`, public, current-ticket-relative): `expected_repair_times()`, `skill_match_scores()`, `available_mask()`, `assignment_counts()`, `assignment_cost_matrix()`, side-effect-free `assignment_reward_estimates()` (snapshots/restores `_prev_finished_products`/`_prev_fleet_knowledge`/normaliser freeze). New structured-obs field `technician_expected_repair` (flag `include_repair_estimate_in_observation`, default on); `Agent.attach_env()` gives baselines the handle (eval calls it). `tests/test_baselines.py` (10 tests incl. greedy-trap + reward-probe idempotency); full suite **444 pass**.
- **§2 + Table 1 restructure** (paper 113ac5f → parent fccb321): see Key decisions row. Build 0 err / 29pp.

## Session notes (2026-07-07, part 2)

- Manuscript builds **0 errors / 0 warnings** (28pp). §4 reworked (agent-vs-dispatcher fix, design-properties paragraph, rich figure caption); Appendix A: six-entity fix, φ hash-encoder explanation, ρ semantics + archetype-axis rationale + specialists documented, disruption trigger taxonomy + fatigue→exhaustion loop, defaults (f=0.3, α=0.15, Δ_travel=15), reward-stack paragraph replaced by pointer to §5.4, sampler paragraph updated to 3-mode + episodes_per_scenario, Bernoulli per-cycle clarification, eval-protocol cross-ref. §6.1 now states unit coefficients + normalization.

## Session notes (2026-07-06)

- Code side (uncommitted): signed RepairTimeDelta, `mttr_rolling`/`mtbf`/`workload_balance` metrics, 3-mode technician-count scenario sampler + `episodes_per_scenario`, knowledge-sensitivity retune 0.002→0.15 (old default was a bug), new tests.
- Paper submodule work also uncommitted: Grillo realignment, §5, bib repairs (`grillo2022hrap40`→`grilloHumanResourceAllocation2022`, added `kuhn1955hungarian`/`bouajaja2017survey`, tien `collaborator`→`author`), `bm` package, `\graphicspath`, duplicate-label fixes.
