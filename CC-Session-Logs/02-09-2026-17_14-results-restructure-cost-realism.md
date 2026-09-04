# Session Log: 02-09-2026 17:14 - results-restructure-cost-realism

## Quick Reference (for AI scanning)
**Confidence keywords:** manuscript, results-section, restructure, ablation-first, HTT-RL, PO-HTT-RL, ft_quality, summary-table pivot, distance-to-best, worked example, guard script, results-only, figures, rolling MTTR, fleet knowledge, disruptions per output, calendar axes, dagger asterisk, multirow baselines table, deep dive, band semantics, window std, machine band, technician band, cost analysis, Rodriguez RCIM 2022, Eq 5, profit maps, p_max, wear-driven breakdowns, self-throttling, realistic lifespan, machine templates, MTBF, load-to-capacity, contention, dgy, GPU2, disruption metrics, per-type disruption time, lifecycle deep-dive, artifacts, knowledge bands, hvp_eval_v6w, hvp_eval_disr
**Projects:** knowledge-aware-technician-assignment (KATA/FactoReal), paper submodule JMS_Adrien_RL
**Outcome:** §Results restructured end-to-end under a Results-only guard (ablation-first, pivoted verdict table, per-scenario 2×2 figures, PO-HTT-RL ablation subsection, deep-dive placeholder); per-type disruption instrumentation built and a 14-agent re-generation launched on dgy; cost-analysis direction settled (corrected Rodríguez Eq. 5); realistic-lifespan sweep located the contention frontier where human-centric dispatching pays.

## Decisions Made
- **Results-only edit guard** (`scripts/check_paper_results_only.sh`, run before every paper commit; `--bypass`/`GUARD_BYPASS=1` only under explicit authorization). Artifact exemption narrowed to `Manuscript.*` build products — figure PDFs are content.
- **Section order**: 6.2 Objectives Ablation *selects* HTT-RL^quality → carried into 6.3 Performance; 6.4 Deep Dive (placeholder for now, user request); 6.5 Ablation Study groups {Human-centred vs performance-only, Architecture, Information} as subsubsections.
- **Naming**: fine-tunes = HTT-RL\textsuperscript{lever}; the performance-only *full retraining* = **PO-HTT-RL** (superscript wrongly implied fine-tune); informed marker † → `*` manuscript-wide + analyzer labels; ‡ unchanged; main comparison drops `-last` twins (appendix keeps them).
- **Verdict table pivoted**: scenarios as rows (S1–S5 + Overall), 14 representative deployable columns, worked-example preamble (HTT-RL industrial: 2.8/4.4/0.8/5.6% gaps → score 3.4); per-KPI frontier still over the full 25-agent field.
- **Figures**: per-scenario 2×2 (rolling MTTR 50-window median-smoothed ~3% horizon; fleet knowledge; disruptions/10³ products bars = the score's metric; cumulative products), Okabe-Ito accents + gray context, calendar top axes (1 t.u. ≈ 1 min), old results_over_time/results_lifecycle figures removed.
- **Deep-dive band semantics** (user): MTTR + rolling repair-quality bands = std *within* the rolling window; throughput → per-machine processing rate, band = std *across machines*; knowledge/fatigue bands = std *across technicians*.
- **Cost analysis**: adopt Rodríguez RCIM'22 Eq. (5) *with the working-state term corrected to P/P_max* — raw `a`=availability inverts in a wear-driven world. Alternative frameworks (dominance/implied-exchange-rates/minimax-regret; time-denominated prices via timesteps-per-product anchor) designed and discussed, on hold.
- **Table 5**: archetype multirow grouping (Multicriteria/Rule-based/Optimization/Reward oracle/Learning-based) + corpus-only citation column ("---" where no Table-1 antecedent; RL anchors' antecedents are Table-2 works).

## Key Learnings
- **Breakdowns/product ≈ 9–11 for every agent** — failures are use-driven, so per-event cost penalties tax production; also why the raw Eq. (5) plane is won by line-stallers (LeastFatigued 99% of industrial cells).
- **The disruption count decomposes** as ~675 policy-independent floor (300 Poisson injuries + 375 vacations at industrial) + exhaustion ∝ mean fatigue (expected = 0.001·fatigue·T per tech; poll interval cancels); validated ±2–9% vs observed for 5 agents.
- **Emp-Topsis's published industrial mean (1,964) hides a collapsed episode** (1,417 vs 2,227/2,247 — byte-identical in published data); learned agents' worst episodes 2,111/2,193.
- **Across-technician knowledge std ≈ 0.9× the mean for every policy** (CV 0.91–0.97) — key-person concentration made visible; policies differ in means, not dispersion shape.
- **Realistic-lifespan worlds self-throttle**: fewer techs → longer machine downtime → less usage → fewer wear failures; queues never form even at 3 techs/90 machines at ×25 lifespans. Contention (and the human-centric advantage, +7.7% over RR) returns at ×10 lifespans @ 3 techs. The switch is load-to-capacity, not failure density.
- **`queue_size` step metric is sampled at decision instants** → reads 0 even at availability 0.58; use availability/MTTR as contention evidence.
- Density rescaling (MTBF ×k → b,c ÷k) leaves cost-map region structure invariant; only per-event axis labels move.

## Solutions & Fixes
- Guard test-tamper cycle proved the span check; `.pdf` exemption hole found via the figures commit and closed.
- elsarticle floats execute their body twice → `\providecommand` (not `\newcommand`) inside floats; rotated headers abandoned for resizebox + abbreviations.
- `medfilt` zero-padding dragged curve edges to 0 → pandas centered rolling median+mean.
- Lifecycle MTTR y-floor bug: retirement `axvline`s are 2-point (0,1) Line2D in `ax.get_lines()` → filter `len(ydata)>10` before computing limits.
- "Appendix Appendix B" doubles: `\ref{app:oracle}` already renders "Appendix B".
- Machine-rate column empty: `_factory_machines()` lives on env, not dispatcher.
- `Ref1`/`2` undefined citations + `sec:FactoReal` renames are the co-author's red placeholders (theirs to fix).
- dgy ssh + zellij launch pattern, byte-offset remote log monitors, `--extra-machine-templates` runtime registration via `register_template()`.

## Files Modified
- `paper/Manuscript.tex`: full §Results pass (see 0MS in CLAUDE.md); commits 9cb4606(merge)→eb1a610→1ae63a3→04c8342→33fbac7→545ea76→9e9634e→def85af→e93fb83→d1c1b70→cb2f64a→d8441e3→6a0593b→650069d→6b10720→e9eaf4b→afb195a→7b1f8dd.
- `paper/figures/`: mttr_knowledge_{small_scale,baseline,massive_scale,very_long,lifecycle}.pdf (2×2, calendar axes), deepdive_industrial.pdf.
- `scripts/check_paper_results_only.sh` (guard), `scripts/make_scenario_figures.py`, `scripts/make_deepdive_figure.py`, `scripts/dgy_disr_benchmarks.sh`, `scripts/dgy_deepdive_lifecycle.sh`.
- `scripts/eval_human_vs_performance.py`: per-type disruption records, `mttr_rolling_std`, `machine_rate_mean/std` (id-keyed 5k-t.u. window), `fleet_knowledge_std`, `--extra-machine-templates`, scenarios `very_long_realistic{,_5t,_3t}`, `very_long_realistic10_3t`.
- `src/kata/entities/technicians/GymTechnician.py` (`disruption_time_by_type` elapsed-hold), `src/kata/metrics.py` (6 per-type episode metrics), `tests/test_disruption_metrics.py` (3 tests).
- `run_configs/realistic_lifespan/`: `machine_templates_realistic{,10}.json` (`_rl`/`_rl10`, ×25/×10 lifespans), `very_long_realistic*.json`, README.
- `scripts/analyze_hvp_results.py` (labels †→\*), `CLAUDE.md` (preserved via /preserve, commit 1cfbe07).

## Pending Tasks
- **dgy disr generation** (zellij `disr`, marker `DISR BENCH QUEUE DONE`): 14 agents × 5 scenarios with per-type disruption metrics → `reports/hvp_eval_disr` (NEVER merge into v6w). On DONE: per-type analysis; enables cost analysis with absence times + §6.3 bar decomposition by type.
- **dgy lifecycle deep-dive** (zellij `ddlc`, marker `DEEPDIVE LC DONE`, GPU 2): 4 agents, extended records → `reports/deepdive_lc_parts/`.
- **§6.4 Deep Dive**: red placeholder — write in a future pass (pipeline + industrial data ready; lifecycle data incoming).
- Cost analysis: corrected Eq. (5) maps exist as artifact a087e631; not in manuscript. Realism sweep → §8 scoping paragraph (outside guard; needs authorization).
- Stopped local queues `hcext`/`hcwr` still await manual resume (CLAUDE.md 0EXT-c). Abstract draft ready with PO slot filled (chat, 2026-08-27 turn). §7 nondeterminism softening pending user go.

## Errors & Workarounds
- Repeated zsh/zoxide cwd traps (`cd paper` from wrong dir silently jumping or failing) — mitigated with absolute paths / `git -C`; several commit commands re-run from repo root.
- AskUserQuestion rejects single-option questions (custom-note step skipped as a result).
- Rodríguez paper fetched via ORBilu bitstream PDF (ScienceDirect 403).

## Custom Notes
None

---

## Quick Resume Context
§Results is fully restructured and pushed (paper 7b1f8dd; deep-dive is a placeholder awaiting a writing pass). Two dgy queues run under markers (`DISR BENCH QUEUE DONE`, `DEEPDIVE LC DONE`); the realistic-lifespan sweep (isolated in run_configs/realistic_lifespan/) showed dispatching pays only under load-to-capacity contention — the ×10@3-techs point where HTT-RL^quality leads by 7.7%. Cost analysis direction: Rodríguez Eq. (5) with P/P_max numerator (raw form inverts; maps in artifact a087e631). CLAUDE.md Next steps 0MS/0RL/0DDLC/0DISR carry the durable state.

---

## Raw Session Log

Pre-compaction history lives in the full transcript:
`~/.claude/projects/-home-gourmet-repositories-knowledge-aware-technician-assignment/b62f1dcb-80ec-41db-ab42-c1454e915e9c.jsonl`

Chronological exchange record (user messages verbatim; assistant outcomes condensed):

1. **User:** "Start some experimetns on the local machine, I want yo uto train an agent identical to the one that performs best (last version of HTT RL), but without any human centric objectives... after training, run benchmarks" → po_v6 twin trained locally (5h40), benchmarked; lifecycle collapse below Random; eval nondeterminism found via identical-weights pair.
2. **User:** abstract draft request (realism + human-centric story) → draft delivered in chat with PO slot.
3. **User:** "Give me a full report on how these experiments did" → artifact 9e0ee22e (po_v6 report).
4. **User:** "take the best weights of HTT... train them for 600 more, with the normal set of rewards" → hc_v6_ext (found+fixed LR re-arm same-size-extension defect; hc_v6_wr relaunch); later both queues manually stopped on request.
5. **User:** "safely shut off the training, we'll resume it manually tomorrow" → clean shutdown + resume instructions in CLAUDE.md.
6. **User (2026-09-02):** "Let's go back to the Manuscript. First pull the remote changes... don't touch anything outside the Results section (write a small script to verify this...) Move swap Objectives ablation and performance... Rename all ft-... as HTT-RL with the finetuning as exponent... Pivot the table... Provide a small example..." → guard + full restructure (eb1a610).
7. **User:** "Remove the HTT-RL last, keep only HTT-RL" → done (1ae63a3).
8. **User:** "For every scenario, make two figures... rolling mttr... fleet knowledge" (+ later: keep 50-window; smooth; add fatigue + cumulative products; corrected fatigue-per-products; remove figures 7 and 8; rescale lifecycle MTTR y) → per-scenario 2×2 figure set, several iterations.
9. **User:** "Does the number of disruptions not seem very high?" → floor+exhaustion decomposition validated ±2–9%.
10. **User:** "Connect to the uni vpn, check GPUs on dgy, relaunch benchmarks there tracking disruption events by type + cumulated disruption time" → metrics built, disr generation launched (zellij `disr`).
11. **User (mid-turn):** "Create a 'Human-centered vs Performance-focused' ablation before the Architecture Ablation" → §PO-HTT-RL subsection + tab:results_po (cb2f64a); renamed from HTT-RL^perf on request (d8441e3).
12. **User:** "Reorganize 6.4, 6.5, 6.6 as subsubsections under an Ablation study subsection" → done (6a0593b).
13. **User:** "Before the abstraction subsection, create a Deep-Dive subsection..." (4 agents, MTTR/knowledge/throughput/fatigue with std bands) → built with local re-evals; bands reworked on follow-ups (window-std MTTR, machine-band rate, rolling quality panel; afb195a); knowledge bands artifact 0f126ee9.
14. **User:** "In the table presenting the baselines (table 5), add a multirow first column... simplify the last column..." → done (6b10720).
15. **User:** "Across every plot... add 'real temporal' markers" + "Remove the weird symbol... replace it with a simple *" → calendar axes + †→\* (650069d).
16. **User:** "Start the ddep dive benchmark on the lifecycle env... Use GPU2 of DGY..." → ddlc queue launched.
17. **User:** cost-analysis thread — "taking example on... Rodriguez?", "only monetary costs...", "product takes x timesteps... proportion of the profit", "not satisfied... our own way", "Let's try the raw formula of Marcelo", "Explain your correction in detail", "Doesn't this mean that the machines are actually breaking too much...?" → framework designs, Eq. (5) maps artifact a087e631, correction explained, stress-regime analysis.
18. **User:** realistic-lifespan probes — 30/5/3 techs at ×25, then ×10@3 → contention frontier found (+7.7%).
19. **User:** "For now replace the deep dive subsection with a placeholder... commit and push" → done (7b1f8dd).
20. **User:** `/preserve` → CLAUDE.md updated (1cfbe07, 226 lines). `/compress` → this log.
