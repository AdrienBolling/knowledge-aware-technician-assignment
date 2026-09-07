# Session Log: 06-09-2026 18:07 - financial-profit-ratio-maps

## Quick Reference (for AI scanning)
**Confidence keywords:** financial implications, profit ratio, Rodríguez Eq.(5), RCIM 2022, top-profit map, heatmap, break-even, profit border, production uptime (dropped), material cost, human cost, disruptions, maintenance interventions, finished products, §6.6 sec:results:financial, fig:financial, financial_analysis.py, dgy_financial_queue.sh, hvp_eval_v6w, hvp_eval_disr, PYTHONHASHSEED, machine-id hash collision, Machine.productive_time, Evo-TOPSIS weights, serval-dgy, zellij, LaTeX subfigures, artifact
**Projects:** knowledge-aware-technician-assignment (KATA / FactoReal); `paper/` submodule (Manuscript.tex); remote DGX `serval-dgy`
**Outcome:** A "Financial Implications" subsection (§6.6) was added to the Results, built on a profit ratio R = P − (h·D + m·M) computed from existing benchmark counts, with five LaTeX-subfigure top-profit maps carrying a dotted profit border; the human-centric agent holds the entire profitable region at all three realistic scenarios. Nothing committed.

## Decisions Made
- **Formula pivoted mid-session (user).** First design re-instantiated Rodríguez et al. (RCIM 2022) Eq.(5) with a *production-uptime* factor (mean over the last route type's machines of processing time / active life) and costs split Material + Human-resource. User dropped it: *"we don't really care about the time spent in production of the factory either, we don't need to measure it."* Final: `profit = π_p·P − c_m·M − c_h·D` → **`R = P − (h·D + m·M)`**, `h = c_h/π_p`, `m = c_m/π_p`.
- **No re-evaluation needed for the final formula.** P, M, D are `finished_products`, `total_repairs`, `ill_technician_count` — already in `reports/hvp_eval_v6w`. The 170-part dgy re-evaluation queue for the uptime design was killed (0 parts completed).
- **Uptime instrumentation kept, not reverted (user).** `Machine.productive_time`, `created_at`/`retired_at`, harness `_financial_kpis`/`_register_machines`, `scripts/dgy_financial_queue.sh` stay in the tree, flagged as unused by the paper — exact per-machine productive time is now available to a future analysis (deep dive, bottleneck read).
- **Figure form (user).** LaTeX subfigures from standalone panel PDFs (not a matplotlib multi-panel figure); one map per scenario, all five scenarios, dotted line for the profit border.
- **Roster shown = the deployable field** of `tab:results_dist`; the PO twin and the informed baselines are discussed in prose as variants (`--roster twin|all`), not plotted.
- **Panel colours**: scenario-panel accents + Dark2 hues for the rules that actually win somewhere (the original all-gray rule palette was unreadable at S1/S2); loss region = winner colour washed 70% toward white (Rodríguez's light-colour convention).
- **Earlier, superseded decision (recorded for context)**: in the uptime design, β priced technician time on the tool and α priced interventions, because pricing both per count would collapse the plane onto α+β. The final formula deliberately does price both per count — the two counts differ enough (M/P vs D/P) that the plane does not degenerate.

## Key Learnings
- **The maps are governed by slopes, not counts.** Both cost drivers scale with activity (breakdowns are use-driven, so M grows with P; absences grow with fatigue), so a policy wins a region when its output per unit of cost is highest: M/P (9–18 interventions per product; 17.8 at lifecycle) and D/P draw every frontier.
- **Policy-independent disruptions move the border, never the winner.** Injuries + vacations are 620–660 of the 1,000–1,600 industrial events for every policy, charging each the same `h·D₀`. The exhaustion-only variant roughly doubles the h borders and changes no leader.
- **Verdict (deployable roster):** S1 Emp-Topsis leads at the origin but LeastBusy holds 58% of the gain region (takeover at m 0.74 / h 1.86); S2 Emp-Topsis leads, HTT-RL overtakes at m 0.195 (border 0.455) and BatchMilp at h 0.98 (3.97), BatchMilp holding 65%; **S3–S5 HTT-RL qua. holds 100% of the gain region** (S3's only challenger RoundRobin needs m ≥ 0.106 against a 0.113 border — within 6% of where the plant itself stops being profitable; S4/S5 no challenger at any profitable price).
- **Leader break-evens:** S3 h 1.88 / m 0.113, S4 1.91 / 0.116, S5 0.72 / 0.056. The PO twin changes no cell at S3–S5. Full field: GreedyReward* takes S3 at m ≥ 0.025, leads S5 at the origin by 0.2% of output, S4 stays HTT-RL qua. until h 1.57.
- **Absolute thresholds are world-specific.** The benchmark plant's failure density is far above an industrial plant's; what transfers is the structure — the more realistic the scenario, the larger the share of the price plane where the human-centric dispatcher is the profitable choice.
- The old uptime quantity was ≈1.1× the env's per-decision sampled estimate (exact vs sampled), and final-stage processed count equals finished products exactly — a useful invariant if that instrumentation is ever used.

## Solutions & Fixes
- **`scripts/financial_analysis.py` (new, rewritten for the final formula).** Loads v6w episode means, applies `reports/hvp_eval_disr` per-type shares for the exhaustion-only variant, sweeps a 200×200 (h, m) grid to 15% past the roster's widest break-even, writes cell-winner CSVs, gain-region shares, break-even frontiers, a LaTeX row table, and the panel PDFs.
  ```bash
  uv run python scripts/financial_analysis.py                       # deployable roster, all disruptions
  uv run python scripts/financial_analysis.py --roster all --no-figures --out /tmp/fin_all
  uv run python scripts/financial_analysis.py --disruptions exhaustion --no-figures --out /tmp/fin_exh
  ```
- **First-challenger logic**: a takeover only counts if the challenger is still profitable at that price (`x <= P_other / q_other`), otherwise the "challenger" only wins inside the loss region and is not a deployment choice.
- **Dotted profit border**: `ax.contour(H, M, best, levels=[0.0], linestyles="dotted")`. A tuple dash pattern `(0, (1.2, 1.6))` passed as `linestyles` raises `ValueError: Do not know how to convert [0] to dashes` in this matplotlib — pass the named style and set dashes on the collection afterwards.
- **Panels for LaTeX subfigures**: fixed physical size 2.1×1.9 in, no tight-bbox cropping, included at `0.32\textwidth` — the same pattern as `scripts/make_scenario_figures.py`, so every panel scales identically.
- **Artifact** (interactive, same data and formula): https://claude.ai/code/artifact/6919b59b-c141-4b47-9b66-284fcf120ca4 — profit-ratio heatmap of a chosen policy, advantage over the best alternative, top-policy map, break-even cards, data table, roster and disruption toggles.

## Files Modified
- `paper/Manuscript.tex`: **§6.6 `sec:results:financial`** inserted before Discussion and then rewritten for the final formula — motivation, `eq:fin:profit`, `eq:fin:ratio`, map construction, results paragraph, borders paragraph; `fig:financial` = five subfigure maps + legend cell at `0.32\textwidth`. Build 0 errors / 32 pp; results-only guard passes without bypass.
- `scripts/financial_analysis.py`: **new** — the analysis and figure pipeline (see above).
- `paper/figures/panels/profit_{small_scale,baseline,massive_scale,very_long,lifecycle}.pdf` + `profit_legend.pdf`: **new** panels.
- `financial/` (repo root, new): `agent_scenario.csv` (P, M, D, slopes, break-evens per agent × scenario), `profit_maps/<scenario>.csv` (per-cell winner and best R), `map_summary.csv`, `frontier_summary.csv`, `frontier_table.tex`.
- `src/kata/entities/machines/machine.py`: `productive_time` accumulated at the production-loop yields (interrupt-safe), `created_at`, `retired_at`. **Kept, unused by the paper.**
- `src/kata/scenario.py`: `retire_machine_from_factory` stamps `retired_at`. **Kept, unused.**
- `scripts/eval_human_vs_performance.py`: `_financial_kpis`, `_register_machines`, `_product_route` and the new episode columns (route, final-stage uptime and processed, fleet uptime, repair labour time, downtime). **Kept, unused.**
- `scripts/dgy_financial_queue.sh`: **new**, the (now killed) 170-part re-evaluation queue; exports `PYTHONHASHSEED=0`. **Kept, unused.**
- `run_configs/agents/evo_topsis_weights.json`: **new locally** — the tuned weights pulled from dgy (were missing here).
- `CLAUDE.md`: rewritten 2026-09-06 session notes, new machine-id hash-collision blocker, rewritten `0FIN` next-step entry.

## Setup & Config
- Every `uv run` on `serval-dgy` must be `--no-sync` (deliberate torch downgrade there). Local runs need `PYTHONPATH=$PWD/src` for the scripts.
- University VPN must be up before reaching dgy: `uni-vpn up` (~10 s); without it the hostname does not resolve.
- `zellij --session X run -- ...` requires the session to exist — create it first with `zellij attach --create-background X`.
- Paper commits go through `scripts/check_paper_results_only.sh`; `--bypass` only under explicit authorization. This session's changes are all inside the Results section, so the guard passes clean.
- Data roots: `reports/hvp_eval_v6w` (canonical benchmark), `reports/hvp_eval_disr` (per-type disruption re-run, never merged into v6w), `reports/hvp_v6w_parts` (idempotent parts).

## Pending Tasks
- **Commit.** Nothing is committed. The tree carries: the §6.6 rewrite and panels, `financial/`, `scripts/financial_analysis.py`, the unused uptime instrumentation, `scripts/dgy_financial_queue.sh`, the tuned Evo-TOPSIS weights, and pulled-but-unmerged parts for `evo_topsis{,_inf}`, `hc_v6_ext{,_last}`, `hc_v6_wr{,_last}`.
- **Decide whether Evo-TOPSIS and the ext/wr rows enter the paper** — needs `merge_hvp_parts.py --scenario` per scenario into the local `reports/hvp_eval_v6w`, then `analyze_hvp_results.py` and `summary_scores.py`.
- **`--roster all` and `--disruptions exhaustion` variants** exist and are quoted in the §6.6 prose but are not plotted; decide whether either deserves a panel or an appendix table.
- **Machine-id hash collision root fix** (see below) — deferred because it would change observation content relative to the trained checkpoints.
- Carried over: §6.4 Deep Dive is still a red placeholder; per-type disruption analysis script; summary-score variant choice (`noavail_mean` recommended); §7/§8 items.

## Errors & Workarounds
- **Machine-id hash collisions (new blocker).** `ScenarioBuilder` derives `machine_id = hash(name) % 10000` under Python's per-process randomised string hashing, so `dispatcher.machines` silently loses a machine in ~2% of 20-machine and ~37% of 97-machine processes (verified: `PYTHONHASHSEED=3` collides at the baseline layout; 0, 1, 2 are collision-free for all five layouts). Dynamics are unaffected (tickets reference objects, feeders use lists); observations, per-machine stats and availability totals are biased for that process. Mitigations written but unused by the published analysis: the queue script pins the hash seed, and the harness registry keys machines by object identity from `factory_handles.machines_by_type`.
- **`pkill -f` matched the ssh remote shell itself.** Killing and relaunching the dgy queue in one ssh command silently no-ops, twice, because the pattern appears in the remote shell's own command line. Do the kill and the relaunch in separate ssh sessions.
- **matplotlib `linestyles` tuple** rejected in `contour` (see Solutions).
- **Local dev dataset was not canonical**: 30-technician learned rows on the local RTX 4080 differ from the DGX v6w rows within the known ±3% GPU-nondeterminism band, so between-family orderings at S3/S4 could not be read from it.

## Key Exchanges
- User asked for the alternative Rodríguez formulation with a production-uptime factor and a material/human cost split, CSVs at the repo root, computation on dgy if available, and a new Results subsection. Delivered: instrumentation, harness KPIs, a 170-part dgy queue, the analysis pipeline, and a §6.6 draft with a red placeholder for the pending data.
- User then dropped the uptime factor entirely and gave the simpler three-count formula, asking for a heatmap artifact. The queue was killed, the artifact built from existing benchmark data.
- User asked for the paper section to be updated, with the most-profitable-policy figures for all five scenarios as LaTeX subfigures and a dotted profit border. Delivered and built clean.

## Custom Notes
None.

---

## Quick Resume Context
The Results section now ends with §6.6 "Financial Implications": a currency-free profit ratio `R = P − (h·D + m·M)` computed from the benchmark's own per-episode counts, swept over the (h, m) price plane and drawn as five LaTeX-subfigure top-profit maps with a dotted profit border; the headline is that HTT-RL qua. is the most profitable deployable policy over the entire profitable region of all three realistic scenarios. Everything is uncommitted, and the working tree also holds unused per-machine uptime instrumentation (kept deliberately), the tuned Evo-TOPSIS weights, and pulled-but-unmerged benchmark parts for Evo-TOPSIS and the hc_v6 extension runs. The open calls are whether to commit as-is, whether Evo-TOPSIS and the extension rows enter the paper, and whether the full-field or exhaustion-only map variants deserve their own exhibit.

---

## Raw Session Log

> User messages are verbatim. Assistant turns are condensed to their substance and the tool work they carried; the session opened from a compaction summary of earlier work (abstract sentence, intro results paragraph, baseline-reward clarification).

**[context]** Session continued from a compacted conversation. Carried state: paper commit 47ad23b and code commit 5e9f3c4 pushed (abstract simulator-gap sentence, rewritten intro Main-results paragraph, contributions bullet, §6 baseline count); a pending clarification about which baselines run on the agent's reward.

**Assistant:** Reported both repos pushed and in sync, build clean at 0 errors / 30 pages, and answered the baseline-reward question: the non-rule baselines do not run on our reward. TOPSIS variants rank on their own three criteria (repair time 0.5, fatigue 0.3, workload 0.2); Hungarian and BatchMILP minimise expected repair time (MILP adding a workload cap as a constraint); GreedyReward is the only baseline consuming our reward, and it maximises the evaluation scenario's stack rather than the v5 training stack, so the Table 5 caption's "exact per-assignment reward of HTT-RL" overstates it; the three MLP anchors are trained on the identical v5 human-centric reward.

**User:** "Let's get back on the small "Financial implications" subsection of the results.
I have decided what I want you to do, we are gonna create an alternative version of the approach of Marcelo Rodriguez.
Let's take its formula, and substitue with what makes sense for us :
What is gonna be considered the uptime factor (percentage of time the factory spent being productive), we're gonna underqstand this as the mean of the ratios of time spent producing (not healthy, but in production) over all instances of the last machine of the products route. The idea being that we consider taht the factory produces value as long as these machines are producing.
Then for the costs, let's split thgem into two : instead of the cost of manufacturing action and the cost of maintenance, we're gonan, rephrase as "Material Cost" + "Human Ressource Cost". The idea being that all parts that have to be replaced, tools etc go towards material costs, and if skilled technicians have to be brought in from the machines manufacturer, additionnal outside training, overtime etc, goes to Human cossts.
I want you to :
Compute the relevant CSV to compute this formula (which is basically gonna be the production uptimes over episodes for all agents for all scenarios averaged over the relevant machines, and the number of maintenances for all agents for all scenarios)
Compute the formula and establish a 2d map similar to what Marcelo did
Add a subsection "Financial Implications" of the results section, with the explanation of the factors, the motivations of the analysis, teh explanation of the formula, and the resulting 2d visualisations
I want the csv outputed to a folder at the root of the project, name it however you like.
If you need to recompute some things, do it on DGY if its available."

**Assistant:** Recovered the earlier Rodríguez cost-map work from the previous session transcript; read the simulator's machine and routing code; added exact `productive_time` plus lifecycle timestamps to the machine class; added `_financial_kpis` and a machine registry to the eval harness; verified on baseline, industrial and a time-compressed lifecycle probe; ran the full test suite (664 passed); wrote and launched `scripts/dgy_financial_queue.sh` on dgy (170 parts, 18 lanes, GPUs 0–2); found and worked around the machine-id hash-collision bug and relaunched with `PYTHONHASHSEED=0`; built `scripts/financial_analysis.py`; drafted §6.6 with a red placeholder and a commented-out figure block; pulled the finished Evo-TOPSIS and ext/wr parts and the tuned weights from dgy. Reported the design, the two judgement calls (human-cost driver, horizon-free coefficients), the preview from a local dev run, and the hash-collision finding.

**User:** "Actually, we don't really care about the time spent in production of teh factory either, we do'nt need to measure it
All we need is : the number of products produced, the number of maintenance actions, the number of human disruptions
Let's call h and m respectively the percentage of profit of a product a maintenance costs humanly, and materialy
Then we say to the reader :
profit = profit per product * products - cost of a maintenance * number of maintenance - cost of human disruption * number of human disruptions
which we can simplify by saying
profit ratio = products - (h * number of human disruptions + m * number of maintenances)
Thus we can present a 2d heatmap, with h and m as x and y axis, and the profit ratio as the computed value.
Try this and show it to me in an artifacxt"

**Assistant:** Killed the dgy queue (0 parts completed) and its monitor; extracted P, M, D per agent and scenario from the existing v6w episodes plus the per-type disruption shares; built and published an interactive artifact with three linked maps (profit ratio of a chosen policy with dashed break-even, advantage over the best alternative, top-policy map with washed loss region), break-even cards, a data table and roster/disruption toggles. Reported the verdict: Emp-Topsis leads the toy scales but is overtaken as costs rise; HTT-RL qua. holds the entire gain region at the three realistic scenarios.

**User:** "Update the related section in the paper. I want the figures describing the most profitabel policy for all 5 scenarios. Still used the subplots from latex not matplotlib. Include a dotted line showcasing the profit border."

**Assistant:** Rewrote `scripts/financial_analysis.py` around `R = P − (h·D + m·M)`, emitting `financial/*.csv` and one standalone panel PDF per scenario plus a legend panel, with the dotted profit border as the R = 0 contour of the winning policy and the loss region washed out; recoloured the rule winners so the toy-scale maps read; rewrote §6.6 with the two equations, the figure of five LaTeX subfigures plus legend cell, a results paragraph and a borders paragraph; built clean (0 errors, 32 pages) and confirmed the results-only guard passes. Reported the verdict, the borders, and the leftover uncommitted instrumentation.

**User:** `/preserve` → chose to preserve all categories and to keep the uptime instrumentation while noting it as unused.

**Assistant:** Rewrote the 2026-09-06 session notes in CLAUDE.md for the pivoted design, corrected the hash-collision blocker's stale reference to the killed re-run, and tightened the `0FIN` next-step entry. CLAUDE.md at 243 lines, under the 280 target, no archiving needed.

**User:** `/compress` → chose all seven sections, topic `financial-profit-ratio-maps`, no extra note.

**Assistant:** Wrote this session log.
