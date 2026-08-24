# Session Log: 13-08-2026 17:50 - v6w-read-mlp-baselines

## Quick Reference (for AI scanning)
**Confidence keywords:** v6w results, hc_v6, overall rank, 5M collapse reversed, knowledge maintenance, analyzer AGENT_LABELS drop, lifecycle benchmark, zellij v6vec v6ft trad, ft_protect, ft_fatigue, ft_quality, reward fine-tune, lr re-arm, seed 4242, compare_wall_time, ms/decision, KPI curves artifact, training stability artifact, wandb datastore, entropy, KL early stop, PopArt sigma, ultracode, workflow, MLP baselines, a2c_mlp, grpo_mlp, dql_mlp, SetObsFlattener, one-hot, symlog, set-flatten, architecture ablation, SET_OBS_AGENT_TYPES, train_multiscale_v5_mlp, range-mode fleet, scenario rotation off-by-one, reset_scenario_cache, GRPO horizon confound, group z-score, Double DQN, private RNG streams, replay buffer, adversarial review, Fable verification, dgy, ssh-agent, serval-paris down
**Projects:** knowledge-aware-technician-assignment (KATA/FactoReal paper, J. Manufacturing Systems)
**Outcome:** v6w results read (hc_v6 best-learned-ever, 5M collapse reversed), lifecycle benchmark + 3 reward fine-tunes launched on dgy, wall-time comparison script shipped, and a fully reviewed traditional-baselines package (A2C/GRPO/DQL MLPs on the flattened set obs) built via 4 workflows and queued for training+benchmarking.

## Decisions Made
- **Traditional MLP baselines = set-flatten architecture ablation** (ultracode design, Fable-synthesized): all three consume the SAME set observation as HTT, flattened agent-side (`SetObsFlattener`: vocab-driven one-hots for categorical slot positions + stateless symlog on COUNT/TIME kinds, out_dim 5204) into plain 512×512 MLPs with a padded masked Discrete(30) head. Kills the architecture×observation-richness confound; fleet-independent vocab + fixed head = every scenario evaluable (unlike token anchors). Rejected: structured obs (information-poor, un-vec-trainable), new env obs mode (ripples mid-generation).
- **Scratch/no-BC, per-decision γ 0.997, no PopArt/semi-MDP for the baselines** — DQN can't BC-init (uniform protocol); γ^Δt/PopArt are contribution stack, not the traditional strawman-free baseline.
- **Range-mode training world for the baselines** (`train_multiscale_v5_mlp.json`, fleets U(4,30)): per-slot MLP heads need every action index in training — ratio mode reaches slots 27–29 with p<2%, leaving untrained head rows that industrial/very_long argmax over (confirmed review finding).
- **GRPO: group = 8 complete episodes on ONE scenario at FIXED 275k horizon** (train_hydra pins both): episode-sum outcome z-scored in-group must not encode exogenous draws — horizon spread empirically gave corr(horizon, return) ≈ −0.71. Old `grpo.py` NOT reused (ratio≡1 by construction; verified) — left registered, excluded from rosters.
- **ft fine-tunes: one reward lever per variant, same seed 4242** — protect (fleet_availability 2.0 + workload_balance 1.0, symptom), fatigue (fatigue_cost 2.5, cause), quality (repair_quality 2.5, weakest KPI); 100 eps vec5 from hc_v6 final.pt, ctor lr 3e-5 cosine→1.5e-6 (re-arm uses ctor lr). User cut from 150→100 eps and ordered auto-benchmark after.
- **Benchmark-path immutability**: the scenario-rotation fix deliberately touches ONLY training env construction — eval harness sequences stay byte-identical to the existing v6w generation.
- **hc_v6 fine-tune init = final.pt** (strongest stretch, natural continuation) rather than best.pt.

## Key Learnings
- **v6w results (corrected world, 16 agents)**: hc_v6 4th overall (mean rank 6.13) — best learned placement ever (v3/v4 gens ~11th); **5M collapse reversed**: hc_v6 #1 raw 5M products (110,295, edging greedy_reward‡ 110,211), hc_v6/last #1–2 final knowledge (65.2k/65.5k > every baseline); industrial 3rd (2164, above all honest baselines); TOPSIS-emp only honest agent ahead overall (4.29). Weak: small_scale (noise regime, random ranks 3rd), quality 0.26–0.39, illness/availability (fleet protection — the ft target).
- **5M curve signature of upskilling**: hc_v6 rolling MTTR FALLS over the episode (86→82) while greedy‡ degrades (83→86); knowledge keeps growing through the last quarter (61.5k→64.4k) while TOPSIS† flatlines — maintenance + growth vs equilibrium. Fatigue rides 0.19 vs 0.15 (protection gap = constant offset, not late divergence).
- **Training stability read (wandb datastore extraction)**: infra-stable, signal-plateaued — entropy 0.26→1.32→dip 0.11→0.38 (no collapse), KL early-stop fired 199/447 updates (44%, LR/epoch at trust-region edge), vf_loss 0.35→0.003, PopArt σ 45→655→~555 settled, LR schedule live to the 1.5e-5 floor; episode returns trendless after ~ep150, final-50 strongest (+3,760 vs −3,590 first-50) — justifies final.pt canonicalisation.
- **Wall-time (dynamics-normalized)**: heuristics 2.6–3.0 ms/decision ≈ SimPy env floor (Hungarian/MILP add nothing); greedy‡ 18.3 (counterfactual probe); hc_v6 28.2 (~11× floor; 7.9 h per 5M episode vs 47 min for TOPSIS†) — deployment-cost sentence for §7.
- **Scenario-rotation off-by-one (affects ALL historical trainings)**: `KataEnv.__init__` bootstrap build consumed one sampler draw → every `episodes_per_scenario` block straddled two factories (7+1). Mattered newly because GRPO groups key on block alignment. Empirically reproduced by two independent verifiers.
- **Analyzer label trap**: `AGENT_ORDER` filtering silently drops any eval key missing from `AGENT_LABELS` (bit hc_v6 — both learned agents absent from the v6w tables while episodes.csv was fine; hc_v5 pair also missing).
- **DQL global-RNG hazard**: agent exploration/replay drawing from stdlib `random` perturbs the simulator's seeded breakdown stream — world dynamics become a function of agent internals; fixed with private per-purpose `random.Random` streams persisted in checkpoints.
- **Untrained-head-slot hazard for per-slot MLP heads**: no parameter sharing across action slots (unlike the pointer head) — slots absent from training are random-init at eval and can dominate a greedy argmax.
- **A2C approx_kl ≡ 0 by construction** when old/new log-probs come from the same weights — the metric must be a post-step recompute for a single-step algorithm.
- **Workflow ops**: `args` passed to Workflow must be the real JSON value (a placeholder string reached the script verbatim — run stopped and relaunched); pipeline finder→verifier keeps wall-clock tight (18 Fable verifications overlapped 4 finder lenses).

## Solutions & Fixes
- **Analyzer**: hc_v6/hc_v6_last + hc_v5 pair + 6 MLP labels added to `AGENT_LABELS`; v6w analysis regenerated locally over the pulled 81 MB tree.
- **Lifecycle benchmark**: `scripts/dgy_v6w_lifecycle.sh` (16 agents × lifecycle 5M, same seed/parts tree, merge into `hvp_eval_v6w/lifecycle`, marker `V6W LIFECYCLE DONE`).
- **ft queue**: `scripts/dgy_v6_ft_queue.sh` (3 variants × 100 eps on GPU lanes 1/2 + integrated best+last × 4-scenario benchmark, marker `V6 FT QUEUE DONE`); `ft_*` eval keys + labels.
- **Wall-time**: `scripts/compare_wall_time.py` — ms/decision (wall_s/n_steps) + s/1M t.u. (wall_s/final_sim_time), per-scenario mean±std + pooled ratio-of-sums, reuses analyzer labels, emits `wall_time.csv` + `wall_time_table.tex`.
- **MLP baseline package (commit c8a758f, 30 files, +6,583 lines)**: `mlp_encoder.py` (flattener + trunk + 3 heads + cont-kinds layout-drift guard), `a2c_mlp.py` (vec-capable, corrected GAE, post-step KL, vocab-rebuilding attach), `grpo_mlp.py` (episode-group advantages, frozen pre-epoch old log-probs), `dql_mlp.py` (Double DQN, 500k uniform replay, gradient cadence in observe_transition, private RNGs); registry/Literal/`SET_OBS_AGENT_TYPES` single-sourced; per-family injection branch + 1.25 schedule inflation + GRPO pins + DQL seed injection in train_hydra; eval MLP branch (`n_actions=max_techs`, `legacy_obs=False`, `net.eval()`); rotation fix in `runner._build_env` AND `vec_env._build_worker_env` (`reset_scenario_cache()` post-construction); vec-loop wandb monotonic step + real axes as fields; `scripts/dgy_trad_baselines_queue.sh` (GPU gate, final.pt completion marker, per-key part gates). Suite **644 passed**; micro-train (all 3 families) + micro-eval smokes proved the full path pre-review.
- **Review process**: 4 Opus finder lenses → 18 Fable adversarial verifications (repro-or-refute) → 17 confirmed, 1 refuted → 4 parallel fixers + orchestrator fixes → suite green.
- **dgy git pulls from this shell**: `eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519 && ssh -A serval-dgy 'git pull --ff-only'; ssh-agent -k` (no agent in the tool shell).

## Files Modified
- `scripts/analyze_hvp_results.py`: +14 labels across three commits (hc_v6 pair, ft six, hc_v5 pair + MLP six).
- `scripts/eval_human_vs_performance.py`: ft keys; MLP_AGENT_CLASSES/MLP_CHECKPOINTS + fourth build branch; `load_set_tokenizer(peek=)`; `trained` shorthand.
- `scripts/train_hydra.py`: per-family injection branch (set_transformer unchanged; a2c/grpo/dql specifics), `_sized_rounds` 1.25 inflation, GRPO scenario+horizon pins, DQL seed injection, agent-aware prints.
- NEW `scripts/dgy_v6w_lifecycle.sh`, `scripts/dgy_v6_ft_queue.sh`, `scripts/compare_wall_time.py`, `scripts/dgy_trad_baselines_queue.sh`.
- NEW `src/agents/networks/mlp_encoder.py`, `src/agents/a2c/{__init__,a2c_mlp}.py`, `src/agents/grpo/grpo_mlp.py`, `src/agents/dqn/dql_mlp.py`; `src/agents/networks/replay_buffer.py` (optional rng); `src/agents/__init__.py` (exports).
- `src/experiment/config.py` (Literal + `SET_OBS_AGENT_TYPES`), `src/experiment/runner.py` (registry, machine_slot_length injection, rotation fix, wandb steps), `src/experiment/vec_env.py` (membership representation + rotation fix).
- NEW configs: `run_configs/agents/{a2c_mlp,grpo_mlp,dql_mlp}.json`, `run_configs/benchmark_suite/train_multiscale_v5_{grpo,mlp}.json`, conf symlinks.
- NEW tests: `test_set_flattener.py`, `test_a2c_mlp.py`, `test_grpo_mlp.py`, `test_dql_mlp.py`, `test_scenario_rotation.py`; `test_hydra_configs.py` extended (agent groups + grpo pin).
- `CLAUDE.md` (2026-08-13 preserve).
- Commits: `fb72b9b` (analyzer hc_v6 + lifecycle queue) → `bc5e5c1` (ft queue) → `bc57d5f` (ft 100 eps + bench + keys) → `b4b8d9b` (wall-time) → `c8a758f` (MLP package) → `e2d7064` (preserve). All pushed; dgy at parity.

## Pending Tasks
- **Three dgy queues in flight** (watchers die with this session — logs/markers are the durable interface): lifecycle (`v6vec`, `reports/v6w_lifecycle_queue.log`, `V6W LIFECYCLE DONE`); ft (`v6ft`, `reports/v6_ft_queue.log`, `V6 FT QUEUE DONE`); trad baselines (`trad`, `reports/trad_baselines_queue.log`, `TRAD BASELINES DONE`, self-gated until GPUs 0–2 free).
- On completion: read regenerated analyses (all new rows land beside hc_v6), report ft cost/benefit (illness/availability/quality vs the 5M throughput crown) and the MLP-anchor gap (§7.4 story).
- Serval-paris checklist unchanged (still down, re-verified 08-13): hc_v5 D1 round-count check, historical checkpoints into the v6w parts tree, v4/very_long re-merge.
- Paper: §6/§7/abstract numbers once the roster settles; lifecycle table; notation package; zhang bib swap sign-off.

## Errors & Workarounds
- **Workflow args placeholder**: passed `"SEE_SCRIPT_NOTE"` instead of the brief → stopped the run (TaskStop) and relaunched with the full brief JSON in `args`.
- **Cross-fixer test breakage**: the flattener's new cont-kinds guard tripped `test_grpo_mlp.py`'s obs helper (no kinds channel) — helper updated to emit kinds rows like its siblings.
- **`machine_slot_length` injection** would have TypeError'd SetTransformerAgent — guarded to the MLP family only.
- **eval script needs PYTHONPATH=src** when run locally (no editable install — the old serval lesson, still true locally).
- **zsh `echo ===`** fails (`=cmd` expansion) — quote it.

## Custom Notes
None

---

## Quick Resume Context
v6w read is done and preserved: hc_v6 is the best-learned agent ever benchmarked here (4th of 16 overall; #1 raw 5M throughput above the reward oracle; #1–2 knowledge maintenance), with fleet protection and repair quality the open gaps. Three dgy queues are (or will be) producing the next roster additions: lifecycle scenario for all 16 agents, three one-lever reward fine-tunes of hc_v6, and three adversarially-reviewed traditional MLP baselines (same set information, plain networks — the §7.4 architecture ablation). Check the three markers first (`V6W LIFECYCLE DONE` / `V6 FT QUEUE DONE` / `TRAD BASELINES DONE` in reports/ on dgy), then merge reads + the §6/§7/abstract number work. Everything through `e2d7064` is pushed; serval-paris still down.

---

## Raw Session Log

**Turn 1 — /resume.** Loaded CLAUDE.md + 3 session summaries (06-08 vec-fix/KATA-1/v6-relaunch; 03-08 v3/v4/semi-MDP/v5; 20-07 baselines/benchmark). Reported project state; offered to check the v6w queue on dgy.

**Turn 2 — "Check the dgy status first, then report."** ssh to serval-dgy: `V6W BENCH DONE` at 2026-08-07 00:03Z, all parts/merges/analysis rc=0. Found rank tables held only 14 agents — hc_v6/hc_v6_last missing. Root cause: `analyze_hvp_results.py` AGENT_LABELS lacked v6 keys; AGENT_ORDER filter silently dropped them (data intact in episodes.csv). (Plan mode briefly active, then exited.) Added the labels, pulled the 81 MB merged tree local via tar-over-ssh, regenerated analysis. Computed headline KPIs + final/peak knowledge from steps.csv.gz. Reported: overall rank TOPSIS-emp 4.29 > TOPSIS† 5.38 > greedy‡ 5.71 > hc_v6 6.13; very_long: hc_v6 #1 products 110,295, knowledge #1–2 (65.2k/65.5k), illness mid-pack; industrial: hc_v6 2164 best-honest-info; baseline: beats every informed baseline + oracle, only TOPSIS-emp ahead; small_scale weak. Serval-paris ping: still down.

**Turn 3 — "What about the lifecycle benchmark?"** Never run — user's own 08-06 scoping decision ("4 standard scenarios only"); merge log shows `[skip] lifecycle: no artefacts yet`. All infra exists (config with 10-event schedule, eval scenario entry, analyzer support, serval queue fan-out pattern). dgy idle. Estimated ~0.5–1 day for 16 agents. Offered to launch + fold in the analyzer-fix commit.

**Turn 4 — "Do all of that yes."** Wrote `scripts/dgy_v6w_lifecycle.sh` (per-agent parts into hvp_v6w_parts/<agent>/lifecycle, seed 20260722, merge + full analysis at end). Committed `fb72b9b` + pushed; dgy pull needed an ssh-agent (none in this shell) — throwaway agent dance. Launched in existing zellij `v6vec` (`zellij --session v6vec run`); verified ARMED + first 6 parts running, GPU 0 at 54%, part logs clean (24 techs, 5M horizon). Armed a background 10-min watcher on the queue log.

**Turn 5 — "Current performance of my best agent vs unfairly informed + normal baselines, all environments?"** Computed per-scenario tables with class annotations (INF†/ORACLE‡/HONEST/LEARNED) incl. knowledge from steps. Reported: overall 4th of 16, beats 5/6 informed baselines; per-scenario detail (5M #1 products above oracle; industrial 3rd; baseline tied 3rd above all informed; small_scale 5th); read: thesis horizon won, fleet protection + quality the open gaps, TOPSIS-emp wins on protection not production.

**Turn 6 — "Produce some curves... mean mttr across time, fatigue, knowledge match etc."** Loaded dataviz + artifact-design skills; binned steps.csv.gz to 120 points × 4 scenarios × 5 agents (hc_v6, hc_v6_last, greedy‡, TOPSIS†, TOPSIS-emp) × 6 KPIs; validated the 5-slot categorical palette light+dark (script validator); built a token-themed HTML artifact (global legend, crosshair tooltip all-series, keyboard nav, per-scenario data tables). Published: https://claude.ai/code/artifact/f2d0534b-c5ea-4ce0-b72e-f3902d600c06. Trend stats: 5M hc_v6 MTTR 86→82 falling vs greedy‡ 83→86 rising; knowledge 61.5k→64.4k growing vs TOPSIS† flat; fatigue offset 0.19 vs 0.15. Industrial: MTTR drifts up (upskilling doesn't pay back in 100k).

**Turn 7 — "Show me training curves... how stable the learning is."** Located offline wandb run `09r693tg` on dgy; scanned the datastore (history records) → 600 per-episode returns, 447 update series, 3 inline evals. Built the stability artifact (stat tiles + wide return chart with rolling mean±σ band + eval markers + 8 panels incl. log-scale KL/vf_loss): https://claude.ai/code/artifact/8e74ca9d-98f6-4187-8c94-e90226a5f88d. Verdict: infra-stable (no divergence/collapse, guarded KL 199/447 early stops, converged critic, live LR schedule), signal-plateaued (trendless after ~ep150, final-50 strongest, noisy 5-ep eval selector).

**Turn 8 — "Prepare training scripts... finetuning with specific rewards. Suggest rewards, an lr schedule, create scripts ready to launch on DGY."** Inspected reward config surface (normalize_components=true → coefficients are shares), launcher params, hc_v6 checkpoints on dgy, GPU state (lifecycle owns GPU 0). Designed 3 one-lever variants (protect/fatigue/quality), lr 3e-5 warmup-10 cosine→floor (ctor-lr re-arm makes it safe), 150 eps vec5, seed 4242, init final.pt. Wrote `dgy_v6_ft_queue.sh` (GPU lanes 1/2), dry-run validated all 3 composes, committed `bc5e5c1`, pushed, dgy pulled. Gave the launch one-liner, offered to fire.

**Turn 9 — "Launch it and benchmark afterwards, no need for as much training time, let's start with 100 episodes each."** Reworked the queue: 100 eps + integrated benchmark stage (best+last × 4 scenarios, eval seed 20260722, same parts tree, merge + analysis); added `ft_*`/`ft_*_last` eval keys + 6 analyzer labels. Committed `bc57d5f`, pushed, dgy pulled, launched in new zellij `v6ft` (13:45Z): ft_protect on GPU 1 + ft_fatigue on GPU 2 training, ft_quality queued on lane A. Armed a second watcher.

**Turn 10 — "Create a script to compare wall time for all baselines + our agent... normalized by simulation/decision steps."** Wrote `scripts/compare_wall_time.py` (ms/decision + s/1M t.u.; per-scenario mean±std; pooled ratio-of-sums; analyzer-label reuse; csv+tex out). Ran on v6w: heuristics 2.60–2.97 ms/dec ≈ env floor, greedy‡ 18.3, hc_v6 28.2 (~11×). Committed `b4b8d9b`, pushed, dgy pulled.

**Turn 11 — /effort ultracode + "Create new learned baselines with traditional algorithms (A2C / GRPO / DQL) with traditional architectures. Goal: showcase the worth of our architecture."**
- *Workflow 1 (map, 5 agents, 816k tok)*: 4 Opus readers (agent suite / env obs / training loops / eval wiring) + Fable synthesis → design brief. Core verdicts: set obs is the ONLY fixed-shape mode (Discrete(30)+mask padding set-only); invalid actions LIVELOCK; serial loop updates once/episode (the Rainbow starvation lesson); old grpo.py ratio≡1; information parity requires feeding MLPs the set payload.
- *Workflow 2 (implement, 4 agents, 607k tok)*: first launch passed an args placeholder — stopped, relaunched with the brief. Fable built `mlp_encoder.py` (one-hot LUTs from the 152-token vocab with OTHER bins, symlog dispatch, mask features, out_dim 5204, 16 tests) then 3 parallel Opus implementers built the agents + configs + own test files (green, disjoint; A2C added vec interface + snapshot/restore; GRPO 31 tests incl. ratio-moves pin; DQL cadence counters + scale transfer).
- *Wiring (orchestrator)*: registry/Literal/`SET_OBS_AGENT_TYPES` single-sourced (vec_env by membership), machine_slot_length injection (guarded off set_transformer after catching a would-be TypeError), per-family train_hydra branch, eval MLP branch, labels, hydra tests. Suite 621. Dry-runs + real micro-trainings for all 3 families + micro-eval of the smoke checkpoints — full path proven.
- *Queue*: `dgy_trad_baselines_queue.sh` (GPU gate, 3 parallel trainings, canonicalise, benchmark, merge).
- *Workflow 3 (review, 22 agents, 2.34M tok)*: 4 Opus lenses → 18 Fable repro-or-refute verifications → 17 CONFIRMED / 1 refuted. Criticals: GRPO horizon-sum confound (corr −0.71 reproduced), scenario-rotation off-by-one (7+1 blocks reproduced; affects ALL historical trainings). Majors: untrained slots 27–29, DQL global-RNG world perturbation, replay window < scenario dwell, queue gating holes. Minors: A2C approx_kl≡0, vocab-LUT rebuild gap, wandb step collisions, tu_per_decision 21% off, kinds-drift blindness, chmod.
- *Workflow 4 (fix, 4 agents, 452k tok)* + orchestrator: DQL private RNGs + 500k buffer; A2C post-step KL + vocab rebuild; flattener cont-kinds guard; runner rotation fix + wandb monotonic steps. Orchestrator: GRPO horizon pin, 1.25 sizing inflation, range-mode `train_multiscale_v5_mlp.json`, queue rewrite (final.pt completion marker, per-key gates), vec_env rotation fix, DQL seed injection, grpo test helper kinds. Suite **644**.
- Committed `c8a758f` (30 files, +6,583), pushed, dgy pulled (131 package tests green there), queue armed in zellij `trad` at 15:43Z (gate: 2 GPUs busy → waiting), third watcher armed.

**Turn 12 — /preserve.** All categories selected. CLAUDE.md updated (Next steps rewritten around the 3 queues + v6w results; +3 Key-decisions rows; +2 Blockers; 2026-08-13 session notes), 204 lines, committed `e2d7064` + pushed.

**Turn 13 — /effort xhigh (ultracode off) + /compress.** This log.
