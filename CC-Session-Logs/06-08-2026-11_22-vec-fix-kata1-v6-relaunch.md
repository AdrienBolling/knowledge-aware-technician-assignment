# Session Log: 06-08-2026 11:22 - vec-fix-kata1-v6-relaunch

## Quick Reference (for AI scanning)
**Confidence keywords:** vec, AsyncVectorEnv, parallel-envs, gymnasium-1.2, NEXT_STEP autoreset, KATA-1, config-DI, get_config singleton, travel_time, failure_wise_knowledge_parameters, corrected world, generation boundary, hc_v6, vec5, dgy, serval-dgy, zellij, v6w, hvp_eval_v6w, benchmark generation, LR schedule, total_updates, final.pt, canonicalise, SeededResetWrapper, wandb offline, datastore, plateau, training curve, artifact, BC warm-start, TOPSIS, serval-paris, checkpoint custody, ultracode, workflow, Fable review
**Projects:** knowledge-aware-technician-assignment (KATA/FactoReal paper)
**Outcome:** Parallel-env training fixed+verified (17-defect audit → SHIP), KATA-1 world fix landed, v6 retrained vec5 in corrected world (plateau ~ep400, products ×3), v6w benchmark generation launched on dgy.

## Decisions Made
- **Parallel envs are BACK** (user reversal): the "SimPy ill-suited" verdict traced to the *lookahead deepcopy* failure, not AsyncVectorEnv (workers rebuild simulators; nothing live is pickled).
- **KATA-1 fix = dependency injection, not singleton-sync** — required `sim_cfg` ctor param, both import-time captures deleted; no default on purpose (missed call sites fail loudly).
- **Travel time 15 activated** (user: "either is fine") — configs+paper already said 15, so zero doc edits; **failure-wise knowledge REQUIRED live** (user: "absolutely want").
- **Generation boundary**: existing hvp_eval_v4 CSVs stay archived+valid (all agents 10/False both sides); new corrected-world generation `hvp_eval_v6w` for everything; BC re-collected.
- **v6 serial run CUT at ~ep 200** and relaunched vec5 after a 20-min local typical-run gate (user directive); serial partial archived, not a clean ablation (LR bug).
- **Lifecycle eval stays later**; benchmark generation = 4 standard scenarios only (user choice).
- **zellij over tmux** for all job launches (user preference; installed 0.44.3 on dgy).
- Model policy sharpened: Opus for coding agents, Fable reserved for planning/audit/critical logic.

## Key Learnings
- **Vec audit (ultracode: 4 Opus lenses + Fable synthesis, ~850k tok, 17 verified defects).** Headliners: vec loop never saved `final.pt` (every v3/v4/v5 "last" = last periodic round, ~11% early); D1 `total_updates` sizing was vec-cadence-only → the serial dgy run trained at near-constant LR; vec inline-eval stats computed then discarded; vec autoreset episodes 2+ drew disruption timing from OS entropy; serial loop never seeded process globals; intervals meant rounds (vec) vs episodes (serial); vocab drift RE-INDEXES keys (not append-only) → prefix-copy embedding resize would silently scramble rows.
- **KATA-1 mechanism**: `GymTechnician.py:20` + `GymTechDispatcher.py:17` captured `get_config()` at import while launchers point KATA_CONF_PATH at a nonexistent file → effective travel_time 10 (not 15) and failure-wise False (not true) in ALL trainings/evals ever. Only two such captures existed; survey found no other inert sim.* surface.
- **v6 training read** (curve artifact https://claude.ai/code/artifact/8ff8c06b-b217-4015-b4ac-427838f0e988): plateau from ~ep 400; eval products 304→929→925 while illness 865→1083, availability 0.659→0.628 — throughput learned, fleet protection NOT → **reward-design signal** for the deferred reward work. Entropy 0.26→0.38 (no collapse); LR verified annealing 3.0e-4→1.5e-5 floor (first live schedule in any vec run); BC corrected-world 88.2% ≈ old-world 88.4% → the 74.1→88 jump was architecture, not world.
- **wandb offline extraction**: per-episode vec returns exist ONLY in the offline .wandb file; scan with `wandb.sdk.internal.datastore.DataStore` → protobuf history records; key = `item.key or "/".join(item.nested_key)` (newer protos use nested_key).
- **zellij headless**: `action write-chars` into a background session drops keystrokes; `zellij --session S run -- cmd` is reliable (used for both launches).
- **Vec-vs-serial wall-clock**: ~11× gap was ~10× parallelism + only ~1.1–1.3× hardware; per-episode cost 4080≈V100-node because SimPy (single-thread Python) + batch-1 forwards dominate.

## Solutions & Fixes
- **Vec parity package (commit 255560a, suite 509)**: final ckpt in vec loop; cadence-aware `total_updates` (serial=episodes, warmup ep/25); vec eval logging; derived-seed autoresets (SeededResetWrapper; sanity fingerprints episode 2 — bit-identical solo-vs-vec); `_run_episode` seeds globals; episode-denominated intervals both modes; `_resolve_agent_vocab` guard at load AND vec-build + non-prefix resize aborts; autoreset_mode pinned NEXT_STEP + guard; mp context fork; sync-n>1 warns; queue scripts canonicalise final>round*>ep*>best. Fable adversarial review: SHIP (2 riskiest fixes re-simulated pre-fix and shown to fail).
- **KATA-1 (commit 7d9f547, suite 516)**: sim_cfg DI through ScenarioBuilder + dispatcher.sim_cfg (lifecycle hires); `tests/test_config_injection.py` (7, incl. per-failure-key multipliers + meta-guard against reintroducing captures).
- **dgy cutover sequence**: 20-min local vec5 gate (36 eps, ~33 s/ep, no crashes) → push → tmux kill-session v6train → archive (hc_v6_serial_partial, bc_topsis_v6_oldworld, logs renamed) → pull → 30 smoke tests → zellij `v6vec` relaunch (fresh BC → 600 eps vec5).
- **Monitor pattern**: new-content-only byte-offset watcher over ssh (stale ABORT lines can't re-fire); one poll per 5–10 min; terminal events exit the monitor.
- **uni.lu VPN**: per-login-session NM profile; reconnect needs interactive secret → `! nmcli connection up "uni.lu VPN" --ask`.

## Files Modified
- `src/experiment/runner.py`: vec final ckpt, eval logging, `_resolve_agent_vocab`, episode-keyed intervals, `_run_episode` global seeding, shared best threshold.
- `src/experiment/vec_env.py`: derived-seed autoresets, NEXT_STEP pin+guard, fork context, sync warning.
- `src/agents/ppo/ppo_set_transformer.py`: snapshot preserves `_last_sim_time`; `_resize_token_embedding` prefix-property validation.
- `scripts/train_hydra.py`, `scripts/train_hc_improved.py`: cadence-aware total_updates.
- `scripts/dgy_v6_train.sh` (vec5 relaunch), `scripts/serval_v6_train_queue.sh` (canonicalise fix), `scripts/sanity_vec_env.py` (episode-2 fingerprints), NEW `scripts/dgy_v6w_benchmarks.sh` (894179f).
- `src/kata/entities/technicians/GymTechnician.py`, `src/kata/entities/tech_dispatcher/GymTechDispatcher.py`, `src/kata/scenario.py`, `src/kata/env.py`, `scripts/disruption_stats.py`: KATA-1 DI.
- NEW `tests/test_vec_training.py` (6), `tests/test_config_injection.py` (7); adjusted test_dispatcher/test_gym_technician/test_technician_profiles (ctor plumbing only).
- `CLAUDE.md` (2× session notes), memory `zellij-over-tmux.md`.
- Commits: 255560a, 7d9f547, a49fc8f, 894179f + preserve commit; all pushed.

## Pending Tasks
- **v6w benchmark generation RUNNING on dgy** (zellij `v6vec`, log `reports/v6w_bench_queue.log`, marker `V6W BENCH DONE`, out-root `reports/hvp_eval_v6w`): hc_v6/hc_v6_last + 14 baselines × 4 scenarios; monitor active.
- **serval-paris return checklist**: v3/v4/gaefix/anchor/human checkpoints live ONLY there → evaluate into the idempotent v6w parts tree; check hc_v5 round-count (D1); re-merge hvp_eval_v4/very_long from parts (destructive-merge risk); decide rerun-vs-drop for v5 (dead at 2/600).
- **Reward design** (user-deferred, now with a concrete signal: throughput learned, fleet protection not).
- Paper: §6/§7/abstract numbers once v6w lands; lifecycle eval later; notation package; paper-repo push.

## Custom Notes
None

---

## Quick Resume Context
v6 (corrected world: travel 15 + failure-wise knowledge live, vec5) finished 2026-08-06 05:45Z on dgy — plateau from ~ep 400, throughput ×3 but fleet-protection metrics flat/worse. The v6w corrected-world benchmark generation is running on dgy in zellij session `v6vec`; historical learned agents join when serval-paris returns. Everything committed and pushed through 894179f.

---

## Raw Session Log

The complete verbatim conversation (including the pre-compaction portion) is preserved in the Claude Code transcript:
`/home/gourmet/.claude/projects/-home-gourmet-repositories-knowledge-aware-technician-assignment/e3935169-120e-4ada-8fea-9e9adb645474.jsonl`

Chronological narrative of the session (2026-08-05 → 2026-08-06):

1. **v5/serval status check** → serval-paris found dark (needs power-cycle); v5 queue fate unknown (2/600).
2. **Ultracode agent-improvement investigation** (user request, no code changes): `analysis/agent_improvement_report.md` — defects D1–D12, literature-backed roadmap; `copy.deepcopy(env)` lookahead refuted (SimPy generators unpicklable).
3. **D1–D6 fix triage + implementation** (user directives incl. D3 two-view pooling, D4→lifecycle eval config, D5 de-scoped): suite 501→503 after adversarial review second round; D11/D12 obs bugs found and fixed; lifecycle events implemented; v6 configs prepared.
4. **v6 serial launch on dgy** (user: no parallelisation, push authorized): dgy quirks (GPU3 dead, torch cu126 downgrade + --no-sync, PATH), tmux `v6train`, BC 88.4%.
5. **Monitor false alarm** (stale ABORT line re-fired) → byte-offset new-content-only watcher.
6. **User: zellij over tmux from now on** (memory saved).
7. **Performance questions**: training oscillation explained (episodes_per_scenario=5 blocks); serial-vs-vec slowdown arithmetic (10× parallelism, ~1.2× hardware).
8. **User: "Try to fix parallel envs locally"** (ultracode) → empirical probes (sanity PASS, vec smoke PASS) + 4-lens audit workflow → 17 defects → 4-coder implementation workflow → 509 tests → Fable adversarial review SHIP → two hardening follow-ups (serial vocab guard, resize prefix check) → commit 255560a.
9. **KATA-1 question** ("most logical fix?") → DI recommendation accepted (travel 15, failure-wise required) → Opus implementation → suite 516 → commit 7d9f547.
10. **User: validate locally 20 min, then cut dgy serial and relaunch vec** → gate passed (36 eps/20 min) → cutover (archive, pull, 30 smoke tests, zellij `run` launch after write-chars failed) → BC corrected-world 88.2% → vec5 training live.
11. **User logged off**; VPN down on return path clarified (uni VPN is per-login-session; user reconnected).
12. **v6 DONE** (600 eps, 19h15m, rc=0, final.pt canonicalised as last). Wandb-offline history extracted (datastore scan); plateau analysis; dataviz+artifact-design skills; training-curve artifact published (https://claude.ai/code/artifact/8ff8c06b-b217-4015-b4ac-427838f0e988).
13. **Benchmark roster reality-check**: v3/v4/gaefix/anchors/human checkpoints only on serval-paris; hc_v5 nonexistent. User approved full-roster-on-dgy + lifecycle-later → `scripts/dgy_v6w_benchmarks.sh` (894179f) launched in zellij; quiet monitor armed.
14. **/preserve** → CLAUDE.md updated (190 lines) + pushed; **/compress** → this log.
