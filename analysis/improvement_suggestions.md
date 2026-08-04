# Prominent Improvement Suggestions — post-fix roadmap

**Date**: 2026-08-04. Companion to `analysis/agent_improvement_report.md` (full evidence,
citations — all adversarially verified, 102/114 clean, zero fabricated — and the defect audit).

**Standing decisions honored here**: reward design deferred; checkpoint selection stays
best+last (D5); knowledge saturation accepted as realistic (D4) with the **lifecycle scenario**
as its evaluation counterpart (implemented, runs later); **no environment parallelisation**
(single-env training from v6 on).

**Already done (v6 package, in this commit)**: D1 LR schedule (sizing + floor + extension
re-arm), D2 dropout out of acting/eval, D3 role-bound slot fusion + two-view feature-context
pooling, D6 anchor GAE, D11 boolean tokens un-UNK'd, D12 machine-history features un-zeroed,
lifecycle events engine + 5M lifecycle eval config. v6 must land before anything below is
measured — every earlier number carries the defects.

---

## 1. Pair-scoring policy head (top architecture priority)

The one strong architectural theme the learned-dispatching literature agrees on: score the
**(technician, ticket) pair**, not a pooled context against technician embeddings.

- Cheapest retrofit: **MatNet-style mixed-score attention** — inject the env's per-pair
  expected-repair-time scalar (already computed for the baselines via
  `assignment_cost_matrix()`) as a bias inside the tech↔ticket cross-attention scores. The
  policy then *starts* from the information SPT/TOPSIS rank by and learns residuals.
- Structural upgrade: **DAN-style interleaved cross-attention** (within-set self-attention ×
  cross-set attention, both directions) + score `MLP([s; e_i; s⊙e_i])` instead of the bare
  rank-64 bilinear form; **tanh-clipped logits (C≈10)** per Kool et al. — a known stabiliser.
- D3's binder + feature-context is the enabling first step (features are finally
  distinguishable); the pair head is the remaining step. Note it re-triggers BC re-collection
  and touches §5.2.2/§7.4 prose.

Evidence: Song et al. (IEEE TII 2023), DAN (TNNLS 2024), ScheduleNet, MatNet (NeurIPS 2021),
Kool et al. (ICLR 2019), LEHD (NeurIPS 2023: capacity into the per-decision decoder).

## 2. Critic & discounting package

The gap to the *myopic* reward oracle is a critic-quality symptom — RL only beats myopia
through value estimates that see past one step.

1. **γ_t.u. sweep first** (config-only, Hydra multirun {0.9995, 0.9999, 0.99995}): the
   single most impactful hyperparameter (Andrychowicz et al., ICLR 2021), and the
   discriminating experiment between "defects explained it" and "horizon explained it".
2. **HL-Gauss categorical value head** (Farebrother et al., ICML 2024): ~50 lines, no extra
   simulation, best-evidenced critic upgrade under noisy non-stationary targets.
3. **Per-scale PopArt statistics** (multi-task PopArt, AAAI 2019): multiscale training is
   multi-task; one scalar μ/σ lets big-factory returns dominate.
4. **Drop value-loss clipping** (`clip_eps_vf: null`) — empirically unhelpful-to-harmful.
5. **λ^Δt** (Doya 2000): one line; the advantage half-life currently shrinks to ~600 t.u.
   exactly when the factory is busiest — the pathology γ^Δt was built to remove, reintroduced
   through λ.
6. **γ-curriculum** (OpenAI Five precedent): anneal γ_t.u. upward over training.
7. **Average-reward PPO** (ATRPO/APO) as the principled 5M endgame *if* the γ-sweep still
   leaves the (re-measured, n>1) 5M gap: differential rewards `r − ρ̂·Δt`, mean-zero critic.
   The SMDP gain criterion literally is "throughput per t.u.".

## 3. Keep exploiting the teachers during RL (no reward change)

BC-init is currently the only use of the experts; the probe API is unused by the agent.

- **Kickstarting anchor**: label PPO rollout states with TOPSIS (near-free), add an annealed
  KL/CE term (TGRL auto-balancing avoids schedule tuning). Fixes the unprotected BC→PPO
  handoff and floors the long-horizon drift. Student-controlled trajectories only (Czarnecki);
  never mix teacher-controlled episodes into the PPO buffer.
- **Asymmetric critic**: probe outputs (`assignment_reward_estimates`, `expected_repair_times`)
  into the value tower only — deployability preserved; keep the observable stream too
  (Baisero & Amato bias caveat).
- **Advantage-weighted imitation** (AggreVaTeD): imitate TOPSIS only where the probe says it's
  good — the route to *exceeding* the myopic oracle, not converging to it.
- **A2D warning**: never use the reward-greedy oracle as an action teacher (privileged,
  unimitable); its information goes to the critic or advantage weights.
- **JSRL roll-ins** for long-horizon training (TOPSIS opens the episode, agent finishes; GAE
  restricted to agent-controlled segments).
- **QDagger handoff** between generations (distill v6 into v7 instead of fresh BC).

## 4. Decision-time lookahead (the direct answer to greedy‡)

The reward-greedy oracle is `argmax_a r(s,a)` — one-step lookahead with V=0. The minimal agent
that dominates it is `argmax_a r + γ^Δt·V(s′)` (Bertsekas: lookahead is a Newton step;
rollout is provably no-worse than its base policy on stochastic scheduling).

- **Feasibility corrected this session**: `copy.deepcopy(env)` fails (live SimPy generators) —
  no cheap clone exists.
- Order of attack: (i) **deterministic one-step approximation** — build s̃′ analytically
  (tech busy for ETA, ticket dequeued, knowledge increment), no sim advance; days of work,
  doubles as the sharpest critic diagnostic (if `r + γ^Δt·V(s̃′)` doesn't beat `r`, the critic
  is indicted); (ii) full state-snapshot clone (~1–2 weeks) only if the probe pays;
  (iii) Gumbel sequential-halving / depth-k rollouts only after that (50–100× eval cost).
- Honesty label: a search-wrapped agent is ‡-class (simulator probes) — frame as digital-twin
  deployment. Single-env training makes this MORE attractive: the GPU idles during env
  stepping, so decision-time compute is comparatively cheap.

## 5. Scale & curriculum

- **Make industrial interior**: extend sampler bounds past the eval point (techs ~34, machines
  ~110 — capped by max_techs/max_machines). Interpolation reliably works where extrapolation
  fails. Config-only. Expect (and budget) a small per-scale cost from the wider distribution.
- **Prioritized Level Replay** over scenario configs (score = value-loss magnitude, already
  computed). Zero extra simulation.
- **Held-out-template eval** to rule out template memorisation (12+7 templates is few).

## 6. Benchmark statistics (minimal, per the D5 decision)

No selector work — but two cheap benchmark-side items protect every future conclusion:
- **Re-measure 5M with n≥5 episodes** before designing anything else for it (every current 5M
  claim is n=1).
- **Common random numbers done right**: per-process RNG streams (each machine-component
  hazard, illness, horizon draw) so all agents face identical failure realisations — paired
  comparisons with drastically tighter intervals at the same cost.

## 7. Memory (last, expectations tempered)

Memory does not improve credit assignment (Ni et al., NeurIPS 2023) — it will not fix the
oracle gap or the 5M gap. The honest observability additions first: running per-tech empirical
repair statistics (what empirical-TOPSIS uses), remaining-busy/disruption durations,
time-since-decay. Recurrence only with stored-state burn-in BPTT (the zero-BPTT GRU attempt
was the known-worst configuration and proves nothing); GRU over GTrXL for endless horizons.

## 8. Single-env consequences (new constraint)

Dropping env parallelisation makes wall-clock the binding constraint (~3–4 days/600 eps on a
V100-class GPU vs ~5 h vec10). This re-ranks the menu toward **sample-efficiency**:
teacher-in-the-loop (§3) and PPG-style auxiliary value phases (reuse collected data at higher
value-side sample reuse) gain priority; anything demanding more environment interaction
(bigger sweeps, PLR breadth) loses. If wall-clock becomes prohibitive, the principled fallback
is fewer, longer-trained generations with QDagger handoffs rather than re-enabling vec.

## 9. Lifecycle evaluation (prepared, runs later)

`run_configs/benchmark_suite/lifecycle.json` + the `lifecycle` eval scenario: 5M horizon,
senior retirements → trainee/junior hires, capacity additions, most-breakdowns renewals.
What to read when it runs: post-shock recovery slope (throughput after each retirement wave),
**novice onboarding rate** (knowledge growth of the hires under each dispatcher — the
upskilling thesis in its sharpest form), and renewal response (do policies exploit
good-as-new machines). The knowledge-aware agent should win onboarding; the informed
heuristics should win the immediate post-shock throughput. Either outcome is a §7 result.
