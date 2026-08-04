# Agent Performance Improvement Report

**Scope**: architecture, algorithms, and training-procedure add-ons for the KATA set-transformer
agent. Reward design explicitly excluded (user decision, 2026-08-04). Nothing implemented —
this is an investigation report.

**Method**: 4 parallel code audits (network architecture, PPO/training loop, observation/action
design vs baseline information, empirical evidence base) + 7 literature surveys (114 reference
checks), followed by (a) first-hand re-verification of every load-bearing code claim against the
source (file:line cited throughout — no Part I item is taken from the audit agents on faith),
(b) an adversarial web-verification pass over every citation (outcome in Part IV: 102/114 fully
clean, zero fabricated papers, 12 corrections applied below), (c) an independent completeness
critique of the draft, whose accepted findings are folded in, and (d) one empirical feasibility
test (W5's env-cloning assumption — it failed; see W5).

**Date**: 2026-08-04. Code state: local `main` @ 178cfc4 (= serval state at v5 launch).

---

## 0. Executive summary

Two conclusions, in tension, both true:

1. **Nine verified implementation defects (six major) currently cap the agent** — Part I. They
   plausibly account for much of failure (a) (the gap to the myopic reward oracle), most of (c)
   (checkpoint instability), part of (e) (the unresolved quality collapse), and an unquantified
   share of (b) (the 5M collapse). They cost days to fix, and every method comparison — including
   the in-flight v5-vs-v4 verdict — is confounded until they land.
2. **The 5M failure (b) is most likely an objective-horizon problem on top of that.** With
   γ = 0.9999 per t.u., the credit horizon (~10k t.u.) is 0.2% of the 5M episode — under that
   objective a myopic oracle is near-optimal *by construction*, matching the project's own FT
   verdict ("the 5M collapse is an OBJECTIVE problem"). The defects corrupt the 5M inputs too
   (D4), but the strongest internal evidence (§D4 counter-evidence) says features alone don't
   explain the 5M ordering. The root treatments are algorithmic: γ-curriculum, then
   average-reward PPO (W3.7).

Priority order (full roadmap in Part III):

1. **Fix-first + measure-first package (P0)**: the six major defects, plus the evaluation
   statistics (CRN, power analysis, n>1 at 5M) — moved to P0 on the critic's correct argument
   that every later phase verdict is adjudicated by the current noise-dominated protocol.
2. **Pair-centric policy head + critic upgrade** (W2/W3): score the (technician, ticket) pair;
   categorical (HL-Gauss) value head; γ_t.u. sweep via the existing Hydra multirun first.
3. **Teacher-in-the-loop + asymmetric critic** (W4): keep exploiting TOPSIS and the probe API
   *during* RL, not only at BC-init.
4. **Decision-time lookahead** (W5): still the direct counter to the reward-greedy oracle, but
   demoted from "cheap" to "feasibility spike required" — the deepcopy-clone assumption was
   tested this session and **failed** (SimPy generators are unpicklable).
5. **Curriculum & scale** (W6), **memory** (W8, tempered expectations).

⚠ **Urgent, independent of this report**: defect D1 affects the **in-flight v5 run** and
compromised the v4 plateau gate. Verify on serval as soon as it is back (one `ls` — see D1).

---

## Part I — Verified defects in the current implementation

Every item confirmed by direct code reading in this session. Severity reflects expected impact.

### D1 (CRITICAL) — The cosine LR schedule reaches exactly 0 mid-run; plateau extensions train at lr=0

- `scripts/train_hydra.py:90-101` sizes `total_updates` with a hard-coded **60 t.u./decision**
  heuristic: for v4/v5 (600 eps, vec10, mean sim 275k) this gives `max(200, 134) = 200` updates.
- Measured decision density is **~24 t.u./decision** at multiscale (solo probe 2026-07-22: 8,249
  decisions / 200k t.u.), implying **~330–370 actual rounds** — and the training loop is
  episode-bounded (`runner.py:1228`, `while episodes_done < cfg.n_episodes`), not update-bounded.
- `_cosine_warmup_lr` (`ppo_transformer.py:132-140`) clamps progress at 1.0 → multiplier is
  **exactly 0.0** for every step ≥ `total_updates`; stepped once per round
  (`ppo_set_transformer.py:815`).
- **Consequence 1**: an estimated **~40% of the v4 and v5 runs trained at lr = 0** (rollouts and
  gradient computation still paid — pure waste plus a frozen policy tail).
- **Consequence 2**: `load()` restores `lr_scheduler` state (`ppo_set_transformer.py:934-937`),
  so a plateau **extension** resumes at `last_epoch ≈ 330+` inside a fresh 200-step schedule —
  **entire extensions run at lr = 0**. The plateau gate (`check_plateau.py`) then reads a flat
  last quarter: **the defect masks itself as "plateau confirmed"**. The v3 and v4 plateau
  verdicts are contaminated in proportion to how much of the last quarter was frozen.
- **Verification when serval returns**:
  `ls checkpoints/hc_v5/set_transformer_round*.pt | wc -l` (vs `total_updates=200` in the train
  log), or the `update/lr` curve of wandb run `xb7yx45l`.
- **Fix**: derive `total_updates` from measured decision density; floor the tail multiplier
  (e.g. 0.05–0.1×); re-derive or reset `last_epoch` on extension resume. ~10 lines.
- **Implication for the quality-collapse mystery (e)**: hc_v2 (2000 eps) and v3 (1200 eps) used
  the same heuristic via `train_hc_improved.py:139-146` — the "long training" suspect may partly
  be "long training at lr=0". This *shrinks* the open suspect set (see P0-abl).

### D2 (CRITICAL) — Dropout (p=0.1) is active during rollouts, inline eval, AND all benchmark evals

- No live code path ever calls `net.eval()` on the set-transformer agent: not
  `select_action/select_actions` (`ppo_set_transformer.py:405-495`), not the runner, not
  `eval_human_vs_performance.py::run_episode`. Verified by grep (only `warmstart_bc.py:213` and
  unrelated agents switch modes). Dropout 0.1 is configured (`conf/agent/set_transformer.yaml`)
  and lives in every transformer block and the cross-attention refiner.
- **Training correctness**: `old_logprob` is computed under one dropout mask at collection and
  recomputed under fresh masks in the update — importance ratios ≠ 1 at epoch 0 *before any
  gradient step*, biasing the clipped objective and inflating approx-KL (spurious early stops).
- **Evaluation**: the 5-episode inline eval AND the 24-agent benchmark numbers are stochastic
  across dropout masks despite `deterministic=True`. Every learned agent in `hvp_eval_v3/v4`
  carries this handicap; heuristics don't.
- **Fix**: `net.eval()` for acting/eval, or dropout=0 for RL training (standard practice).
  **Cheap experiment first**: re-benchmark existing checkpoints with eval-mode forwards — no
  retrain, could move all learned-agent numbers, and quantifies the damage.

### D3 (HIGH) — Within-slot feature aliasing: same-kind continuous features are interchangeable

- `_SlotFuser.forward` (`set_transformer.py:124-136`): the categorical/role embedding is **zeroed**
  at non-categorical positions (`out = x_cat * m_cat`), so the role tokens (`<RATIO:FATIGUE>` vs
  `<RATIO:MATCH>` — which `env.py:344-350`'s emitter comment says exist precisely so "each role
  gets its own learnable embedding") never reach the network. Each continuous position is encoded
  **only** by the kind-shared PLE/Time2Vec/Fourier module.
- `_pool_slot` (`set_transformer.py:418-435`) then takes a masked **mean** over the slot's
  positions, destroying positional identity.
- **Consequence**: the pooled technician representation is mathematically **invariant to permuting
  the values** of {FATIGUE, KNOW_SPEC, MATCH, MATCH_N1, MATCH_N2} (ratio kind), {ASSIGNS,
  KNOW_VOL, KNOW_MAX, KNOW_ENT} (count kind), {ETA, LAST_AGE} (time kind) — likewise for the
  machine and env streams. The network cannot distinguish a technician with MATCH=0.9/FATIGUE=0.1
  from MATCH=0.1/FATIGUE=0.9. **This is the exact discrimination the assignment task requires**,
  and ETA/MATCH were the design's compensation for having no per-pair scoring (W2).
- Plausibly contributes to (a) and to the quality dimension of (c)/(e) (repair quality is
  MATCH-driven — the least-identified feature).
- **Fix** (one of): fuse as `x_cat + continuous` instead of gating (one-line intent change);
  concat-project per position; or — slot layouts being fixed-order — flatten+MLP per slot
  (strictly more expressive, cheaper than mean-pool + attention).
- **Knock-on cost**: any fuser/head change invalidates `bc_topsis_v4` (trained through the current
  forward semantics) — budget BC re-collection into the retrain (see P2).

### D4 (HIGH, magnitude uncertain) — Long-horizon observation pathology: saturation + unscaled time + unpersisted scale

Three verified mechanisms that corrupt the 5M observation stream relative to training:

1. **Count-PLE saturates at 10k** (`hybrid_encoder.py:44-57`: top edge log1p(10 000); PLE clamps
   above). At 5M, per-tech knowledge volume is ~75k — `KNOW_VOL`/`KNOW_MAX` are **constant** over
   the entire regime where the upskilling thesis is evaluated; `PROC_TOT`, `BD_COUNT` similarly.
2. **Time2Vec has no input scaling** (`continuous_features.py:154-160`: `t·ω + φ` on raw values,
   linear first component). It consumes raw cumulative sim-time quantities — LAST_AGE, T_AGE,
   DOWNTIME (cumulative per machine), MEAN_TBF ≈ sim_time/bd_count — which at 5M are 10–25×
   beyond the training range. The linear term extrapolates arbitrarily and (pre-norm) can drown
   the rest of the slot.
3. **Fourier `input_scale` is not checkpointed and is rebuilt from the *eval* scenario**:
   plain float, not a buffer (`continuous_features.py:216`); absent from `save()`
   (`ppo_set_transformer.py:856-872`); the eval harness `setdefault`s it to the eval scenario's
   `max_sim_time` (`eval_human_vs_performance.py:273`) — **5,000,000 at very_long vs 275,000
   during v4/v5 training (200,000 for the BC student)**. The same checkpoint reads SIM_T warped
   ~18× at 5M and differently at every other scale.

**Counter-evidence that bounds the magnitude (added after the completeness critique — the draft
overclaimed here).** (i) The long-horizon fine-tune trained with a hand-matched T2V scale and
still evaluated **flat** at 5M (132.0k vs 132.4k) — direct evidence the Fourier-scale mismatch
(mechanism 3) is *not* the binding constraint on 5M throughput. (ii) gaefix — trained on the old
recipe with the **same** Time2Vec/count-saturation pathologies — is the *best learned agent at 5M*
(mean rank 9.8, 117.6k), ahead of hc_v4 (107.8k); HC-v1 (115.3k) also leads hc_v4. Since all
learned agents share D4 but rank very differently at 5M, D4 cannot carry the bulk of failure (b).
Verdict: **verified mechanisms, unquantified magnitude — measure before betting a retrain on it**:

- Cheap diagnostics (P0-eval): probe the frozen encoder with feature values at 1M/3M/5M
  timestamps and measure embedding drift vs the training range; re-run one 5M eval with the
  trained `input_scale` forced.
- Fixes regardless (they cost little and remove a real hazard): persist `sim_time_scale` in the
  checkpoint; re-express cumulative/absolute-time features as horizon-invariant relative
  quantities (rates, ages, windowed means — W2/W6 literature); extend the count-PLE edges.

### D5 (HIGH) — Checkpoint selection is noise-dominated by construction

Four verified compounding mechanisms behind failure (c) (quality 0.32 best vs 0.70 last; ranking
flips across scales):

1. Inline-eval seeds **change every round** (`runner.py:1547`: `10_000 + episodes_done*100 + i`)
   — scores across rounds are unpaired (no common random numbers).
2. The per-episode **horizon is drawn from the process-global, unseeded `np.random`**
   (`env.py:2372`; the comment "np.random is seeded by the harness" holds only for vec workers
   and the benchmark harness — the runner's `_run_episode` never seeds globals; verified by
   grep). Eval returns carry ±27% horizon luck from a stream the training process also consumes.
3. The eval env's **per-component reward normalizer accumulates statistics across all inline
   evals and is never reset** (constructed once, `env.py:395-405`; `reset()` doesn't touch it;
   `normalize_components: true` in the v5 config) — the selection signal's *scale* drifts, so
   argmax-across-rounds compares incommensurable numbers.
4. Selection is on **shaped return**, not the reported KPIs.
Plus D2: the selector's 5 episodes are dropout-stochastic on top.

- **Fixes**: fixed eval seed set + fixed eval horizon; freeze the eval normalizer
  (`freeze_reward_normalizer`, `env.py:1793`); ≥10–20 paired episodes; select on an explicit KPI
  composite — or retire argmax-best (W7).

### D6 (MEDIUM, paper-critical) — The GAE off-by-one survives in the PPO-Transformer anchor

- `ppo_transformer.py::_compute_gae` (l.634-654) still applies the terminal mask one step late
  (transition *t* consumes `dones[t+1]`, deferred update at l.652) — the bug fixed in the subclass
  on 2026-07-20. `tests/test_gae.py` covers only the subclass; the parent re-implements the loop.
- **Consequence**: terminal credit severed at n−2 in anchor training (terminal bonuses enabled),
  plus cross-episode leakage. **§7.4's "anchors rank far below" is partly artifact.**
- **Related confound the draft missed** (completeness critique + "Unraveling the Rainbow", 2025):
  the Rainbow anchor's rank 20/22 conflates *algorithm* with its flat-obs *encoder* — with
  matched graph encoders, value-based methods equal or beat PPO on JSP/FJSP. The fair §7.4
  ablation is Rainbow + the set-transformer encoder. Decide whether §7.4 claims "architecture is
  load-bearing" (defensible) or "PPO is load-bearing" (currently not).
- **Fix**: one-line reorder in the parent (or shared GAE) + retrain anchors before §7.4 is written.

### D7 (MEDIUM) — BC→PPO handoff: confident policy, random critic, no protection

- `warmstart_bc.py:236` optimises masked cross-entropy only — the value head receives no
  gradient. PPO starts with a ~74%-TOPSIS policy and a random critic: early advantages are noise
  at full LR. Nothing shields the BC prior; conversely clip 0.2 + KL-stop may lock the policy
  near TOPSIS all run. **The two failure directions have opposite remedies**, and no diagnostic
  distinguishes them. **P0-eval discriminator**: measure hc_v4/v5's action agreement with TOPSIS
  on ~1k logged decision states (same machinery as W7's churn probe). High agreement → the
  anchor-lock story (fix: earn-the-deviation schedule); low → the destroyed-prior story (fix:
  critic warm-up + KL anchor, W4).

### D8 (MEDIUM) — λ stays per-decision under γ^Δt

- `ppo_set_transformer.py:584-610`: γ discounts by sim-time, λ=0.98 per decision (by design
  comment). With mean Δt ≈ 22 t.u., the advantage estimator's credit half-life is ~28 decisions ≈
  **600–700 t.u.** — ~15× shorter than the nominal 10k-t.u. γ-horizon, and it shrinks in
  sim-time exactly when decisions are dense: the pathology the γ^Δt change was meant to remove,
  reintroduced through λ. Continuous-time TD(λ) supports λ^Δt (Doya 2000).
- **Audit item** (Bradtke & Duff 1994, added from the memory-axis survey): rewards accrued
  *between* decisions should be time-discounted within the interval (integral form) for full
  consistency with γ^Δt — check KATA's reward-accrual convention.

### D9 (MEDIUM) — Action-space semantics the draft under-weighted (obs-action audit)

- **No defer/idle action**: every decision must name a technician. When the whole fleet is
  busy/disrupted the mask falls back to all-ones (`env.py:1720-1722`) and the ticket is
  **irrevocably** bound behind an absence of unobserved remaining duration (BUSY/DISRUPT are
  binary; remaining busy/disruption time is invisible to everyone).
- **FIFO-head-only assignment**: the agent chooses *who repairs the head ticket*, never which
  ticket to serve (`env.py:701-709,730-768`). Hungarian/BatchMILP partially recover ticket-order
  optimisation through the whole-queue cost matrix — a structural asymmetry deeper than the
  2-deep lookahead gap (W2). In-scope options: expose remaining-busy-time estimates and queue
  features (observation builder); a defer action or ticket-choice action head is an env/action
  change to discuss (it changes the MDP, not the reward).

### D10 (LOW, hygiene — one cleanup pass)

- **Value-loss clipping at 0.2 in PopArt-normalised units** (`ppo_set_transformer.py:755-765`) —
  empirically unhelpful-to-harmful (Engstrom; Andrychowicz; Huang); first cheap ablation:
  `clip_eps_vf: null`.
- **Silent semi-MDP degradation**: missing `info["sim_time"]` → dt=0 → γ^0=1, undiscounted with
  no warning (`runner.py:1252-1254`, `ppo_set_transformer.py:532-537`). Add a hard assert.
- `snapshot_stream_state` misses `_last_sim_time` → dt=0 transition for worker 0 after every
  inline eval (`ppo_set_transformer.py:352-385`).
- **RoPE inside the set encoders** (`set_transformer.py:174-184`) breaks permutation invariance;
  slot positions ≥ small-fleet sizes are undertrained — a scale-transfer liability the refiner's
  own no-RoPE comment contradicts.
- **Single global PopArt across scales** (`ppo_set_transformer.py:316-327`) + **no
  N_TECHS/N_MACHINES features** (`env.py:1255-1294`) + size-invariant pooling: the critic must
  explain order-of-magnitude return differences from a nearly size-blind input.
- **Entropy bonus not normalised by valid-action count** (~2× effective-coefficient swing between
  4- and 26-tech scenarios).
- Set-mode queue lookahead **hardcoded to 2** (`env.py:1119`); the v5 JSON's
  `next_ticket_lookahead: 5` and `include_*_tokens` flags are **inert** in set mode
  (`train_hydra.py:49` forces `"set"`) — config-drift trap.
- Sentinel clamp aliasing: `-1` → 0, so "never assigned" ≡ "assigned just now" (`env.py:355-357`).
- BC optimiser wd=0.01 vs PPO wd=0; dead `n_envs` param; `normalize_rewards: true` in agent JSONs
  is a dead path.

---

## Part II — Improvement workstreams grounded in the literature

References: Part IV (verification outcomes marked ✔/△). **W1 = Part I's fixes, first.**

### W2 — Make the (technician, ticket) pair the scored object

**Targets (a), (b), (d). The strongest and most consistent architectural theme in the learned-
dispatching literature** (all refs below verified ✔):

- The current head scores technician embeddings against one pooled context via a bias-free
  rank-64 bilinear form (`set_transformer.py:536-554`); the ticket is never a first-class token;
  tech↔machine interaction happens once, one-directionally, post-pooling.
- L2D (Zhang et al., NeurIPS 2020): graph state over the disjunctive graph, size-agnostic
  dispatch beating PDRs — the size-transfer signal travels through relational structure.
- ScheduleNet (Park et al., arXiv 2021): typed agent–task graph, semi-MDP, assignment probability
  from the **(agent, task) pair embedding** with edge features in type-aware attention — the
  closest structural match to KATA.
- Song et al. (IEEE TII 2023): processing times as **edge features on operation–machine arcs**,
  score the arc — pair/edge scoring beating entity pooling for assignment, in a heterogeneous
  GNN trained with PPO.
- DAN (Wang et al., IEEE TNNLS 2024): interleaved within-set self-attention + cross-set
  attention, action scored from the fused pair — the cleanest incremental blueprint for the
  existing refiner (alternate tech↔ticket cross-attention, score `MLP([s; e_i; s⊙e_i])`).
- MatNet (Kwon et al., NeurIPS 2021): **mixed-score attention** — inject the raw per-pair scalar
  (KATA: the per-(tech, ticket) expected repair time the env already computes for baselines via
  `assignment_cost_matrix()`) as a bias inside cross-attention scores. The network starts from
  the information SPT/TOPSIS rank by and learns residual corrections. Cheapest retrofit, most
  direct connection to (a).
- Kool et al. (ICLR 2019): pointer query = current-ticket embedding + pooled context;
  **tanh-clipped logits (C≈10)** — an exploration/stability device relevant to (e).
- LEHD (Luo et al., NeurIPS 2023): spend capacity in the per-decision decoder (re-attending over
  candidates), not deeper global encoders — also the best explanation of *why* the
  set-transformer transfers scales while flat anchors can't (§7.4 story).
- SLIM (Corsini et al., NeurIPS 2024, added from critique): best-of-N self-labeling — roll out N
  complete episodes, clone the best under the *true objective* (masked-CE pipeline already
  exists). Sidesteps per-step credit assignment entirely; teaches exactly what a myopic teacher
  cannot. Robust to its own hyperparameters (per paper; the draft's "lower variance" clause was
  corrected in verification).

Observation-builder items (in scope, not reward): `N_TECHS`/`N_MACHINES` features; queue
lookahead beyond the hardcoded 2 (the whole-queue cost matrix is the informed baselines' clearest
input edge); horizon-invariant re-expression of cumulative features (BQ-NCO's stationary-tail
principle ✔, Residual Scheduling's state pruning ✔ — now IEEE Access 2024, ALiBi's
relative-not-absolute lesson ✔, Joshi et al.'s aggregation/normalisation audit ✔ — note their
*decoder* lesson actually favours autoregressive decoding; the draft had this inverted).

**Cost (revised per critique)**: days of implementation + 1 run, **plus** BC re-collection
(architecture change invalidates `bc_topsis_v4`) **plus manuscript cost**: §5.2.2 (HTT-RL) and
the §7.4 ablation plan describe the current architecture — a P2 adoption implies rewriting those
sections and re-running the architecture ablations. Real, but bounded: the paper's §6/§7 claims
are frozen pending v5 anyway.

### W3 — Critic and optimizer upgrades

**Targets (a) (critic quality is how RL beats myopia), (b) at the root, (c)/(e) stability.**

0. **Cheapest first (Andrychowicz et al., ICLR 2021 ✔ — "What Matters for On-Policy Deep
   Actor-Critic Methods?"): γ is the single most impactful hyperparameter → sweep γ_t.u.
   ∈ {0.9995, 0.9999, 0.99995} via the existing Hydra multirun** before any architecture work.
   Config-only; also the discriminating experiment between Part 0's two framings.
1. **HL-Gauss categorical value head** (Farebrother et al., ICML 2024 ✔): cross-entropy over a
   discretised return support instead of MSE — consistently better under noisy, non-stationary
   targets; support on PopArt-normalised returns. ~50 lines, no extra simulation.
2. **Per-scale value normalisation** (multi-task PopArt, Hessel et al., AAAI 2019 ✔): multiscale
   training *is* multi-task; bucket PopArt statistics by scenario scale. Lighter: return-based
   scaling (Schaul, Ostrovski, Kemaev, Borsa 2021 ✔ — author list corrected) which explicitly
   handles varying discounting.
3. **Drop value clipping** (D10; Engstrom ✔; Andrychowicz ✔; Huang ✔).
4. **λ^Δt** (Doya 2000 ✔) — one line; sensitivity λ_time ∈ {0.999, 0.9995}/t.u. Plus the
   Bradtke–Duff within-interval reward-discounting audit (D8).
5. **γ-curriculum** (OpenAI Five ✔ — γ annealed upward during training, final half-life ~5 min
   game time; the draft's "45k-step" arithmetic was corrected in verification): anneal γ_t.u.
   0.9995→0.99995 on the existing γ^Δt machinery. Caution from Amit et al. (ICML 2020 ✔):
   reduced discount acts as a regulariser with benefits in limited-data regimes — so anneal with
   data, don't start high.
6. **PPG-style auxiliary value phase** (Cobbe et al., ICML 2021 ✔): decouples policy/value
   gradient interference on the shared trunk (live (e) suspect); value tolerates much higher
   sample reuse — extra epochs on already-collected data.
7. **The principled 5M endgame: average-reward PPO** (ATRPO, Zhang & Ross, ICML 2021 ✔; APO, Ma
   et al., IJCAI 2021 ✔; Mahadevan 1996 ✔; Naik, Shariff, Yasui, Yao, Sutton 2019 ✔): the SMDP
   gain criterion is literally "reward per sim-t.u." — the benchmark metric. Estimate reward-rate
   ρ̂ (EMA), GAE on differential rewards `r − ρ̂·Δt`, differential (mean-zero) value function.
   An *algorithm* change, in scope, and the only item attacking (b)'s root rather than symptoms.
   Intermediate: TD(Δ) multi-timescale critic (Romoff et al., ICML 2019 ✔) or auxiliary multi-γ
   heads (Fedus et al. 2019 ✔).
8. **Representation-collapse diagnostic** (Moalla et al., NeurIPS 2024 ✔): log effective feature
   rank during training — cheap discriminator for the (e) suspect set; PFO only if rank decays.
9. **Time-limit handling** (Pardo et al., ICML 2018 ✔): the factory is a continuing task;
   horizon truncation should bootstrap V(s_T), not write done=1 (`ppo_set_transformer.py:526`
   conflates them). Tension flagged: the terminal bonuses are reward-design elements that make
   truncation look terminal — coordinate with the deferred reward-design phase.

### W4 — Exploit the experts and the probe API during training (no reward change)

**Targets (a), (b)-floor, (e), D7. The project owns every ingredient.** All refs ✔.

1. **Asymmetric critic** (Pinto et al., RSS 2018): probe outputs
   (`assignment_reward_estimates()`, `expected_repair_times()`, `skill_match_scores()`) into the
   **value tower only**; deployed policy untouched. Constraint (Baisero & Amato, AAMAS 2022): in
   a POMDP the privileged critic must keep the observable stream too (augment, never replace) or
   the policy gradient is biased. Hedge: dual critic with blended targets (DCRL, NeurIPS 2024).
2. **Teacher-in-the-loop anchor** (Kickstarting, Schmitt et al. 2018 — arXiv-only, correctly so;
   auto-balanced by TGRL, Shenfeld et al., ICML 2023): label PPO rollout states with TOPSIS
   (deterministic, near-free), add an annealed KL/CE term. Simultaneously the D7 fix (DAgger's
   covariate-shift correction — Ross et al., AISTATS 2011) and a floor under (b). Cheapest
   variant: DAPG-style annealed BC loss on the existing dataset (Rajeswaran et al., RSS 2018);
   VPT (NeurIPS 2022) and AlphaStar (Nature 2019) are large-scale evidence a KL-to-prior anchor
   is load-bearing. **Design constraints** (added per critique): (i) A2D (Warrington et al.,
   ICML 2021) — anchor the policy only to *observable-feature* teachers (TOPSIS/SPT); the
   reward-greedy oracle is privileged and potentially unimitable — its information goes to the
   critic or advantage weights, never action targets. (ii) Czarnecki et al. (AISTATS 2019) —
   student-controlled trajectories with teacher labels only; do **not** mix teacher-controlled
   episodes into the PPO buffer (off-policy GAE contamination — the project's own vec-training
   bug family).
3. **Advantage-weighted imitation** (AggreVaTeD, ICML 2017; AWAC 2020): the probe API gives
   per-state teacher advantages in closed form — imitate TOPSIS only where it is good; the
   principled route to *exceeding* the myopic oracle.
4. **Auxiliary privileged-prediction head** (learning-by-cheating lineage, Chen et al., CoRL
   2019; added per critique): make the *policy encoder* predict probe outputs (expected repair
   time, probe reward per candidate) as auxiliary regression targets — privileged data as
   targets, not inputs; deployable; dense self-supervised signal shaping the representation
   toward exactly what the oracle exploits. Cheap rider on any run.
5. **JSRL roll-in curriculum** (Uchendu et al., ICML 2023): TOPSIS rolls in the first (1−ρ) of
   long episodes, PPO learns the tail, ρ grows. **Compute GAE only over learner-controlled
   segments** (critique item — else cross-segment leakage re-enters one phase after D6 removes
   it).
6. **Critic-only warm-up** (D7): freeze the policy loss for the first K rounds while fitting the
   value head and PopArt statistics.
7. **Generation handoff** (QDagger / Reincarnating RL, NeurIPS 2022): distill hc_v4/v5 into v6
   with a decaying anchor instead of fresh BC each generation.

### W5 — Decision-time planning: the direct answer to the reward-greedy oracle

**Targets (a) head-on; (b) via error compounding. Demoted from "cheap" after this session's
feasibility test.** All refs ✔.

The myopic oracle is one-step lookahead with V=0. The minimal agent that dominates it is
**one-step lookahead with the learned critic**: `argmax_j r(s,a_j) + γ^Δt_j · V(s'_j)`.
Theory: Bertsekas (2022) — lookahead is a Newton step; performance is far less sensitive to
critic error than the critic itself. Bertsekas & Castañón (1999): rollout of a base policy is
no worse than the base policy on stochastic scheduling — the wrapped agent cannot suffer the
compounding errors that hand heuristics the 5M win. Tesauro & Galperin (1996) is the archetype;
Gumbel planning (Danihelka et al., ICLR 2022) and Grill et al. (ICML 2020) give
policy-improvement guarantees at 8–16 simulations; TD-MPC/2 cap useful depth at ~3–5;
OCBA (Chen et al., 2000) allocates stochastic replications; Ovacik & Uzsoy (IJPR 1994/95) is the
OR precedent; SOLO (2021) the closest end-to-end system; Hamrick et al. (ICLR 2021) warn eval
search saturates small — feed targets back via Expert Iteration (Anthony et al., NeurIPS 2017)
for the rest.

**Feasibility — tested this session, the draft's estimate was wrong**:
- `assignment_reward_estimates()` (`env.py:1560-1599`) never advances the simulator — s′ and Δt
  require a clone.
- **`copy.deepcopy(env)` fails**: `TypeError: cannot pickle 'generator' object` (live SimPy
  processes; reproduced on a real `baseline.json` env this session). A one-day deepcopy clone
  does not exist.
- Honest options: (i) **state-snapshot/reconstruct** — serialize domain state (machines'
  component virtual ages, tech states, queue contents, pending event times) and rebuild a fresh
  env with re-armed processes: real engineering, ~1–2 weeks incl. equivalence tests; the
  event-driven engine's inverse-CDF samplers need re-arming from stored hazard state.
  (ii) **Deterministic one-step approximation** — the immediate consequences of an assignment
  are largely deterministic (tech busy for ETA_j, ticket dequeued, knowledge increment): build
  s̃′ analytically in the observation builder, no sim advance, V(s̃′) as the lookahead term.
  Cheap (~days), biased (ignores in-interval stochastics), but an honest first probe of whether
  `r + γ^Δt·V` beats `r` — which is also **the sharpest available critic diagnostic** for (a)/(e).
- **Wall-clock caveat (restored from the survey)**: depth-k rollouts multiply eval cost
  ~50–100×; the one-step variant (~30 forwards/decision, batched) is near-negligible.
- **Protocol caveat**: a search-wrapped agent consumes simulator probes → it carries the ‡
  (reward-oracle-class) marker in benchmark tables, argued as digital-twin deployment. Report a
  search-budget curve (0/1/k-step vs throughput vs ms/decision); Jones (2021) predicts payoff
  grows with scale.

### W6 — Scale generalisation and curriculum

**Targets (a) industrial, (b), (c).** All refs ✔.

1. **Make industrial interior**: extend sampler bounds past the eval point (techs →~34, machines
   →~110). Interpolation works where extrapolation fails (Packer et al. 2018; Kirk et al., JAIR
   2023). Config-only. **Expect and budget a small per-scale cost from the wider distribution
   (Kirk) — do not misread it as regression** (critique item).
2. **Prioritized Level Replay** (Jiang et al., ICML 2021; Robust PLR, NeurIPS 2021): levels =
   seeded scenario configs, scored by value-loss magnitude (already computed); replay high-score
   configs. Zero extra simulation. ACCEL mutations (ICML 2022) to walk toward industrial scale;
   skip full PAIRED (curation beats a learned adversary).
3. **Template-split eval** (Procgen lesson, ICML 2020): 12+7 templates is few — held-out-template
   eval to rule out memorisation.
4. **Test-time adaptation** (EAS, ICLR 2022): pointer-logit biases adapted within a 5M episode's
   first 10–20% — rank *after* D4 fixes (the flat FT says adaptation can't fix corrupted inputs).
5. Horizon-invariance theory: Myers et al. (ICLR 2025) — a property of the representation, not
   of long-horizon exposure; matches the FT refutation.

### W7 — Evaluation, selection, and checkpoint post-processing

**Moved to P0 in the roadmap (critique): every later phase verdict depends on this.** All refs ✔.

1. **Proper common random numbers** (Glasserman & Yao 1992; L'Ecuyer et al. 2002): per-process
   RNG streams — each machine-component hazard, illness process, and the horizon draw gets its
   own generator spawned from (benchmark_seed, episode, process_id). The current single shared
   stream desynchronises at the first divergent assignment — today's seeding is CRN in name only.
   Paired comparisons then shrink CIs drastically at the same episode budget.
2. **Statistical sizing and reporting** (Agarwal et al., NeurIPS 2021 — IQM + stratified
   bootstrap CIs; Henderson, AAAI 2018; Colas et al. 2018 power analysis; Patterson et al., JMLR
   2024; Machado et al., JAIR 2018 — final-window average, not best snapshot): power-analyse the
   existing per-episode variance; **re-measure the 5M failure with n>1 before designing anything
   to fix it** (critique item — currently every 5M claim is n=1).
3. **Weight averaging — highest expected gain per GPU-hour on this axis**: SWA-in-RL (Nikishin
   et al. 2018), SWA (UAI 2018), greedy soups (ICML 2022 — shared BC-init basin, LayerNorm),
   WARP (2024 — policy-gradient-trained transformer policies merge well in weight space; note:
   REINFORCE on Gemma, not PPO — verification correction), EMA guidance (Morales-Brotons et al.,
   TMLR 2024). **PopArt caveat (critique item): value heads of different checkpoints live in
   different (μ,σ) parameterisations — average the policy path only, or carry one set of PopArt
   moments (Nikishin note); do not soup value heads naively, and do not use a souped critic for
   W5 lookahead.**
4. **Selection as best-arm identification** (Konyushkova et al., NeurIPS 2021): inline eval
   shortlists; a CRN-paired racing pass over shortlisted checkpoints picks the canonical agent;
   optionally lower-CI/CVaR selection (Chan et al., ICLR 2020).
5. **Diagnostics**: policy-churn replay (Schaul et al., NeurIPS 2022 — ~10% greedy-action flips
   per update is the deep-RL default; replay ~1k logged decision states through consecutive
   checkpoints) and snapshot logit-ensembles (ICLR 2017; ICML 2017; SUNRISE, ICML 2021) to
   discriminate estimator variance from policy multimodality in (c)/(e). The same replay
   machinery doubles as the D7 TOPSIS-agreement discriminator.

### W8 — Memory (deliberately last, tempered expectations)

Ni et al. (NeurIPS 2023 ✔) decouple memory from credit assignment: **memory architectures do not
improve long-term credit assignment** — none of this fixes (a)/(b). The real observability gaps:

1. **Sufficient-statistic tokens first** (no BPTT, near-free): running per-tech empirical repair
   statistics (what the *honest* empirical-TOPSIS baseline accumulates and the agent never
   sees); remaining-busy/disruption durations (D9); time-since-last-decay.
2. If recurrence is revived: **recurrent PPO done right** (Ni et al., ICML 2022 ✔) — stored
   initial states + burn-in (R2D2, ICLR 2019 ✔; the prior zero-BPTT GRU was R2D2's known-worst
   configuration, so that negative result says nothing), truncated BPTT 64–128, fix the
   `_inline_eval` RNN-state clobber first.
3. Endless-regime choice: GRU over GTrXL (Memory Gym, JMLR 2025 ✔: GRU decisively beats TrXL on
   endless tasks; GTrXL — ICML 2020 ✔ — only if long finite context is proven necessary).

---

## Part III — Prioritized roadmap

Revised per the completeness critique: measurement moved to P0; micro-ablation discipline made
explicit; the (e)-isolation ablation scheduled; costs and decision rules added.

**Total cost accounting**: P1–P6 imply ~5–7 training+benchmark cycles (~5–8 h + ~6 h each) on
shared serval — roughly 2–3 weeks of queue occupancy at the v4/v5 cadence, plus ~1–2 weeks of
implementation. P0/P0-eval are days and mostly local. Phases carry explicit success criteria so
losers exit early. Where a phase bundles changes, the bundle is split into a **micro-ablation**
(one cheap short-horizon run per component, baseline scale only) before the full-scale run — the
project has been burned twice (HC-v2, gaefix) by bundled attribution.

| Phase | Content | Cost | Success criterion / decision rule |
|---|---|---|---|
| **P0 (code, days)** | D1 LR fix; D2 dropout fix; D6 anchor GAE fix; D4.3 `sim_time_scale` persistence; D5 selector fixes; D10 asserts; **CRN per-process streams + power analysis (W7.1/2)** | ~2–3 days | Merged + tests; benchmark harness produces paired episodes |
| **P0-eval (no retrain)** | Re-benchmark existing checkpoints with eval-mode forwards + forced trained time-scale; **5M with n≥5 paired episodes**; D4 embedding-drift probe (1M/3M/5M); D7 TOPSIS-agreement probe; churn replay (W7.5) | hours–1 day | Quantifies D2/D4 damage; re-tests failure (b) at n>1; picks D7 remedy |
| **P0-abl (the (e) ablation)** | Single-factor runs on the shrunken suspect set {vec, PopArt, γλ, rotation} (D1 partially explains "long training"), ordered per Andrychowicz; short-horizon, baseline scale | 3–4 short runs | Quality ≥0.6 at industrial/30k in the exonerating arm |
| **P1 ("v6-clean")** | Retrain v5 recipe with P0 fixes only; **plus γ_t.u. Hydra multirun sweep (W3.0) as siblings** | 1 run + 2 cheap siblings | The honest baseline; γ sweep separates defect-story from horizon-story for (b) |
| **P2 (architecture)** | D3 fuse fix → micro-ablation; then pair-scoring head (DAN/MatNet/Kool, W2) + N_TECHS + lookahead ≥5 + horizon-invariant features; **BC re-collection**; manuscript §5.2.2/§7.4 impact accepted | ~1 wk impl + 1 run | Beats P1 at industrial by a CI-separated margin; knowledge/quality KPIs not degraded |
| **P3 (critic/algorithm)** | HL-Gauss head + per-scale PopArt + no VF clip + λ^Δt + critic warm-up | 1 run | (a) gap to greedy‡ halves at industrial, or exit |
| **P4 (teacher/critic add-ons)** | Asymmetric critic on probes + TOPSIS kickstarting anchor (per D7 diagnosis) | 1 run | Best-learned at every scale among honest agents |
| **P5 (planning spike)** | Deterministic one-step approximation s̃′ first (days); full state-snapshot `env.clone()` only if the probe pays (~1–2 wks); search-budget curve | spike, then decide | If `r + γ^Δt·V(s̃′)` ≤ `r` alone → critic indicted, reinforce P3 before any clone work |
| **P6 (horizon)** | γ-curriculum; average-reward variant if 5M (now n>1-measured) still lags; JSRL roll-ins with segment-restricted GAE | 1–2 runs | Learned agent within CI of best informed heuristic at 5M throughput with knowledge/illness held |
| **P7 (curriculum/eval polish)** | Sampler-bounds extension + PLR; weight-averaged canonical artifact (policy-only); IQM tables into §7 | days | Paper-facing robustness; expect (and don't misread) small per-scale cost from wider sampler |

**On the in-flight v5**: if D1 is confirmed (checkpoint count > 200), v5's last ~40% trained at
lr=0 and any extension was inert. The v5-vs-v4 comparison remains *internally* fair (identical
schedule bug on both sides), but neither run shows its recipe's ceiling, and both "plateau"
verdicts are unsupported. Recommendation: read the v5 result as a lower bound, then fold the
reward package into P1's clean retrain.

---

## Part IV — Reference verification

An independent adversarial pass re-checked all **114** reference entries across the seven survey
axes (existence, bibliographic correctness, fairness of the attributed finding) against
proceedings pages, dblp, arXiv, and publisher records. **Outcome: 102/114 fully clean; zero
fabricated papers; no reference required dropping.** The 12 corrections (all applied in Part II):

| Ref | Issue | Correction |
|---|---|---|
| Residual Scheduling (Ho et al.) | bib | Now IEEE Access 12:14703–14718, 2024 (DOI 10.1109/ACCESS.2024.3357969), not arXiv-only |
| Joshi et al., Learning TSP… | finding | Decoder lesson inverted: paper argues **for** autoregressive decoding as the generalising inductive bias; aggregation/normalisation and size-statistics lessons stand |
| SLIM (Corsini et al.) | finding | "Lower training variance than policy-gradient" is not in the paper; actual claim = robustness to its own hyperparameters |
| Gibbons et al., Bipartite Assignment | bib | First author is **Daniel** Gibbons |
| Liu et al., HGT dynamic FJSP (ESWA 2026) | finding | Partially confirmed — treat quantitative robustness claims cautiously |
| Return-based Scaling | bib | Authors are Schaul, Ostrovski, Kemaev, Borsa (not Hessel/van Hasselt) |
| OpenAI Five | finding | Horizon arithmetic corrected (episodes ~20k steps; γ annealed upward, final half-life ~5 min game time) |
| Naik et al. 2019 | bib | Add missing author Hengshuai Yao |
| Amit et al., Discount-as-Regularizer | finding | "Optimal γ grows with data" not shown; paper shows discount≈regularisation with limited-data benefits |
| Andrychowicz et al. | bib | ICLR 2021 title is "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study" |
| BQ-NCO | bib | Fourth author is **Arnaud** Sors |
| WARP | finding | Uses REINFORCE (Gemma 7B), not PPO — evidence class is "policy-gradient-trained transformers merge well" |

Additionally two refs are marked "probable" rather than independently confirmed at the venue
level (Dossa et al. early-stopping, IEEE Access 2021; Morales-Brotons et al., TMLR 2024) — both
uses in this report are non-load-bearing. Full per-axis verdict JSONs:
session scratchpad `wf1/VERIFY_lit_*.json`; full reference lists with applicability notes:
`wf1/lit_*.json`. These can be promoted into `paper/biblio.bib` on request.

---

## Appendix A — Code-claim verification status

| Claim | Status | Evidence |
|---|---|---|
| D1 LR multiplier exactly 0 past `total_updates`; scheduler state restored into extensions; loop episode-bounded | **Confirmed (code)** | `ppo_transformer.py:132-140`; `ppo_set_transformer.py:815,934-937`; `runner.py:1228`; `train_hydra.py:90-101` |
| D1 magnitude (~40% of v4/v5 at lr=0) | Estimated | ~24 t.u./decision measured (2026-07-22 probe) vs 60 heuristic; **serval check pending** |
| D2 no `net.eval()` in rollout/eval paths; dropout 0.1 configured | **Confirmed (code)** | grep over `src/`+`scripts/`; `conf/agent/set_transformer.yaml` |
| D3 role-embedding zeroing + masked-mean pooling → same-kind permutation invariance | **Confirmed (code)** | `set_transformer.py:124-136,418-435` |
| D4 count-PLE top edge log1p(10k); Time2Vec unscaled; Fourier `input_scale` unpersisted, eval-rebuilt | **Confirmed (code); magnitude bounded by FT/gaefix counter-evidence** | `hybrid_encoder.py:44-57`; `continuous_features.py:154-160,216`; `ppo_set_transformer.py:856-872`; `eval_human_vs_performance.py:273`; CLAUDE.md:69,78 |
| D5 per-round-varying eval seeds; unseeded global-np horizon draw; never-reset drifting eval reward normalizer | **Confirmed (code)** | `runner.py:1547`; `env.py:2372,395-405`; globals seeded only in `vec_env.py:45-46` + eval harness; `train_multiscale_v5.json:170` |
| D6 parent `_compute_gae` consumes `dones[t+1]` for transition t | **Confirmed (code)** | `ppo_transformer.py:646-652` |
| D7 BC optimises CE only (value head untouched) | **Confirmed (code)** | `warmstart_bc.py:236` |
| D8 λ per-decision under γ^Δt by design | **Confirmed (code + notes)** | `ppo_set_transformer.py:584-610`; CLAUDE.md 2026-07-30 |
| D9 no defer action; all-ones mask fallback; FIFO-head-only decisions | **Confirmed (code)** | `env.py:1720-1722,701-709,730-768` |
| Set-mode 2-ticket lookahead hardcoded; v5 token-mode obs flags inert | **Confirmed (code)** | `env.py:1119`; `train_hydra.py:49` |
| Pointer head bias-free rank-64 bilinear | **Confirmed (code)** | `set_transformer.py:536-554` |
| `assignment_reward_estimates` never advances the sim | **Confirmed (code)** | `env.py:1560-1599` |
| **`copy.deepcopy(env)` impossible on a live env** | **Confirmed (empirical, this session)** | `TypeError: cannot pickle 'generator' object` on a stepped `baseline.json` env |
| Benchmark numbers in the failure modes | From CLAUDE.md session notes (2026-07-20 audit found notes exact vs CSVs); CSVs on serval, unreachable | CLAUDE.md:68-79 |
