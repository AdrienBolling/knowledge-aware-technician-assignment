# Reproducibility settings

This document complements the reproducibility appendix (Appendix G) of the paper *Knowledge-Aware Technician Allocation: The Long-Term Impact of Technician Upskilling* (Bolling, Kubler, Ruiz-Rodríguez, and Le Traon). It gives the complete settings of the experiments, which are too long for the appendix. The published results used commit `5ce2a804c434f355addcf73aee3e934b8d94dfa5` (`5ce2a80`) of this repository, and all paths are relative to the repository root at that commit. When the launcher derives a value at run time, such as the length of the learning-rate schedule, the tables give the value that the published run used, as recorded in its checkpoint. The "Source" items under each table and paragraph give the files and the checks behind the values.

## Training hyperparameters

The two tables below give the settings of HTT-RL<sub>ref.</sub>, PO-HTT-RL, and the four fine-tunes. The PPO update uses generalized advantage estimation (GAE) and an early stop on the Kullback–Leibler (KL) divergence between the old and the new policy.

### Training settings

Training settings of HTT-RL<sub>ref.</sub>, PO-HTT-RL, and the one-lever fine-tunes (qua., fat., pro., know.). The first table gives the values that differ between the runs. The second table gives the values that apply to all runs; the behavior-cloning values apply to HTT-RL<sub>ref.</sub> and PO-HTT-RL only (— for the fine-tunes). t.u.: simulated time units.

**Values that differ between the runs**

| Group | Setting | HTT-RL<sub>ref.</sub> | PO-HTT-RL | Fine-tunes (qua., fat., pro., know.) |
|---|---|---|---|---|
| Run | Evaluation keys | `hc_v6`, `hc_v6_last` | `po_v6`, `po_v6_last` | `ft_quality`, `ft_fatigue`, `ft_protect`, `ft_gini`, each with `_last` |
| Run | Environment configuration | `train_multiscale_v5.json` | `train_multiscale_v5_po.json` | `train_multiscale_v5.json` with one lever (table *Commands of the pipeline* in Appendix G.2 of the paper) |
| Run | Initialization | behavior cloning | behavior cloning, re-collected on the RTX 4080 | final checkpoint of HTT-RL<sub>ref.</sub> |
| Run | Episodes | 600 | 600 | 100 |
| Run | Seed s | 42 | 42 | 4242 |
| Run | Evaluation during training | every 200 episodes | every 200 episodes | every 50 episodes |
| Run | Best checkpoint | evaluation after about 400 episodes (update 558 of 832) | final checkpoint (identical weights) | qua.: final checkpoint (identical weights); fat., pro., know.: evaluation after about 50 episodes |
| Optimization | Peak learning rate | 3×10⁻⁴ | 3×10⁻⁴ | 3×10⁻⁵ |
| Optimization | Warmup / length (updates) | 26 / 671 | 33 / 839 | 10 / 200; restarted at update 10 when the checkpoint is loaded |
| Optimization | Updates performed | 832 | 805 | 141 (qua.), 159 (fat.), 158 (pro.), 140 (know.) |
| Behavior cloning | Decisions, agreement | **To be confirmed by the authors:** not recorded; **To be confirmed by the authors:** 88.2% held-out agreement | 115,236; 87.9% held-out agreement | — |

**Values that apply to all runs**

| Group | Setting | Value |
|---|---|---|
| Run | Agent configuration | `run_configs/agents/set_transformer_v6.json` ([HTT-RL network](#htt-rl-network)) |
| Run | Parallel workers | 5 |
| Run | Worker seeds | worker i ∈ {0, …, 4}: first reset with seed 100s+i, later resets with seeds drawn from the worker's seeded NumPy generator; layout sampler seed 1234+1000i+s |
| Run | Episode horizon | U(2×10⁵, 3.5×10⁵) t.u., at most 25,000 decisions; one sampled layout per 5 consecutive episodes of a worker |
| Run | Evaluation during training | 5 deterministic episodes on one layout drawn with seed 4321 from the training configuration; episode seeds 10⁴+100e+j (e: episodes done, j: episode index) |
| Run | Checkpoints | every 50 episodes; best: highest mean evaluation return; last: final |
| PPO | Discount | γ = 0.9999 per t.u., applied as γ<sup>Δt</sup> with Δt the simulated time between decisions (Δt = 0 for the first transition of an episode) |
| PPO | GAE λ | 0.98 per decision |
| PPO | Rollout | 2,048 decisions per worker, 10,240 per update |
| PPO | Epochs, minibatch | 4 epochs, minibatches of 256 |
| PPO | Clip ranges | policy ratio 0.2; value 0.2 in normalized units |
| PPO | Loss coefficients | value 0.5, entropy 0.01 |
| PPO | KL target | 0.02; after each epoch, the remaining epochs are skipped when the mean approximate KL of the last 41 minibatches exceeds 1.5×0.02 (the window is `n // 256 + 1` minibatches, so after the first epoch it includes the last minibatch of the previous epoch) |
| PPO | Normalization | advantages per minibatch; value targets with PopArt (β = 0.995); return scaling off |
| PPO | Action mask | applied when acting and in the loss; mixed precision off |
| Optimization | Optimizer | AdamW, (β₁, β₂) = (0.9, 0.999), ε = 10⁻⁵, weight decay 0; gradient norm clipped at 1.0 |
| Optimization | Schedule | linear warmup, then cosine decay to a floor of 0.05× the peak; one step per update |
| Behavior cloning (HTT-RL<sub>ref.</sub> and PO-HTT-RL) | Teacher | Topsis ([Baselines](#baselines)) over the 30 technician slots |
| Behavior cloning (HTT-RL<sub>ref.</sub> and PO-HTT-RL) | Data | 25 episodes of 2×10⁵ t.u., at most 5,000 decisions each, a new layout per episode from `train_multiscale_v5.json`; sampler seed 7, episode seeds 7000+e |
| Behavior cloning (HTT-RL<sub>ref.</sub> and PO-HTT-RL) | Fit | masked cross-entropy, 3 epochs, batch 64, AdamW with learning rate 3×10⁻⁴ and other PyTorch defaults; 10% of the decisions held out |

- Source: `scripts/dgy_v6_train.sh`, `scripts/local_po_v6_queue.sh`, `scripts/dgy_v6_ft_queue.sh`, `scripts/local_ft_gini_queue.sh` (CLI overrides: episodes, `parallel_envs`, horizons, `eval_interval`, `eval_episodes`, `checkpoint_interval`, seed, `init_checkpoint`, lr 3e-5)
- Source: `conf/train.yaml` (`max_steps` 25000, `episodes_per_scenario` 5, `rollout_steps` 2048, gamma 0.9999, `time_based_discount` true, `use_popart` true, `use_gru` false, `eval_episodes` 5, `tu_per_decision` 24)
- Source: `run_configs/agents/set_transformer_v6.json` (`gae_lambda` 0.98, `clip_eps` 0.2, `clip_eps_vf` 0.2, `entropy_coef` 0.01, `value_coef` 0.5, `target_kl` 0.02, `normalize_advantages`, lr 3e-4, `weight_decay` 0, `max_grad_norm` 1.0, `n_epochs` 4, `minibatch_size` 256, `lr_min_factor` 0.05, `use_action_mask`, `use_amp` false)
- Source: `scripts/train_hydra.py` (`normalize_rewards=False` with PopArt; `rnn_type` none; `total_updates`/`warmup_updates` injection); `src/agents/ppo/ppo_set_transformer.py` (AdamW eps 1e-5 betas (0.9,0.999); `popart_beta` 0.995 default; KL stop `1.5*target_kl` per epoch; value clip in PopArt units; `dt=0` at first transition; GAE `gamma**dt`)
- Source: `src/agents/ppo/ppo_transformer.py` (`_cosine_warmup_lr` `max(min_factor, cosine)`; `_rearm_lr_schedule_if_exhausted` sets `last_epoch=warmup_updates`)
- Source: `src/experiment/vec_env.py` (sampler seed (`rcfg.seed=1234`) + `1000*i` + `base_seed`; `SeededResetWrapper` derived seeds); `src/experiment/runner.py` (`_train_loop_vec`: reset seeds `seed*100+i`, best on eval mean return, final checkpoint; `_inline_eval` seeds `10000+ep*100+i`; eval layout from `randomized_scenario.eval_seed=4321`)
- Source: schedule lengths and update counts read from the checkpoints' `lr_scheduler` state (not in the repository): `hc_v6` best `last_epoch` 558, lr 2.21517e-5 -> unique fit (671, 26); `hc_v6` final `last_epoch` 832; `po_v6` rounds 00061/00354/00540/00696 -> unique fit (839, 33), final `last_epoch` 805; `ft_*` re-armed to 10: last 151 (quality), 169 (fatigue), 168 (protect), 150 (gini); best 151/94/93/92
- Source: identical weights verified tensor by tensor: `po_v6` best == last, `ft_quality` best == last
- Source: `scripts/warmstart_bc.py` (defaults max-steps 5000, epochs 3, batch-size 64, lr 3e-4, holdout 0.1; `TopsisAgent(max_techs)`; episode seeds `seed*1000+ep`; AdamW default weight decay)
- Source: PO BC counts from run log `reports/po_v6_bc_collect.log` (local, not in the repository): 115,236 decisions, val agreement 87.9%
- Source: HTT-RL ref. BC log is not in the repository; 88.2% is taken from the project notes (not in the repository) only -> to be confirmed by the authors

### Differences between the published runs and commit `5ce2a80`

HTT-RL<sub>ref.</sub> was trained with an earlier version of `scripts/train_hydra.py`, which sized the cosine schedule to 671 updates with 26 warmup updates. At `5ce2a80`, the launcher multiplies this estimate by 1.25, so the command of table *Commands of the pipeline* (Appendix G.2 of the paper) gives 839 and 33, the schedule of PO-HTT-RL; the override `tu_per_decision=30` restores 671 and 26. The commit that added this factor (`73e5f8d`) also corrected the rotation of the training layouts: before it, each block of five episodes of a worker contained four episodes of one layout and one episode of the next. HTT-RL<sub>ref.</sub> was trained before this correction and PO-HTT-RL, the know. fine-tune, and the MLP anchors after it, so their sequences of training layouts are not identical. For PO-HTT-RL, the behavior-cloning data were re-collected with the same command and seed on the RTX 4080. Training is not bit-reproducible, because the learner process samples actions with the unseeded PyTorch generator.

- Source: `git show 62819a5:scripts/train_hydra.py` (`est_rounds` without the 1.25 factor -> `int(600/5*(275000/24)/2048)=671`, warmup `max(10,671//25)=26`); `73e5f8d` message and diff ("1.25 schedule inflation"; "scenario-rotation off-by-one fixed in BOTH training env paths")
- Source: `tu_per_decision=30` -> `int(1.25*600/5*(275000/30)/2048) = int(671.39) = 671`, warmup 26
- Source: `src/experiment/runner.py` `_build_env` comment ("every k-block straddles two factories (k-1 episodes of one scenario + 1 of the next)")
- Source: `scripts/local_ft_gini_queue.sh` added in `a380c7d` (after `73e5f8d`); `train_po_v6` at `b2eaa3e` (after); trad queue added in `73e5f8d`. Launch commit of the qua./fat./pro. fine-tunes is not recorded (queue `6be1f57` predates `73e5f8d` by 2 h).
- Source: `src/agents/ppo/ppo_set_transformer.py` `select_actions`: `dist.sample()` (torch RNG); no `torch.manual_seed` call in runner/`train_hydra`

## HTT-RL architecture

The table below gives the network of HTT-RL (Section 5.2.2 of the paper). The same architecture is used by HTT-RL<sub>ref.</sub>, the fine-tunes, PO-HTT-RL, and the behavior-cloning checkpoints.

### HTT-RL network

HTT-RL network (`set_transformer_v6.json` and the defaults of `src/agents/networks/set_transformer.py`).

| Part | Setting |
|---|---|
| Input streams | technicians: 30 slots of 16 positions; machines: 100 slots of 12 positions; global stream: 16 positions |
| Vocabulary | 152 tokens (`run_configs/vocab/set_vocab.json`), one embedding table for all streams |
| Width | d<sub>model</sub> = 128, 4 attention heads |
| Ratio encoder | piecewise-linear encoding, 10 equal bins on [0, 1], MLP 10 → 128 → 128 (GELU) |
| Count encoder | piecewise-linear encoding of log(1+x), bin edges at x ∈ {0, 1, 2, 5, 10, 20, 50, 100, 500, 10³, 10⁴}, MLP 10 → 128 → 128 (GELU) |
| Recent-time encoder | Time2Vec with 1 linear and 16 sine terms (learnable), linear 17 → 128, raw time as input |
| Long-horizon time encoder | random Fourier features, 16 fixed frequencies drawn from N(0, 1), linear 32 → 128; the input is divided by the horizon given at construction: 2×10⁵ (behavior cloning), 2.75×10⁵ (PPO), the scenario horizon (evaluation) |
| Slot fusion | role binding on: token embedding kept at numeric positions, residual MLP 128 → 128 → 128 per position; mean over non-padding positions |
| Feature context | on: per position, mean over the technician slots; linear 16 × 128 → 128 |
| Cross-slot encoders | one per technician and machine stream; 3 pre-norm blocks with RMSNorm, self-attention with rotary position embeddings over the slot index, SwiGLU feed-forward of width 320, and a learnable summary token |
| Global stream | mean pooling, MLP 128 → 128 → 128 (GELU) |
| Cross-attention | 1 pre-norm block, 4 heads: technician slots attend to the machine slots and the global summary; feed-forward of width 320 (GELU); no position encoding; the technician summary becomes the mean of the refined slots |
| Context projection | 4 summaries concatenated (4 × 128), RMSNorm, MLP 512 → 256 → 128 (GELU) |
| Pointer head | d<sub>attn</sub> = 64, scaled dot product, orthogonal initialization with gain 0.01, masked slots at −∞ |
| Value head | MLP 128 → 256 → 1 (GELU), PopArt output rescaling |
| Recurrence, dropout | none; 0 |
| Parameters | 1,885,923 |
| Checkpoint flags | `use_popart`, `slot_role_binding`, `use_feature_context`: true; `rnn_type`: none; `tech_slot_length`: 16. No `numeric_encoding`, `cross_slot`, or `set_positional` entry, so the evaluation uses the defaults hybrid, attention, and true |

- Source: `run_configs/agents/set_transformer_v6.json` (`d_model` 128, `n_heads` 4, `n_layers` 3, dropout 0.0, `value_hidden` 256, `pointer_d_attn` 64, `rnn_type` none, `slot_role_binding` true, `use_feature_context` true)
- Source: `src/kata/core/config.py` `GymEnvConfig` defaults (`max_techs` 30, `max_machines` 100, `set_tech_slot_length` 16, `set_machine_slot_length` 12, `set_env_length` 16, `set_vocab_path`); `run_configs/vocab/set_vocab.json` `n_tokens` 152
- Source: `src/agents/networks/set_transformer.py` (`d_ff` default `round((8*128/3)/64)*64 = 320`; `n_time2vec_freqs` 16; `n_fourier_freqs` 16; `fourier_sigma` 1.0; `input_scale=sim_time_scale`; `_SlotFuser` binder; `feature_context_proj`; `_SetEncoder` `use_rope=set_positional`; `_CrossAttentionRefiner`; `_ContextProjection` `2*d_out` hidden; `PointerActionHead` gain 0.01; value head)
- Source: `src/agents/networks/hybrid_encoder.py` (`DEFAULT_RATIO_EDGES`, `DEFAULT_COUNT_LOG_EDGES`); `src/agents/networks/continuous_features.py` (PLE hidden = `d_model`, GELU; Time2Vec `n_freqs+1`; Fourier `2*n_freqs`, `B ~ N(0, sigma^2)`); `src/agents/networks/modern_transformer.py` (RMSNorm, RoPE base 10000, SwiGLU)
- Source: `sim_time_scale = gym.max_sim_time`: `scripts/warmstart_bc.py` (`--sim-time 200000`), `src/experiment/runner.py` `_build_agent` (`train_hydra` sets `max_sim_time = sim_time = 275000`), `scripts/eval_human_vs_performance.py` `build_agents` (`env_cfg.gym.max_sim_time` of the scenario)
- Source: parameter count from `SetTransformerAgent(**set_transformer_v6 params, vocab_size 152, use_popart).num_parameters() = 1885923`; `state_dict` keys and shapes identical to `checkpoints/hc_v6_final/set_transformer_best.pt`
- Source: improvements dict of `hc_v6`, `po_v6`, `ft_*` checkpoints: `{'use_popart': True, 'rnn_type': 'none', 'rnn_hidden': 128, 'slot_role_binding': True, 'use_feature_context': True, 'tech_slot_length': 16}` (+ PopArt stats); eval defaults in `build_agents`

## MLP anchors

The two tables below give the settings of the three MLP anchors of Appendix B.2 of the paper. They consume the set observation of HTT-RL, flattened into one vector (`src/agents/networks/mlp_encoder.py`).

### MLP anchor settings

Settings of the MLP anchors (`src/agents/a2c/a2c_mlp.py`, `src/agents/grpo/grpo_mlp.py`, `src/agents/dqn/dql_mlp.py`). The first table gives the values that differ between the anchors. The second table gives the values that apply to all anchors.

**Values that differ between the anchors**

| Setting | A2C-MLP | GRPO-MLP | DDQN-MLP |
|---|---|---|---|
| Evaluation keys | `a2c_mlp`, `a2c_mlp_last` | `grpo_mlp`, `grpo_mlp_last` | `dql_mlp`, `dql_mlp_last` |
| Agent configuration | `run_configs/agents/a2c_mlp.json` | `run_configs/agents/grpo_mlp.json` | `run_configs/agents/dql_mlp.json` |
| Heads, parameters | policy (30) and value (1); 2,945,567 | policy (30); 2,945,054 | Q-values (30); 2,960,414 |
| Discount | 0.997 per decision | 1 (undiscounted episode return) | 0.997 per decision |
| Update | GAE with λ = 0.95; one full-batch gradient step per rollout; value coefficient 0.5; entropy 0.01; advantages normalized; rewards divided by the running standard deviation of the discounted return | group of 8 complete episodes on one layout; advantage: z-score of the episode return within the group; clipped surrogate with ε = 0.2; 4 epochs; minibatches of 256; entropy 0.01; no KL target | double Q-learning target with Huber loss; uniform replay of 500,000 transitions, learning from 10,000; one gradient step every 8 decisions; batch 64; target network copied every 2,000 gradient steps |
| Exploration | sampling from the masked policy | sampling from the masked policy | masked ε-greedy, ε from 1.0 to 0.05 over 500,000 decisions |
| Rollout | 128 decisions per worker | 8 episodes | — |
| Optimizer | AdamW, 7×10⁻⁴, ε = 10⁻⁵, weight decay 0 | AdamW, 3×10⁻⁴, ε = 10⁻⁵, weight decay 0 | Adam, 10⁻⁴ |
| Schedule (warmup / length) | 537 / 13,427 updates, cosine, floor 0.05× | 3 / 75 updates, cosine, floor 0.05× | constant |
| Gradient clipping | 0.5 | 1.0 | 10.0 |
| Parallel workers | 5 | 1 | 1 |
| Episode horizon | U(2×10⁵, 3.5×10⁵) t.u. | 2.75×10⁵ t.u. (fixed) | U(2×10⁵, 3.5×10⁵) t.u. |
| Layout rotation | every 5 episodes | every 8 episodes | every 5 episodes |

**Values that apply to all anchors**

| Setting | Value |
|---|---|
| Input | 5,204 features: a one-hot vector per categorical position from the vocabulary of the [HTT-RL network](#htt-rl-network), a symmetric logarithm for counts and times, raw ratios; DDQN-MLP appends the 30 action-mask bits (5,234 features) |
| Network | 2 hidden layers of 512 units, each Linear–LayerNorm–ReLU; linear heads |
| Common | 600 episodes from random initialization; environment `train_multiscale_v5_mlp.json`, which draws the fleet size uniformly from {4, …, 30}; seed 42 (A2C-MLP: worker seeds as in the [training settings](#training-settings); GRPO-MLP and DDQN-MLP: episode seeds 42+e, layout sampler seed 1234; DDQN-MLP also seeds its exploration and replay generators and its PyTorch generator with 42; the launcher does not pass the seed to A2C-MLP and GRPO-MLP, so their network initialization and action sampling use an unseeded PyTorch generator); at most 25,000 decisions per episode; evaluation during training every 200 episodes (5 episodes); checkpoints every 50 episodes, best and last as in the [training settings](#training-settings) |

- Source: `run_configs/agents/{a2c_mlp,grpo_mlp,dql_mlp}.json` (`hidden_sizes` [512,512]; A2C gamma 0.997, `gae_lambda` 0.95, entropy 0.01, value 0.5, `normalize_advantages`, lr 7e-4, wd 0, `max_grad_norm` 0.5, `rollout_steps` 128, `lr_min_factor` 0.05, `normalize_rewards` true; GRPO `group_size` 8, gamma 1.0, clip 0.2, entropy 0.01, `n_epochs` 4, minibatch 256, `target_kl` null, lr 3e-4, wd 0, `max_grad_norm` 1.0, `lr_min_factor` 0.05; DQL lr 1e-4, gamma 0.997, batch 64, `max_grad_norm` 10, buffer 500000, `min_replay` 10000, `train_freq` 8, `target_update_freq` 2000, eps 1.0->0.05 over 500000, store float16)
- Source: `scripts/train_hydra.py` agent branches: a2c `total_updates = max(200, int(1.25*600/5*(275000/24)/128)) = 13427`, warmup `13427//25 = 537`; grpo total = `max(4, 600//8) = 75`, warmup `max(3, 75//25) = 3`, `episodes_per_scenario` pinned to 8, horizon range popped (fixed `sim_time` 275000); dql seed `setdefault(42)`
- Source: `scripts/dgy_trad_baselines_queue.sh` (`parallel_envs` 5/1/1, seed 42, `eval_interval` 200, `eval_episodes` 5, `checkpoint_interval` 50, `env=train_multiscale_v5_mlp`); `run_configs/benchmark_suite/train_multiscale_v5_mlp.json` (`n_technicians_min` 4, max 30, no ratio)
- Source: `src/agents/networks/mlp_encoder.py` (`SetObsFlattener`: one-hot per categorical position from set vocab, symlog for COUNT/TIME/FOURIER kinds, raw RATIO; `MLPTrunk` Linear-LayerNorm-ReLU); `out_dim` 5204 and parameter counts from instantiating the three agents with their JSON params; DQL appends `max_techs` mask bits (2960414 params = 5234 inputs)
- Source: `src/agents/a2c/a2c_mlp.py` docstring (one epoch, one full batch; AdamW eps 1e-5); `src/agents/grpo/grpo_mlp.py` docstring (group of complete episodes, z-scored undiscounted return, AdamW eps 1e-5); `src/agents/dqn/dql_mlp.py` (Double DQN, `smooth_l1_loss`, Adam constant lr, private RNG streams seeded by seed, eps over env steps)
- Source: `src/experiment/runner.py` serial loop: episode seed `cfg.seed + ep`, sampler seed `rcfg.seed` (1234)

## Baseline parameters

The table below gives the rule and the parameters of each baseline. None of the baselines is trained.

### Baselines

Baselines (`src/agents/baselines/heuristics.py`). Every rule chooses among the technicians that the action mask offers, and ties go to the lowest index. c\*<sub>time</sub> is the expected repair time of Eq. (A.4) of the paper and 1−m<sub>k</sub> the knowledge match, both read from the simulator.

| Baseline | Key | Rule and parameters |
|---|---|---|
| Topsis | `topsis` | TOPSIS closeness on three cost criteria: c\*<sub>time</sub>, fatigue, and assignments so far in the episode; vector normalization; weights 0.5, 0.3, 0.2 |
| Spt | `shortest_processing` | lowest c\*<sub>time</sub> |
| ReserveSpec | `reserve_specialist` | lowest 1−m<sub>k</sub> among the technicians with c\*<sub>time</sub> ≤ τ min c\*<sub>time</sub>, τ = 1.5 |
| LeastFatigued | `least_fatigued` | lowest fatigue |
| ShortestQueue | `shortest_queue` | lowest fatigue (same rule as LeastFatigued, separate key) |
| TrainWeakest | `train_weakest` | lowest 1−m<sub>k</sub> |
| RoundRobin | `round_robin` | next index in cyclic order, from index 0 at each episode |
| LeastBusy | `least_busy` | lowest index (deterministic variant) |
| Random | `random` | uniform draw from the NumPy global generator |
| Hungarian | `optimal_assignment` | linear-sum assignment (SciPy) of all open tickets, current and queued, to the available technicians on c\*<sub>time</sub>; the technician matched to the current ticket is assigned |
| GreedyReward | `greedy_reward` | highest immediate reward estimate under the reward of the scenario configuration (Appendix C of the paper) |

- Source: `src/agents/baselines/heuristics.py`: `_available` (`action_mask`), `RandomAgent` (`np.random.choice`), `RoundRobinAgent` (`on_episode_start` resets `_next=0`), `LeastBusyAgent` (`avail[0]` if deterministic), `LeastFatiguedAgent`/`ShortestQueueAgent` (argmin `technician_fatigue`), `ShortestProcessingTimeAgent` (argmin expected repair), `OptimalAssignmentAgent` (scipy `linear_sum_assignment` on `env.assignment_cost_matrix`: current + queue x techs, inf -> big-M), `TopsisAgent` (weights (0.5, 0.3, 0.2); criteria repair, fatigue, `assignment_counts`; vector norm; closeness), `GreedyRewardAgent` (argmax `assignment_reward_estimates`), `TrainWeakestAgent` (argmin `skill_match_scores`), `ReserveSpecialistAgent` (tau 1.5)
- Source: `src/kata/env.py` `expected_repair_times` (`compute_repair_time = base * m_k * m_f`), `skill_match_scores` (`1 - m_k`), `assignment_cost_matrix`; `scripts/eval_human_vs_performance.py` (`deterministic=True`, structured obs, `attach_env`)

## Environment and templates

The [simulator settings](#simulator-settings) table gives the simulator settings shared by all training and evaluation configurations. The [technician archetypes](#technician-archetypes) and [machine templates](#machine-templates) tables give the technician and machine templates, the [training and evaluation worlds](#training-and-evaluation-worlds) table the sampling of training and evaluation layouts, and the [event schedule of S5](#event-schedule-of-s5) table the event schedule of S5. The initial grid of each archetype holds synthetic repairs spread over the whole grid (`expert`, `trainee`), uniformly (`generalist`, `junior`), 70% uniformly and 30% near the mechanical cluster (`senior`), or 85% near the motor or electronics neighborhood (`motor_specialist`, `electronics_specialist`).

### Simulator settings

Simulator settings shared by all configurations of `run_configs/benchmark_suite/` used in the paper. t.u.: simulated time units; N(μ, σ): normal distribution.

| Setting | Value |
|---|---|
| Travel time | Δ<sub>travel</sub> = 15 t.u. per assignment |
| Knowledge multiplier | global f = 0.3, α = 0.15; failure-wise overrides on; every component of the [machine templates](#machine-templates) table sets both f<sup>⋆</sup> and α<sup>⋆</sup> |
| Fatigue multiplier | exponential, α<sub>f</sub> = 0.5 |
| Fatigue increase | Eq. (A.2) of the paper at the end of each repair, with w the base repair time c<sub>time</sub> of the component (integer part) |
| Knowledge decay | on; one decay of each technician's grid every 5,000 t.u., with the decay rule of the pinned ONGOING revision |
| Imperfect repair | Kijima type II, α<sub>r</sub> = 0.25 for every component |
| Knowledge grid | 10 × 10; ticket positions from `run_configs/embeddings/ticket_grid_som.json`: self-organizing map on the 8 × 8 interior, fit seed 2, 31 failure keys; hash fallback on the border ring |
| Injury | Poisson process, rate 10⁻⁴ per t.u.; duration N(240, 60); preemptive |
| Exhaustion | check every 60 t.u., fires with probability min(1, 10⁻³ · F<sub>i</sub> · 60); duration N(120, 30); preemptive |
| Vacation | first at U(0, 8000) t.u., then every 8000 + U(−400, 400) t.u.; duration N(480, 120); waits for the current repair |
| Durations | a non-positive draw is replaced by the mean |
| Weibull components | idle hazard 0.1× the working hazard (thinning of candidates drawn at the working hazard); the failure clocks stop while the machine is broken |
| Bernoulli components | per-t.u. probabilities converted to rates, simulated event by event |
| Product flow | one source, one product every 10 t.u.; machine input and output buffers of capacity 50; routing buffers unbounded |
| Observation | learned agents: set representation, at most 30 technicians and 100 machines; baselines: structured representation; `next_ticket_lookahead` 5; action mask on |
| Metrics | rolling MTTR over the last 50 repairs |

- Source: `run_configs/benchmark_suite/{train_multiscale_v5,train_multiscale_v5_po,train_multiscale_v5_mlp,small_scale,baseline,massive_scale,lifecycle}.json`: `sim.*` identical (checked by loading with `KATAConfig` and diffing): `technicians.travel_time` 15, `fatigue_model` exponential, `fatigue_alpha` 0.5; `repair.min_repair_fraction` 0.3, `knowledge_sensitivity` 0.15, `failure_wise_knowledge_parameters` true, `default_restoration_alpha` 0.25; disruptions injury (random, rate 1e-4, mu 240, sig 60, preemptive), exhaustion (fatigue, coefficient 1e-3, poll 60, mu 120, sig 30, preemptive), vacation (periodic, interval 8000, jitter 400, mu 480, sig 120, not preemptive)
- Source: `gym.knowledge_decay_enabled` true, `knowledge_decay_interval` 5000; `ticket_embedding_path`; `observation_representation` set (`train_hydra`, harness); `max_techs` 30, `max_machines` 100 (defaults); `next_ticket_lookahead` 5; `expose_action_mask` true; `mask_unavailable_technicians` default true; `mttr_rolling_window` 50 (default)
- Source: `src/kata/entities/technicians/GymTechnician.py` (`_fatigue_disruption_loop` `p = min(1, coef*fatigue*poll)`; `_periodic_disruption_loop` initial `U(0, interval)`, jitter `U(-j, j)`; `_sample_duration` non-positive -> mu; `repair_finished`: `_increase_fatigue(int(request.get_repair_time()))` = component base repair time; `get_fatigue_multiplier` `exp(alpha*F)`; `decay_knowledge` -> ongoing `KnowledgeGrid.decay_knowledge`)
- Source: `src/kata/env.py` `_maybe_decay_knowledge`; `src/kata/entities/requests/RepairRequest.py` `get_repair_time` (component base)
- Source: `run_configs/embeddings/ticket_grid_som.json` (`grid_shape` [10,10], method "SOM (8x8 interior ...)", `fit_seed` 2, 31 placements); `src/kata/entities/encoder/precomputed.py` (hash fallback projected to border ring)
- Source: `src/kata/features/breakdown/simple_breakdown.py` (Weibull `IDLE_HAZARD_FACTOR` 0.1; Simple `_rates` `-log(1-p)/dt`; Kijima II repair `age*alpha`); `src/kata/entities/machines/complex_machine.py` (clocks advance while machine is up); `src/kata/scenario.py` (Buffer capacity 50 for machine in/out; route/type/sink unbounded; Source `interarrival_time` 10.0; restoration alpha global when component unset); `src/kata/entities/sources/source.py` (fixed timeout)

### Technician archetypes

Technician archetypes (`technician_templates.json` in `src/kata/resources/templates`). ρ: diminishing-returns parameter (`knowledge_learning_rate`); λ, μ: fatigue accumulation and recovery rates; σ<sub>prop</sub>: diffusion bandwidth; Trans.: `knowledge_transmission_factor`; Init.: synthetic repairs in the bundled initial grid. All grids are 10 × 10.

| Archetype | ρ | λ | μ | σ<sub>prop</sub> | Trans. | Init. |
|---|---:|---:|---:|---:|---:|---:|
| `expert` | 0.51 | 0.005 | 0.08 | 1.5 | 0.7 | 1,500 |
| `senior` | 0.6 | 0.005 | 0.08 | 1.5 | 0.7 | 500 |
| `generalist` | 0.7 | 0.01 | 0.05 | 1.0 | 0.5 | 200 |
| `junior` | 0.85 | 0.02 | 0.03 | 0.5 | 0.3 | 40 |
| `trainee` | 0.95 | 0.025 | 0.025 | 0.3 | 0.2 | 5 |
| `motor_specialist` | 0.6 | 0.01 | 0.05 | 1.0 | 0.5 | 600 |
| `electronics_specialist` | 0.6 | 0.01 | 0.05 | 1.0 | 0.5 | 600 |

- Source: `src/kata/resources/templates/technician_templates.json` (`knowledge_learning_rate`, `fatigue_lambda`, `fatigue_mu`, `knowledge_propagation_sigma`, `knowledge_transmission_factor`, `knowledge_k_shape` [10,10], `initial_knowledge_grid_path`)
- Source: `src/kata/resources/technician_profiles/profiles.txt` (`n_tickets` 1500/500/200/40/5/600/600 and placement descriptions)

### Event schedule of S5

Event schedule of S5 (`gym.lifecycle_events` in `lifecycle.json`). A retirement removes the active technician with the largest knowledge volume. A replacement removes the machine with the most breakdowns after its current repair ends and adds a machine of the first template, in alphabetical order, with the same machine type.

| Time (10⁶ t.u.) | Event | Template or selection | Count |
|---:|---|---|---:|
| 0.80 | retire technicians | largest knowledge | 2 |
| 0.82 | add technicians | `trainee` | 2 |
| 1.50 | add machines | `cnc_weibull` | 2 |
| 2.00 | replace machines | most breakdowns | 3 |
| 2.50 | retire technicians | largest knowledge | 2 |
| 2.52 | add technicians | `junior` | 2 |
| 3.20 | add technicians | `expert` | 1 |
| 3.80 | replace machines | most breakdowns | 2 |
| 4.20 | retire technicians | largest knowledge | 1 |
| 4.22 | add technicians | `trainee` | 1 |

- Source: `run_configs/benchmark_suite/lifecycle.json` `gym.lifecycle_events` (time, kind, count, template, select); `src/kata/env.py` `_execute_lifecycle_event` (replace waits while broken; template None -> `_template_for_machine_type`: first of sorted `list_templates` with same `machine_type`), `_select_technician_for_event` (highest volume among non-retired), `_select_machine_for_event` (`most_breakdowns`)

### Machine templates

Machine templates and their components (`src/kata/resources/templates/machine_templates.json`). The first three columns give the template name, the machine type, and the process time (t.u.). Model W: Weibull with shape β and scale η (t.u.); model B: Bernoulli with failure probabilities per t.u. when working (p<sub>w</sub>) and idle (p<sub>i</sub>). c<sub>time</sub>: base repair time (t.u.); f<sup>⋆</sup>, α<sup>⋆</sup>: overrides of the knowledge multiplier (Eq. (A.1) of the paper). The 35 components give 31 failure keys (machine type, component family).

| Template | Machine type | Process time | Component | Family | Model | β or p<sub>w</sub> | η or p<sub>i</sub> | c<sub>time</sub> | f<sup>⋆</sup> | α<sup>⋆</sup> |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|
| `cnc_weibull` | CNC | 200 | `spindle_0` | spindle | W | 2.5 | 800 | 480 | 0.2 | 0.075 |
| `cnc_weibull` | CNC | 200 | `coolant_pump_0` | pump | W | 3 | 1,200 | 180 | 0.4 | 0.225 |
| `cnc_weibull` | CNC | 200 | `main_drive` | drive | W | 3 | 200,000 | 48,000 | 0.6 | 0.03 |
| `assembly_mixed` | Assembly | 120 | `motor_0` | motor | W | 2.5 | 800 | 240 | 0.35 | 0.15 |
| `assembly_mixed` | Assembly | 120 | `bearing_0` | bearing | B | 2·10⁻³ | 2·10⁻⁴ | 35 | 0.55 | 0.45 |
| `assembly_robot` | Assembly | 80 | `servo_motor_1` | motor | W | 3 | 3,000 | 300 | 0.3 | 0.12 |
| `assembly_robot` | Assembly | 80 | `gripper` | end effector | B | 4·10⁻³ | 4·10⁻⁴ | 35 | 0.45 | 0.3 |
| `assembly_robot` | Assembly | 80 | `controller` | electronics | B | 10⁻³ | 10⁻⁴ | 360 | 0.25 | 0.09 |
| `assembly_robot` | Assembly | 80 | `vendor_pcb` | electronics | B | 10⁻⁵ | 10⁻⁶ | 48,000 | 0.8 | 0.015 |
| `conveyor` | Conveyor | 100 | `belt_motor` | motor | W | 1.5 | 5,000 | 200 | 0.35 | 0.18 |
| `conveyor` | Conveyor | 100 | `belt` | mechanical | B | 5·10⁻³ | 5·10⁻⁴ | 30 | 0.6 | 0.75 |
| `welder` | Welder | 150 | `welding_torch` | torch | W | 2 | 1,500 | 120 | 0.25 | 0.12 |
| `welder` | Welder | 150 | `cooling_unit` | pump | W | 3 | 2,500 | 180 | 0.4 | 0.225 |
| `welder` | Welder | 150 | `power_supply` | electronics | B | 8·10⁻⁴ | 10⁻⁴ | 360 | 0.2 | 0.06 |
| `inspection` | Inspection | 60 | `camera` | sensor | B | 3·10⁻³ | 3·10⁻⁴ | 30 | 0.55 | 0.6 |
| `inspection` | Inspection | 60 | `sensor_array` | sensor | B | 2·10⁻³ | 2·10⁻⁴ | 120 | 0.45 | 0.3 |
| `inspection` | Inspection | 60 | `controller` | electronics | B | 10⁻³ | 10⁻⁴ | 240 | 0.25 | 0.09 |
| `press` | Press | 180 | `hydraulic_pump` | pump | W | 2.2 | 4,000 | 420 | 0.3 | 0.15 |
| `press` | Press | 180 | `ram` | mechanical | W | 3.5 | 8,000 | 240 | 0.45 | 0.27 |
| `press` | Press | 180 | `die` | mechanical | B | 4·10⁻³ | 3·10⁻⁴ | 90 | 0.55 | 0.525 |
| `lathe` | Lathe | 140 | `spindle_l` | spindle | W | 2.5 | 1,500 | 360 | 0.25 | 0.12 |
| `lathe` | Lathe | 140 | `tool_holder` | end effector | B | 3·10⁻³ | 3·10⁻⁴ | 60 | 0.5 | 0.375 |
| `lathe` | Lathe | 140 | `chuck` | mechanical | B | 2.5·10⁻³ | 2·10⁻⁴ | 75 | 0.55 | 0.45 |
| `grinder` | Grinder | 90 | `grinding_wheel` | mechanical | W | 1.8 | 1,000 | 45 | 0.6 | 0.675 |
| `grinder` | Grinder | 90 | `spindle_g` | spindle | W | 2.8 | 2,500 | 420 | 0.2 | 0.09 |
| `grinder` | Grinder | 90 | `dust_collector` | sensor | B | 5·10⁻³ | 5·10⁻⁴ | 90 | 0.5 | 0.45 |
| `mill` | Mill | 160 | `spindle_m` | spindle | W | 2.4 | 1,800 | 480 | 0.22 | 0.105 |
| `mill` | Mill | 160 | `coolant_pump_m` | pump | W | 3 | 1,100 | 180 | 0.4 | 0.225 |
| `mill` | Mill | 160 | `tool_changer` | end effector | B | 3·10⁻³ | 3·10⁻⁴ | 120 | 0.45 | 0.33 |
| `furnace` | Furnace | 240 | `heating_element` | electronics | W | 4 | 12,000 | 600 | 0.15 | 0.075 |
| `furnace` | Furnace | 240 | `temp_sensor` | sensor | B | 1.5·10⁻³ | 1.5·10⁻⁴ | 60 | 0.55 | 0.45 |
| `furnace` | Furnace | 240 | `insulation` | mechanical | W | 5 | 25,000 | 360 | 0.3 | 0.3 |
| `packager` | Packager | 75 | `drive_motor` | motor | W | 2.5 | 2,200 | 210 | 0.35 | 0.18 |
| `packager` | Packager | 75 | `pneumatic_arm` | end effector | B | 4·10⁻³ | 4·10⁻⁴ | 90 | 0.5 | 0.375 |
| `packager` | Packager | 75 | `label_sensor` | sensor | B | 5·10⁻³ | 5·10⁻⁴ | 45 | 0.55 | 0.525 |

- Source: `src/kata/resources/templates/machine_templates.json`, the 12 templates listed in `randomized_scenario.machine_templates` of `train_multiscale_v5.json` (rows generated from the JSON: `component_type`, `base_repair_time`, `breakdown_model`, `weibull_breakdown.shape`/`scale`, `simple_breakdown.failure_prob_working`/`idle`, `min_repair_fraction`, `knowledge_sensitivity`)
- Source: 31 keys = placements in `run_configs/embeddings/ticket_grid_som.json`; the templates generic/drill/paint are not used by any configuration of the paper

### Training and evaluation worlds

Training and evaluation worlds. Configuration files are in `run_configs/benchmark_suite/`. A technician count drawn from a ratio is round(n<sub>m</sub> · U(0.15, 0.30)), at least 1, with n<sub>m</sub> the machine count. The route visits distinct machine types, and its length is limited by the number of types in the layout. The evaluation layout is the layout that `RandomScenarioSampler` draws once with seed 20260722; all agents and episodes use it.

| Setting | Training (HTT-RL runs) | Training (MLP anchors) | S1 Small | S2 Baseline | S3 Industrial | S4 Very-long | S5 Lifecycle |
|---|---|---|---|---|---|---|---|
| Scenario key | — | — | `small_scale` | `baseline` | `massive_scale` | `very_long` | `lifecycle` |
| Configuration | `train_multiscale_v5.json`, `train_multiscale_v5_po.json` | `train_multiscale_v5_mlp.json` | `small_scale.json` | `baseline.json` | `massive_scale.json` | `massive_scale.json` | `lifecycle.json` |
| Machines | 4–100 | 4–100 | 3–5 | 12–22 | 80–100 | as S3 | 70–80 |
| Technicians | ratio | 4–30 | 2 | 4 | 30 | as S3 | 24 |
| Machine templates | all 12 | all 12 | `cnc_weibull`, `assembly_mixed` | `cnc_weibull`, `assembly_mixed`, `assembly_robot`, `conveyor`, `welder`, `inspection` | all 12 | as S3 | all 12 |
| Technician templates | all 7 | all 7 | `generalist`, `junior` | all except the two specialists | all 7 | as S3 | all 7 |
| Route length | 3–6 | 3–6 | 2–3 | 3–6 | 8–15 | as S3 | 8–15 |
| Horizon (t.u.) | U(2×10⁵, 3.5×10⁵) | U(2×10⁵, 3.5×10⁵); GRPO: 2.75×10⁵ | 2×10⁵ | 2×10⁵ | 10⁵ | 5×10⁶ | 5×10⁶ |
| Episodes | 600 or 100 | 600 | 3 | 5 | 3 | 1 | 1 |
| Decision cap | 25,000 | 25,000 | 20,000 | 20,000 | 25,000 | 1,500,000 | 1,500,000 |
| Evaluation layout | — | — | 5 machines (3 CNC, 2 Assembly), 2 `generalist`, route of 2 types | 20 machines, 4 technicians (2 `generalist`, 1 `senior`, 1 `expert`), route of 3 types | 97 machines, 30 technicians, route of 9 types | as S3 | 78 machines, 24 technicians, route of 9 types; events of the [event schedule of S5](#event-schedule-of-s5) |

- Source: `KATAConfig` dumps of the seven configurations: `randomized_scenario` (`n_machines_min`/`max`, `n_technicians`, `n_technicians_min`/`max`, `techs_per_machine_min`/`max`, `machine_templates`, `technician_templates`, `route_min`/`max_length`); `src/kata/EntityFactories/scenario_sampler.py` (`_sample_n_technicians` ratio round, clamp >= 1; route length min(max, n types))
- Source: `scripts/eval_human_vs_performance.py` `SCENARIOS` (`small_scale` `n_eps` 3 sim 2e5 steps 2e4; baseline 5/2e5/2e4; `massive_scale` 1/1e5/1e4 overridden by `--steps 25000 --n-eps 3`; `very_long` uses `massive_scale.json` 1/5e6/1.5e6; lifecycle 1/5e6/1.5e6); `build_scenario` `RandomScenarioSampler(seed=EVAL_SEED).sample_config()`
- Source: `reports/hvp_eval_v6w/*/manifest.json` (`eval_seed` 20260722, `n_eval_episodes` 3/5/3/1/1, `max_eval_steps` 20000/20000/25000/1500000/1500000); max `n_steps` 1294/5544/20316/1020872/564350 < caps; `final_sim_time` equals the horizon in every row
- Source: evaluation layouts computed with `RandomScenarioSampler(KATAConfig(<config>), randomized_scenario, seed=20260722).sample_config()` at `5ce2a80`
- Source: training: `conf/train.yaml` `max_steps` 25000; `scripts/train_hydra.py` horizon range; GRPO fixed `sim_time` 275000

## Reward configuration

Table 3 *Reward components* of the paper (Section 5.4) gives the reward components and their coefficients. All training configurations set `normalize_components` to true: each enabled component, including the terminal ones, is standardized as (r̃<sup>(c)</sup><sub>t</sub> − μ<sub>c</sub>)/σ<sub>c</sub>, with running statistics that start at μ<sub>c</sub> = 0 and σ<sub>c</sub>² = 1 with a pseudo-count of 10⁻⁴ (`normalize_components_eps`), are updated before each use, and persist across the episodes of an environment worker. The checkpoints do not store these statistics. The knowledge shaping term is potential-based (`knowledge_increment_potential_based`) with γ<sub>p</sub> = 0.9999 per t.u. (`knowledge_potential_gamma`) and Δt = 0 at the first decision of an episode; the knowledge scale is s = 10 (`fleet_knowledge_scale`). The training configurations also enable `busy_technician` with coefficient 1, whose raw value is −1 when the assigned technician is busy with a repair; under the action mask, this occurs only when no technician is available. The production-only configuration disables `fatigue_cost`, `workload_balance`, `fleet_availability`, `knowledge_increment`, and `terminal_fleet_knowledge`; the fine-tunes change the coefficients of table *Commands of the pipeline* (Appendix G.2 of the paper). The five scenario configurations keep an earlier reward (knowledge increment floored at zero, without potential-based shaping, workload-balance, or availability terms). The evaluation harness freezes the normalizer before the first decision, and only GreedyReward uses the reward to choose actions.

- Source: `run_configs/benchmark_suite/train_multiscale_v5.json` `gym.reward` (`normalize_components` true, `normalize_components_eps` 1e-4, `knowledge_increment_potential_based` true, `knowledge_potential_gamma` 0.9999, `busy_technician` enabled coefficient 1.0); `gym.fleet_knowledge_scale` 10.0
- Source: `train_multiscale_v5_po.json` diff: `fatigue_cost`, `fleet_availability`, `knowledge_increment`, `terminal_fleet_knowledge`, `workload_balance` disabled; `train_multiscale_v5_mlp.json` reward identical to v5
- Source: `small_scale`/`baseline`/`massive_scale`/`lifecycle.json` diff: `fleet_availability` and `workload_balance` disabled, `knowledge_increment_potential_based` false, `knowledge_potential_gamma` 1.0
- Source: `src/kata/core/reward_normalizer.py` (`(raw - mean)/std`, update before normalize unless frozen); `src/agents/networks/running_stats.py` (mean 0, var 1, count eps); `src/kata/env.py` `_reward_component` (terminal components via `_reward_component`), `_reward_for_assignment` (`busy_raw` -1 if `tech.busy`; PBRS `gamma_p**dt`), reset (`_prev_decision_sim_time` at first decision); `_action_mask` (all-unavailable fallback); checkpoint keys (net, optimizer, `lr_scheduler`, `return_rms`, `max_techs`, `max_machines`, `vocab_size`, improvements, vocab): no reward-normalizer state
- Source: `scripts/eval_human_vs_performance.py` `run_episode` (`env.freeze_reward_normalizer()` after reset); `heuristics.py` `GreedyRewardAgent`

## Evaluation protocol and determinism

### Protocol

Each evaluation part builds the layout of the [training and evaluation worlds](#training-and-evaluation-worlds) table and uses it for all episodes and agents. Episode e has seed 100 × 20260722 + e, which seeds the Python, NumPy, and Numba global generators before the reset and the disruption generators of the technicians at the reset. All agents except Random act deterministically: HTT-RL and the MLP anchors take the highest masked logit or Q-value with the network in evaluation mode, and the rules use their deterministic variants ([Baselines](#baselines)). Random ignores the deterministic flag and samples uniformly among the offered technicians with the seeded NumPy generator, so its episodes are repeatable but its policy is stochastic. The heuristic baselines receive the structured observation and the learned agents the set observation (Section 5.1 of the paper). In the published records, every episode ends at the horizon and not at the decision cap; the longest episodes have 1,294, 5,544, 20,316, 1,020,872, and 564,350 decisions at S1 to S5. The step records contain every decision at S1 to S3, and every 200th decision and the last decision at S4 and S5; the episode KPIs use every decision.

- Source: `scripts/eval_human_vs_performance.py` (`build_scenario`: one `sample_config` with seed `EVAL_SEED`; main: seed = `EVAL_SEED*100 + ep`; `run_episode`: `np.random.seed`, `random.seed`, `seed_numba_rng`, `env.reset(seed)` -> `dispatcher.seed_disruptions`; `deterministic=True`; `build_agents`: `net.eval()`, structured env for `HEURISTICS`, set env for checkpoints; `record_every` keeps terminal step; metric sums over every decision)
- Source: `src/agents/ppo/ppo_set_transformer.py` `select_action` (argmax of masked probs, `_eval_mode_if`); `a2c_mlp.py`/`grpo_mlp.py` (argmax when deterministic); `dql_mlp.py` (greedy masked argmax when deterministic)
- Source: `reports/hvp_eval_v6w/<S>/episodes.csv` (max `n_steps` per scenario; `final_sim_time` equal to horizon in all rows)

### Run-to-run variation

Two pairs of checkpoints have identical weights: PO-HTT-RL and its final checkpoint, and HTT-RL<sub>qua.</sub> and its final checkpoint. Their evaluation records are identical at S1, S2, and S5 but differ at S3 and S4, which have 30 technicians: the mean number of finished products differs by 2.2% (PO-HTT-RL) and 0.9% (HTT-RL<sub>qua.</sub>) at S3, and by 3.5% (PO-HTT-RL) at S4. With identical layouts and seeds, two sources can explain these differences, and the records do not separate them: nondeterministic GPU kernels, which can change near-tied actions of the network, and the machine identifiers described below. The published runs did not enable the deterministic algorithms of PyTorch. Differences of a few percent between learned agents at S3 and S4 are therefore within the run-to-run variation of this protocol.

- Source: `checkpoints/po_v6_final/set_transformer_{best,last}.pt` and `checkpoints/ft_quality/set_transformer_{best,last}.pt`: all 118 net tensors equal (`torch.equal`)
- Source: `reports/hvp_eval_v6w/massive_scale/episodes.csv`: `po_v6` 2163.33 vs `po_v6_last` 2210.67 (+2.19%); `ft_quality` 2227.67 vs `ft_quality_last` 2206.67 (-0.94%); `very_long`: `po_v6` 112117 vs `po_v6_last` 116055 (+3.51%), `ft_quality` 113069 = `ft_quality_last`; `small_scale`, baseline, lifecycle: identical means for both pairs
- Source: no `torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` in `src/` or `scripts/`

### Machine identifiers and `PYTHONHASHSEED`

`ScenarioBuilder` sets the identifier of each machine to `hash(name) % 10000`. Python randomizes string hashing in each process, so two machines of a layout can receive the same identifier. The machine registry is keyed by this identifier, so a collision removes one machine from the registry that the observation, the availability reward, the per-machine statistics, the availability and MTBF metrics, and the lifecycle selection of machines read; the simulation of product flow, failures, and repairs does not read this registry. None of the published training or evaluation runs set `PYTHONHASHSEED`, so each process used its own hash seed. We recommend `export PYTHONHASHSEED=0` for new runs: with Python 3.13 at commit `5ce2a80`, this value gives distinct identifiers in the five evaluation layouts, whereas the values 3 and 5 give one collision in the layouts of S2 and S3, respectively.

- Source: `src/kata/scenario.py` `_build_simple_machine`/`_build_complex_machine` (`machine_id=hash(name) % 10000`), `dispatcher.machines` dict keyed by `machine_id`; `src/kata/metrics.py` `FleetAvailabilityRate` and `MeanTimeBetweenFailures` read `env._factory_machines()` and `_machine_breakdown_counts`, `TotalBreakdowns` counts popped tickets; `src/kata/env.py` `_factory_machines` (reads `dispatcher.machines`; used by `_set_obs`, rewards, `_update_machine_state_tracking`, `_select_machine_for_event`); machine/feeder/dispatcher repair paths do not read the registry
- Source: grep `PYTHONHASHSEED` at `5ce2a80`: only a queue script of the financial analysis (not used for the published records, and not in the code release) and a comment in `scripts/eval_human_vs_performance.py`; none in `dgy_v6_train.sh`, `dgy_v6_ft_queue.sh`, `dgy_trad_baselines_queue.sh`, `dgy_v6w_benchmarks.sh`, `dgy_v6w_lifecycle.sh`, `local_po_v6_queue.sh`, `local_ft_gini_queue.sh`, `local_lifecycle_parts.sh`, `dgy_disr_benchmarks.sh`
- Source: collision check (Python 3.13.14, `PYTHONHASHSEED=0..5`, names of the evaluation layouts from `RandomScenarioSampler` seed 20260722): 0,1,2,4 -> no collision at S1..S5; 3 -> 1 collision at S2; 5 -> 1 collision at S3/S4
