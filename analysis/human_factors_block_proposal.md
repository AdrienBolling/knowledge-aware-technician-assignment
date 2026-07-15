# Human-factors block of Table 1: diagnosis and redesign proposal

> **STATUS: IMPLEMENTED 2026-07-07** (same day, on user approval). Coupling code
> `MOD` renamed `MDL` (collision with the objective code MOD). New block =
> 15 columns: SCAL VECT LAT | BIN ORD CONT | LBD DEC TRN STATIC | MDL ANT TGT
> PHY TRF. Migration mechanical except the judgment sets below; new column
> totals: SCAL 13, VECT 21, LAT 9 / BIN 7, ORD 10, CONT 16 / LBD 14, DEC 11,
> TRN 7, STATIC 24 / MDL 11, ANT 13, **TGT 0**, PHY 4, TRF 3 over the 43
> surveyed works — our row is the only TGT (+PHY+TRF) entry.
>
> Judgment sets applied (flag to co-authors for review):
> - **ANT** (optimizer anticipates skill dynamics): han, yang, bao, henao,
>   liuWorker2016, liuTraining2013, norman, mcdonald, suer, szwarc, chen2024,
>   xu, heuser.
> - **MDL** (state modelled/predicted only): ranasinghe, denu, alvarez,
>   long-fei, tien, zhang(flagged cite), lee, liuEKT, liang, hashemifar, wasi.
> - **PHY** (evolving fatigue/physiological state): joo, calzavara, ferjani,
>   denu. (bao considered, left out — well-being objective without clear
>   evolving fatigue state; check full text.)
> - Breadth tie-break: LAT wins over VECT when both present (the embedding is
>   the operative representation); value left empty for purely latent states.

Date: 2026-07-07. Based on the 43 verified rows of `tab:hrap_taxonomy_aggregated`
(post Grillo realignment). Companion to `taxonomy_row_verification.md`.

## Diagnosis: what the current codes conflate

### Skill representation (BIN LEV EFF COMP MSK EXP LAT SIMIL NONE)

**31 of 43 papers (72%) carry two or more representation codes** (3 carry
three). The co-occurrence pattern shows the column set mixes four orthogonal
properties of a skill state:

| Combo | n | What it reveals |
|---|---|---|
| BIN+MSK (7), LEV+MSK (3), COMP+MSK (2), EFF+MSK (3), LAT+MSK (3) | 18 | `MSK` is not an alternative to BIN/LEV/EFF/COMP — it is the *breadth* dimension (per-task-type vector) stacked on a *value type*. 18/24 MSK papers also carry a value-type code. |
| EFF+LEV (4), EFF+EXP (2), EXP+LEV (2) | 8 | `EFF` is not a structure but an *effect channel* (skill enters the model as a processing-time multiplier); `EXP` is *provenance* (state derived from accumulated exposure). |
| EFF+SIMIL (2), LAT+SIMIL (1) | 3 | `SIMIL` is a *transfer structure* across tasks, orthogonal to everything else. |

A category system where three quarters of the population needs multiple codes
per axis is a tag bag, not a taxonomy.

### Skill dynamics (STATIC TRAIN LEARN FORG. L/F XTRAIN UPSKILL POLICY NONE)

Cleaner (36/43 single-coded), but four defects:

1. **Non-canonical encodings**: 3 papers are coded `LEARN`+`FORG.` while 8
   others use `L/F` — two encodings for the same phenomenon.
2. **`POLICY` is not a human factor**: all 4 POLICY papers are RL/ML-method
   papers with no other dynamics code; POLICY ≡ (Method ∈ {RL, ML} ∧ skills
   STATIC) — fully derivable from two existing columns, and it describes the
   *algorithm*, not the worker.
3. **`TRAIN`/`XTRAIN`/`UPSKILL` overlap** (TRAIN+XTRAIN ×2, XTRAIN+LEARN/L-F
   ×2): all three are deliberate skill interventions; the distinctions carry
   7 papers total.
4. **Missing axes** — the ones the paper's thesis actually turns on:
   - *Decision coupling*: is the evolving human state merely **modeled**,
     **anticipated** by the optimizer when assigning, or **targeted** as an
     objective? Currently invisible; it is the precise difference between
     knowledge tracing (modeled), learning-aware scheduling à la Xu/Heuser/
     Chen (anticipated), and our work (targeted).
   - *Physiological state*: fatigue/recovery dynamics have **no column at
     all** (only the WELL objective code in Common parameters, 3 papers);
     a fatigue state that evolves and feeds back into performance — central
     to FactoReal and to several C3-cluster papers — is uncodable.
   - *Timescale*: within-horizon runtime learning vs between-horizon
     training decisions (old H2-vs-H3 distinction) is only recoverable by
     combining TRAIN vs LEARN.

## Proposed redesign (three orthogonal groups, 12 columns vs today's 18)

**Skill state** — single choice per sub-axis:
- Breadth: `SCAL` scalar per worker | `VECT` per-task-type vector | `LAT` latent embedding
- Value: `BIN` binary | `ORD` discrete levels | `CONT` continuous coefficient

**Skill dynamics** — drivers, multi-select (multi-marks now *mean* something):
- `LBD` learning-by-doing (absorbs LEARN, the learning arm of L/F, and EXP's provenance)
- `DEC` decay/forgetting (absorbs FORG. and the forgetting arm of L/F)
- `TRN` deliberate development interventions (absorbs TRAIN, XTRAIN, UPSKILL-as-intervention)
- `STATIC` none (unchanged)

**Human-state / optimization coupling** — the new discriminating group:
- `MOD` human state modeled or predicted only
- `ANT` dynamics anticipated by the allocation decision
- `TGT` human-state growth is itself an objective
- `PHY` physiological (fatigue/recovery) state modeled as evolving
- `TRF` cross-task transfer structure (absorbs SIMIL; also codes our
  knowledge-grid diffusion kernel)

## Why this is better

1. **Orthogonality**: the 72% multi-coding on the representation axis
   disappears by construction (breadth × value are independent single
   choices); remaining multi-marks (drivers) are semantically meaningful.
2. **Canonicality**: one encoding per phenomenon (kills LEARN+FORG vs L/F).
3. **No leaked solution-method code**: POLICY dropped, derivable from
   Method + STATIC.
4. **The gap the paper claims becomes a visible column**: under MOD/ANT/TGT,
   the 43 surveyed works split roughly into MOD (knowledge tracing,
   simulation, assessment), ANT (learning/forgetting-aware scheduling), and
   **zero TGT rows** — our row is the only TGT+PHY entry. The taxonomy then
   *shows* the contribution instead of narrating it.
5. **Cheap migration**: every new code except ANT/TGT and PHY is a
   deterministic function of the old codes; ANT/TGT/PHY need one targeted
   verification pass over the 43 rows (same pipeline as the 2026-07-06
   realignment; the dynamics-bearing subset is only 19 papers).

## Old→new mapping (mechanical part)

| Old | New |
|---|---|
| BIN | value=BIN (+ breadth from context: alone→SCAL, with MSK→VECT) |
| LEV | value=ORD (breadth from context) |
| COMP, EFF (value aspect) | value=CONT |
| EFF (effect aspect) | (effect channel folds into prose; optional `TIME` column if kept) |
| MSK | breadth=VECT |
| LAT | breadth=LAT |
| EXP | provenance only — no driver (several EXP papers hold skills fixed at runtime: stein, goncalves); LBD comes from LEARN/L-F |
| SIMIL | TRF |
| LEARN / FORG. / L/F | LBD / DEC / LBD+DEC |
| TRAIN, XTRAIN, UPSKILL | TRN |
| POLICY | dropped (≡ Method RL/ML + STATIC) |
| STATIC, NONE | unchanged |

Open question for the author: whether to also keep an explicit *effect
channel* sub-axis (gate / time-multiplier / quality / prediction-target).
It adds 4 columns; the information is partially recoverable from the
objective codes (QUAL) and prose. Recommended: fold into prose, keep the
table at 12 human-factor columns.
