# Taxonomy table (Table 1) row-by-row verification — 2026-07-06

Audit trail for the Grillo-framework realignment of `tab:hrap_taxonomy_aggregated`.
Old→new codes: MKS→EXE, THRPT→RAT, FIT→QUAL, CMP→ELIG, CAP→RES, BD→DISR, MPER→REC.
Each surveyed paper was checked against its title/abstract (web search where missing).
"applied" = change made to the table; "skipped" = agent suggestion rejected (kept original coding).

## `jooTaskAllocationHuman2022`
- **+HIST** (applied, high): Abstract explicitly states that historical data accumulated in the process of job assignment are exploited to allocate tasks.
- note: STO, BD, HM, RL, LAT (unobservable/implicit operator characteristics learned by the DRL agent), and POLICY are all directly supported by the abstract. BAL and PREC are not visible in the abstract but are plausible full-text features; kept per conservative rule. Fatigue is accommodated as a state feature rather than a clear well-being objective, so WELL was not added.

## `hanMemeticAlgorithmBased2025a`
- **+PREC** (skipped, medium): Flexible job-shop scheduling inherently involves operation precedence within each job, a first-class feature of the FJSP formulation.
- note: MKS (makespan) and BAL (maximum worker workload minimisation) are explicit bi-objectives; LF and worker cooperation are the paper's core novelty. PREC addition depends on survey convention for job-shop routing precedence. A mathematical model is formulated but solved only by the memetic algorithm, so MP was not added. CMP (worker-machine eligibility) is likely in the full model but not evidenced by the abstract.

## `yangRealTimeHumanMachine2025a`
- **-WELL** (applied, medium): Reported objectives are completion time, system energy consumption, and delay time; despite Industry 5.0 framing, no explicit human well-being, fatigue, or ergonomic objective appears in the abstract.
- note: Learning-forgetting model dynamically assesses worker efficiency (EFF, EXP, LF, HIST all supported). Task slack-based matching is a rule-based heuristic (HEUR); validation is via engineering case study across order scales, consistent with SIM. Energy consumption here reads as production energy, not operator energy expenditure. Domain label could be sharpened to 'Human-machine-logistics workshop scheduling' but 'Smart manufacturing' is acceptable.

## `calzavaraMultiobjectiveTaskAllocation2023`
- note: All codes well supported: makespan minimisation plus operator energy expenditure and average mental workload (WELL) in a multi-objective optimisation model (MP); tasks split between human operators and cobots (HM) with eligibility restrictions (CMP). Assembly precedence constraints are likely in the full model but not evidenced by the abstract, so PREC was not added.

## `ferjaniSimulationoptimizationBasedHeuristic2017`
- note: Mean flowtime minimisation maps to MKS (completion-time criterion). Multi-skilled workers (HET, MSK, CMP), stochastic job-shop (STO), online TOPSIS-based assignment rule (HEUR) with weights tuned by offline simulation-optimisation (SIM), and fatigue as a processing-time inflation factor (EFF) are all explicit. Fatigue is a modelled state affecting durations, not an objective, so WELL correctly absent; skill levels themselves are fixed (STATIC).

## `safaeiSwiftHeuristicMethod2018`
- note: Total weighted completion time minimisation (MKS), skill-designation eligibility (CMP, BIN), per-skill workforce-size bounds (CAP), time-indexed mathematical model (MP) plus a decomposition heuristic (HEUR) are all explicit. MSK is borderline: the decomposition into single-skill sub-problems hints technicians may each hold a single skill designation, but the abstract does not clearly contradict multi-skill profiles, so it is kept.

## `caricatoWorkforceInfluenceManufacturing2021`
- note: Abstract confirms parallel-machine scheduling with sequence-dependent setups (SETUP), CP for the machine problem and an ad hoc procedure for workforce planning (CP, HEUR). MKS, CMP, TEMP, BIN, MSK not explicit in the abstract but consistent with the described real industrial workforce problem; kept conservatively. The two-stage decomposition (CP + ad hoc procedure) could arguably be coded HYB, but both component methods are already marked.

## `saberModelingIntegratedFlexible2024`
- **+CMP** (applied, medium): Integrated FJSP-operator scheduling with binary multi-skill operator profiles (BIN/MSK already coded) implies a qualification mask restricting which operators can perform which operations.
- note: Shift-based operator constraints explicitly confirm TEMP; MIP and CP models explicitly confirm MP and CP. MKS is the standard FJSP objective and consistent with the paper. Domain label is acceptable; 'Flexible job-shop scheduling' would be slightly more precise.

## `normanWorkerAssignmentCellular2002`
- **+CAP** (applied, medium): Abstract explicitly varies total training time and per-worker available training time, i.e. training-budget/capacity bounds are modeled constraints.
- note: Effectiveness objective combines productivity (THRPT), quality, and training costs (COST); MIP formulation (MP); workers differ in technical and human skills (HET, MSK, COMP); explicit training decisions to change skill levels (TRAIN). All current codes well supported. Output quality is also an objective term but has no dedicated code.

## `baoEnhancingGarmentManufacturing2025`
- note: THRPT (production efficiency, +28% daily capacity), WELL (worker well-being, ergonomics), HIST/EXP (historical performance), HEUR (priority-based three-phase scheme), SIM (simulation evaluation) all explicit in abstract. BAL and PREC are not visible in the abstract but are plausible for a garment sewing-line model with order/job phases; kept conservatively on the author's full-text coding.

## `henaoMultiskilledPersonnelAssignment2023`
- **+MPER** (applied, high): Abstract explicitly plans working-hour allocation and productivity for each week of a multi-week planning horizon.
- **-THRPT** (applied, medium): The stated objective is minimizing understaffing (demand coverage, already captured by FIT); worker productivity appears as a model parameter/output, not a maximized throughput objective.
- note: STO kept: the MIP itself is deterministic, but uncertain demand is a first-class feature tested over nine variability levels. ROT covers the training-in-which-task-types decisions; LF explicit (learning-forgetting phenomena). Caveat for the author: the case study is a Chilean retail store, not a manufacturing setting; if category A is strictly industrial workforce, this paper could arguably sit in C, though the model is core multiskilled-HRAP methodology.

## `egilmezStochasticSkillbasedManpower2014`
- **+HET** (applied, high): Abstract explicitly models each worker's individual expected value and standard deviation of processing time on each operation, i.e. workers have differing efficiencies.
- note: THRPT (maximize production rate), RISK (specified risk level, robust hierarchical optimization), STO (stochastic processing times and demand), CAP (manpower levels / probabilistic capacity requirements), MP (stochastic nonlinear mathematical models), EFF (individual processing-time performance) all explicit. IID sampling is a statistical conversion step, not system simulation, so SIM not added.

## `liuWorkerAssignmentProduction2016`
- **+COST** (applied, high): The stated objective is explicitly 'to minimize backorder cost and holding cost of inventory'.
- **+CAP** (applied, medium): The abstract explicitly frames the problem around 'limited production capacity' causing backorders and early production.
- **+HYB** (applied, medium): The method is a self-described hybrid bacteria foraging algorithm with an embedded two-phase heuristic for initial solutions.
- **-THRPT** (applied, medium): Raising bottleneck production rate is only the motivation for reassignment; the optimization objective is backorder plus inventory holding cost, not throughput.
- note: Cellular manufacturing (fiber connectors) with learning/forgetting-driven bottleneck shifts across periods; joint worker reassignment and production planning. EFF+MSK+LF and MPER/HET/META all clearly supported.

## `liuTrainingAssignmentMultiskilled2013`
- **+XTRAIN** (applied, medium): The training plan creates multi-skilled workers across seru tasks, i.e. cross-training decisions, consistent with the seru literature's cross-training framing.
- **-HYB** (applied, medium): The abstract describes a three-stage constructive heuristic (nine steps), not an explicit hybrid of distinct method families.
- note: Bi-objective model: minimize total training cost and balance processing times among workers in each seru; task-to-worker training plan plus worker-to-seru assignment. COST, BAL, ROT, TRAIN all explicit in abstract.

## `staruchCompetencebasedAssignmentTasks2021`
- **+HET** (applied, high): Competence coefficients explicitly encode differing per-worker skill levels for each task, so workers are heterogeneous by construction.
- **+CAP** (skipped, medium): The ILP models are stated to be closely related to the Generalized Assignment Problem, whose defining feature is per-worker capacity constraints.
- note: ILP task-to-worker assignment with competence coefficients that also serve as a qualification mask (zero competence blocks assignment); dummy worker guarantees feasibility. CMP, COMP, MP, STATIC clearly supported.

## `chenTechnicianRoutingScheduling2024`
- **+STO** (applied, high): The problem is a multi-period dynamic MDP with future uncertainties and randomly distributed customers, i.e. stochastic request arrivals.
- note: ADP/CFA (implicit cross-training) for dynamic technician routing and scheduling; learning-by-doing reduces service times (EFF/EXP/LEARN) and balanced skill-building yields workforce flexibility (FLEX). All current codes supported; workload balance is reported as an outcome, not an objective, so BAL correctly absent.

## `kataokaMathematicalModelConsidering2023`
- note: Multi-period MIP for reconfigurable manufacturing cells jointly considering multi-skilled human operators and industrial robots (HM clearly applies); solved via 2-phase optimization. Abstract does not state the objective explicitly but a cost objective is standard for these cell-formation models; keeping COST as marked.

## `mcdonaldDevelopmentApplicationWorker2009`
- **+SIM** (applied, medium): Assignment schedules are additionally evaluated with a simulation model to assess further performance metrics.
- **+LEV** (applied, medium): The model uses discrete skill depth levels, with quality varying by skill depth and tasks requiring minimum skill depth.
- **+CMP** (applied, medium): Tasks carry skill-depth requirements that workers must meet, i.e. a qualification/eligibility constraint on assignments.
- **+FORG** (skipped, medium): Job rotation is explicitly included to retain skills, implying modeled skill decay when tasks are not practiced.
- note: Net-present-cost worker assignment model for a lean electronics-assembly cell, deciding training and rotation for cross-trained workers; optimization results complemented by simulation evaluation. FORG coexists with TRAIN/XTRAIN since retention-driven rotation implies forgetting dynamics.

## `steinLearningStateDependentPolicy2024`
- **+ROUT** (applied, high): The paper is explicitly a technician ROUTING problem: tours are built each day to visit geographically spread customers.
- **+STO** (applied, high): Customers request service dynamically over time and technician absences occur, making it a stochastic sequential decision process.
- **+MPER** (applied, medium): The problem spans a number of service days with rework revisits linking decisions across days, i.e. a multi-period/repeated-period horizon.
- **-BAL** (applied, high): The only 'balance' in the paper is the state-dependent balance among routing efficiency, urgency, and rework risk in the policy parametrization; the objective is minimizing customer delay/inconvenience, not workload fairness across technicians.
- note: Stein/Hildebrandt/Thomas/Ulmer dynamic technician routing with rework; skill-vs-complexity mismatch drives rework probability, so LEV/EXP and REW are appropriate. RL tunes the policy weights while technician skills stay fixed (POLICY).

## `aliOptimizingAircraftMaintenance2025`
- **+MKS** (applied, high): The model minimizes downtime and the headline result is a 25% reduction in completion time of the maintenance check, i.e. makespan/completion-time minimisation.
- **+PREC** (applied, high): The formulation explicitly accounts for sequential/parallel task constraints, i.e. task precedence.
- **+HET** (applied, high): Tasks are assigned based on differing manpower expertise and skill levels, so workers are heterogeneous.
- **+EFF** (applied, medium): Manpower efficiency is explicitly incorporated into the MILP with task durations tied to skill level, consistent with an efficiency effect on processing time.
- note: TEMP kept conservatively but the abstract gives no direct evidence of shift/break/calendar constraints (only task durations and precedence for a 50-hour Cessna 172 check); worth spot-checking the model. REG is clearly supported (regulatory compliance is a stated model ingredient).

## `lianMultiskilledWorkerAssignment2018`
- **+MP** (applied, medium): A mathematical (bi-objective) model is formulated and solved exactly via a weighted-sum method on the numerical example before resorting to NSGA-II.
- **+CMP** (applied, medium): Workers have differing skill sets, implying tasks can only be assigned to workers holding the required skill (qualification mask).
- note: Seru worker grouping + cell loading + task assignment with inter-seru and inter-worker workload balance objectives; skill sets (MSK) with proficiency levels (LEV) and no skill dynamics (STATIC) all check out.

## `imranhasantusarTechnicianAssignmentMultishift2024`
- **+CMP** (applied, high): The model requires that technicians are assigned only tasks that match their skills, an explicit skill-task eligibility constraint.
- **+AVAIL** (applied, medium): The primary objective is minimizing unmet maintenance needs / unassigned positions, i.e. a service-coverage/availability objective alongside workload balance.
- note: Multi-shift schedules with labor-regulation, overwork-prevention, and break constraints support REG and TEMP. Well-being is promoted only via constraints (breaks, no excessive work), so WELL as an objective is correctly not marked.

## `suerMultiperiodOperatorAssignment2008`
- **+MP** (applied, high): Mathematical models are explicitly used in all three phases (cell configuration, cell loading/crew sizing, operator assignment), alongside the heuristics.
- **+MKS** (skipped, medium): Makespan is the reported performance criterion (heuristic Max yields lower makespan), indicating completion-time minimisation is an objective of the assignment models.
- note: Skill-based operation times justify EFF; Max/MaxMin operator-assignment heuristics prefer most-qualified assignment (FIT); learning and forgetting effects across periods justify LF and MPER; math models + heuristics justify HYB.

## `szwarcRobustSchedulingMultiSkilled2024a`
- **+BD** (applied, medium): The model explicitly targets robustness to unexpected disruptions (employee absenteeism, shifts in project priorities), which are first-class problem features.
- note: Software project scheduling problem (SPSP) variant with learning/forgetting-aware job rotation of versatile programmers, solved via constraint programming; a slightly more specific domain label would be 'Software project scheduling', but the current label is acceptable. Placement in A is reasonable since it is multi-skilled workforce-to-task allocation, though it sits at the boundary with C (non-manufacturing domain).

## `xuSequencingLearningForgetting2025`
- **-PREC** (applied, medium): The problem is pure job sequencing on a single production system; the abstract describes no precedence constraints between tasks (sequence-dependent processing times via learning/forgetting are not precedence).
- **-MP** (skipped, medium): The exact method is a dynamic programming algorithm (plus DP-based lower bounds), not LP/MILP/MINLP as MP is defined.
- note: If the survey's convention maps exact/DP methods onto MP, the MP mark can stand, but strictly by the given definition it does not fit. SIMIL is well supported (full-matrix task-type similarity drives learning/forgetting). EFF fits since learning/forgetting act as processing-time modifiers.

## `heuserSinglemachineSchedulingProduct2023`
- **-PREC** (applied, medium): Single-machine sequencing with no precedence constraints mentioned; ordering matters only through the category-based learning/forgetting effect.
- note: MKS covers both makespan and total completion time objectives stated in the abstract. SIMIL is justified via the product-category structure (intra-category learning, inter-category forgetting). MP kept since the 'optimal solution methods' plausibly include exact formulations, though the abstract does not specify.

## `ranasingheEffectsStochasticHeterogeneous2024`
- **-MP** (applied, medium): The paper builds an analytical stochastic performance-evaluation model of a two-workstation line; there is no LP/MILP/MINLP optimization program.
- note: Descriptive performance-analysis paper, so MOD plus THRPT (throughput-time and related measures) is the right objective coding. Worker well-being is only a closing motivation, not a modeled objective, so no WELL. The analytical Markov-style model has no exact method code in the taxonomy; SIM kept as it is the closest match for the numerical evaluation.

## `felberbauerStochasticProjectManagement2019`
- **+META** (applied, high): The matheuristic explicitly uses an iterated local search metaheuristic to determine project schedules.
- **+CAP** (applied, high): Internal-resource capacity limits are first-class: when required capacity exceeds internal capacity, costly external resources are hired.
- **+MPER** (applied, medium): The Heimerl-Kolisch-type model schedules work packages and staffs resources over a multi-period planning horizon.
- **block change** (applied): A. Core HRAP
- note: Placement concern: this is simultaneous project scheduling and personnel assignment with static skills (no learning/fatigue/knowledge dynamics), so it matches A. Core HRAP rather than B. Industrial dynamics. If B was chosen for stochastic (non-human) dynamics, that conflicts with the stated category definition. All currently marked codes are well supported.

## `denuHumanCentricDigitalSimulation2024`
- note: Methodology/guidelines paper for digital simulation of operator skill acquisition and health in circular manufacturing; WELL (health) and UPSKILL (skills acquisition) are explicit in the abstract. LEV/EXP details are not visible in the abstract but are consistent with the described skill-acquisition modelling; no clear grounds to change.

## `alvarezHumanProfessionalSkills2025`
- note: MOD, LEV ('categorizing into different skill levels'), EFF (Wright learning curve on processing time), UPSKILL (explicit upskilling model) and HIST are well supported. SIM and ML are not evidenced by the abstract (the core method is an analytical modified learning-curve model with a case study); flagging for author double-check but not removing, since the full paper may use simulation or data-driven fitting.

## `long-feiModelingUserBehaviors2020`
- note: Behavioral modelling of operator skill from recorded machine-operation (sewing) sessions for adaptive guidance; all marked codes fit (prototype selection + experience integration supports ML/LAT, observed continuous skill improvement supports LEARN). Borderline HCI paper, but the machine-operation setting justifies keeping it in B rather than C.

## `tienModelingInfluenceTechnician2018`
- **+REW** (applied, high): Imperfect maintenance is modeled explicitly via a probability model of repair quality subject to technician proficiency.
- **+BD** (applied, high): Machine failures/degradation are the core setting (run-to-failure, preventive and condition-based maintenance on a health-index phase-type model).
- **+STO** (applied, high): Phase-type degradation, probabilistic imperfect repair, and the simulated queueing model are inherently stochastic.
- **-CMP** (applied, medium): There is no qualification/eligibility constraint or assignment decision; technician proficiency is a parameter affecting repair outcome, not a skill-task mask.
- note: Simulation study of how technician proficiency interacts with maintenance strategies; AVAIL (machine availability) is the central metric, with cycle time secondary (THRPT could arguably also apply). EFF kept since proficiency effectively scales maintenance effectiveness, though it acts on repair quality more than on processing time.

## `muklasonSolvingNurseRostering2024`
- **+COST** (applied, medium): The NRP objective here is minimising the weighted penalty cost of soft-constraint violations, which maps to cost minimisation rather than makespan.
- **+CMP** (applied, medium): Abstract explicitly lists nurses' abilities among the scheduling constraints, i.e. a qualification/eligibility restriction on which nurse may cover a shift.
- **-MKS** (applied, high): Nurse rostering has a fixed planning horizon and no makespan/completion-time objective; the objective is soft-constraint penalty minimisation.
- note: RL is used as the hyper-heuristic selection mechanism on top of Simulated Annealing with reheating, so META+RL+POLICY are all correct. Norwegian hospital benchmark instances. BIN and MSK both plausible for nurse skill categories; kept conservatively.

## `chenCombinedMixedInteger2023`
- **+MP** (applied, high): Mixed integer programming is in the title and is embedded in the reconstruction step of the algorithm.
- **+HYB** (applied, high): Abstract explicitly states the authors 'design a hybrid algorithm with learning and optimization methods'.
- **+COST** (applied, medium): Standard NRP objective is minimising the weighted penalty cost of soft-constraint violations, better captured by COST than by MKS.
- **-MKS** (applied, high): No makespan/completion-time objective in nurse rostering; the horizon is fixed and the objective is constraint-violation penalty minimisation.
- note: The DNN selects among low-level improvement heuristics (hyper-heuristic style), so META is defensible, though HEUR could equally apply. 'Reducing work pressure on nurses' in the abstract is motivational framing, not an explicit well-being objective, so WELL was not added.

## `lvTeamFormationLarge2024`
- **+CMP** (applied, medium): The formed team must 'possess the necessary skills to meet project requirements', i.e. skill coverage is a hard eligibility requirement on the selection.
- note: GCO-DQN = DQN (RL) over GNN employee embeddings (ML, LAT, GRPH); objective is minimising scheduling cost and disruption to the department communication network (COST). Abstract also mentions 'resource constraints' which could support CAP, but too vague to add. Workers clearly have heterogeneous skills, but HET appears reserved for the industrial blocks by the author's convention, so not added.

## `zhangGraphNeuralNetworks2022`
- **+D.NONE** (skipped, medium): The chapter contains no workers or skills at all, so skill dynamics are not applicable.
- **-BAL** (skipped, high): This is a methodological book chapter on GNN-based link prediction; it contains no workload-balance or fairness objective across workers.
- **-MSK** (skipped, high): No worker skill profiles appear anywhere in the chapter; it is a generic graph-ML methods survey.
- **-STATIC** (skipped, medium): There are no worker skills to be static; D.NONE is the applicable skill-dynamics code.
- **domain** (flagged): GNN link prediction (methods reference)
- note: LIKELY CITATION MISMATCH: the bib entry is the 'Link Prediction' chapter of the book Graph Neural Networks: Foundations, Frontiers, and Applications, but the row's domain label ('Crowdsourcing') and codes (BAL, MSK, STATIC) strongly suggest the author intended a crowdsourcing worker-task assignment paper. Please check the intended reference. LAT was kept because latent node embeddings are literally the chapter's subject and may be the reason it is cited as a representation-method reference; if the row keeps this bib entry, MOD/GRPH/ML/LAT are the only defensible codes.

## `goncalvesHumanResourceAllocation2025`
- **+EXP** (applied, medium): Work experience is explicitly part of the worker representation used for allocation (title and abstract: allocation considering aptitude and work experience).
- **block change** (applied): A. Core HRAP
- note: The abstract is squarely about assigning industrial workers to workstations/operations on manufacturing production lines to maximise system performance, which matches the definition of block A (allocation/scheduling of industrial human workforce) rather than C. Two ILP models (parallel and sequential), validated on a real case, so MP is correct; worker 'restrictions' support CMP; aptitude scores support COMP.

## `muraretu_initial_2017`
- **+SIM** (applied, medium): The five allocation strategies are compared in a simulated (agent-based) project environment, per the paper's description.
- **+MKS** (applied, medium): Effectiveness of the allocation strategies is measured by project delivery time (the headline finding is a fivefold speed-up in delivery time), i.e. a completion-time criterion.
- note: Verified via web search (no abstract in bib): Muraretu & Ilie, Annals of the University of Craiova, 2017. Tasks and employees are modelled as skill vectors (supports MSK/COMP) and five fit-based fitness functions serve as allocation heuristics (supports FIT/HEUR). Caveat on CMP: the finding that 'suboptimal' employees can be assigned suggests soft matching rather than a strict qualification mask, but a minimum-eligibility rule cannot be ruled out from available sources, so CMP was conservatively retained. Non-industrial (software-project) setting, so block C is correct.

## `leeIndustrialHumanResource2020`
- note: Person-job matching on skill preferences and MBTI with a working-location constraint; FIT/HET/STATIC clearly supported by the abstract. MP+ML pairing (matching optimization inside the AI-based AID platform) and COMP/LAT skill representation are plausible from the abstract and not contradicted, so retained. C placement is right: hiring/career matching, not shop-floor allocation.

## `liuEKTExerciseawareKnowledge2019`
- **+SIMIL** (applied, medium): The EERNNA/EKTA attention mechanism explicitly weights past exercises by cosine similarity between exercise-content embeddings, i.e., a task-similarity model drives the prediction.
- note: IEEE TKDE paper (venue missing in bib). FORG has no dedicated decay mechanism but knowledge-state decay is captured by the recurrent state; this matches the author's evident LEARN+FORG convention for DKT-family models, so it is kept.

## `liangHELPDKTInterpretableCognitive2022`
- note: DKT variant for programming with feature-rich input (raw code embeddings, error-class concept indicators) and interpretable ability estimates; ability estimates remain latent vectors so LAT is the right skill representation. FORG kept under the same DKT-family convention (implicit decay via recurrent state).

## `hashemifarPersonalizedStudentKnowledge2025`
- note: KMaP: stateful multi-task model combining knowledge tracing with behavior/resource-preference prediction via clustering-based student profiles; all codes consistent. Domain label 'Knowledge tracing' still accurate even though it also predicts future learning-resource choices.

## `wasiHRGraphLeveragingLLMs2024`
- note: No abstract in bib, verified from knowledge of the paper: LLMs extract HR entities (skills, jobs, employees) from CVs/job descriptions into knowledge graphs, then GNN-style information propagation over node embeddings yields job and employee recommendations. Skills appear as multi-skill entity profiles (MSK) embedded in graph space (LAT); no skill dynamics (STATIC). HIST was considered (CV/experience documents) but the data is static-document rather than longitudinal history, so not proposed.
