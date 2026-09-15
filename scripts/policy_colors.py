"""Single source of truth for the policy colours of the Results figures.

Imported by make_scenario_figures.py (scenario panels and legend),
financial_analysis.py (profit maps and legend) and make_performance_profile.py
(profile panels and legend), so a policy has the same colour in every figure.
Hues are Okabe-Ito.  Hungarian* is drawn as a grey "other rule" in the scenario
panels; its colour here applies where it is tracked (profiles, profit maps).
"""
COLOR = {
    'hc_v6': '#0072B2',               # HTT-RL ref.
    'ft_quality': '#009E73',          # HTT-RL qua.
    'topsis': '#D55E00',              # Topsis*
    'shortest_processing': '#CC79A7', # Spt*
    'optimal_assignment': '#E69F00',  # Hungarian*
    'random': '#4D4D4D',              # Random
    'greedy_reward': '#000000',       # GreedyReward* (full field only)
    'reserve_specialist': '#56B4E9',  # ReserveSpec* (full field only)
}
