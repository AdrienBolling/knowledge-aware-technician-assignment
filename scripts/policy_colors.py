"""Single source of truth for the policy colours of the Results figures.

Imported by make_scenario_figures.py (scenario panels and legend),
financial_analysis.py (profit maps and legend) and make_performance_profile.py
(profile panels and legend), so a policy has the same colour in every figure.
Hues are Okabe-Ito.  Hungarian and ReserveSpec are drawn as grey "other
rules" in the scenario panels; their colours here apply where they are tracked
(profiles, profit maps).
"""
COLOR = {
    'hc_v6': '#0072B2',               # HTT-RL ref.
    'ft_quality': '#009E73',          # HTT-RL qua.
    'topsis': '#D55E00',              # Topsis
    'shortest_processing': '#CC79A7', # Spt
    'optimal_assignment': '#E69F00',  # Hungarian
    'random': '#4D4D4D',              # Random
    'reserve_specialist': '#56B4E9',  # ReserveSpec
}
