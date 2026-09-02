# Realistic machine lifespans (isolated probe, 2026-09-02)

`machine_templates_realistic.json`: the 12 benchmark machine templates
with component lifespans scaled x25 toward realistic magnitudes
(weibull scale x25; simple failure probabilities /25) under 1 t.u. ~ 1
minute — per-machine MTBF moves from ~6.5 operating hours to ~7 days.
Registered ONLY when an eval passes `--extra-machine-templates` with
this file; names carry the `_rl` suffix so the regular experiment
world is untouched. `very_long_realistic.json` = massive_scale layout
on the `_rl` park (used by eval scenario `very_long_realistic`).
