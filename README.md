# knowledge-aware-technician-assignment

## Time units

Simulation time is unitless; the calibration used throughout the released
configurations reads **1 time unit ≈ 1 minute**. Under that anchor:

| Quantity | Config value | Reads as |
|---|---|---|
| Technician travel delay | 15 t.u. | 15 minutes |
| Base repair times | 10–150 t.u. | 10 min – 2.5 h |
| Industrial-scale MTTR | 70–100 t.u. | ≈ 1.2–1.7 h |
| Benchmark horizons | 1–2 × 10⁵ t.u. | ≈ 10 weeks – 4.5 months |
| Very-long study | 5 × 10⁶ t.u. | ≈ 9.5 years (career-scale) |

The simulator models continuous coverage (no shift patterns), so the anchor
is indicative rather than literal. See the paper's experimental-setup section
for the same statement in context.

## Observation (`obs`) format in `KataEnv`

`KataEnv` supports two observation representations, selected with
`gym.observation_representation` in `GymEnvConfig`:

- `structured` (default): numeric dictionary (legacy behavior)
- `tokens`: fixed-shape textual tokens

### Token observations (fixed shape)

When `observation_representation="tokens"`, `obs` has the shape:

```python
{
  "tokens": tuple[str, ...]  # length == token_observation_length
}
```

The tuple length is always exactly `token_observation_length`:

- if generated tokens are fewer, it is padded with `token_pad_value` (default: `"<PAD>"`)
- if generated tokens are more, it is truncated

Token size is constrained by Gym space `Text(max_length=token_max_length)`.

### Observation modes

Choose with `gym.observation_mode`:

1. `ticket_only`
   - Includes ticket/simulation context tokens such as:
     - `OBS_MODE:*`
     - `SIM_TIME:*`
     - `HAS_OPEN_TICKET:*`
     - `TICKET_CREATED_AT:*`
     - `TICKET_MACHINE_ID:*`

2. `broken_machine`
   - Includes all `ticket_only` tokens, plus machine-level tokens for the broken machine:
     - `MACHINE_ID:*`
     - `MACHINE_BROKEN:*`
     - `MACHINE_PROCESSING:*`
     - `MACHINE_TOTAL_PROCESSED:*`
     - `MACHINE_INPUT_BUFFER:*`
     - `MACHINE_OUTPUT_BUFFER:*`

3. `factory_level`
   - Includes all `broken_machine` tokens, plus factory aggregate tokens:
     - `FACTORY_MACHINE_COUNT:*`
     - `FACTORY_BROKEN_COUNT:*`
     - `FACTORY_PROCESSING_COUNT:*`
     - `FACTORY_TOTAL_PROCESSED:*`
     - `FACTORY_QUEUE_SIZE:*`

### Optional fleet technician tokens

In token mode, you can add fleet-wide technician data:

- `gym.include_technician_fatigue_tokens=True`
  - Adds `TECH_{i}_FATIGUE:*` for each technician
- `gym.include_technician_knowledge_tokens=True`
  - Adds `TECH_{i}_KNOWLEDGE:*` for each technician

### Minimal config example

```python
GymEnvConfig(
    observation_representation="tokens",
    observation_mode="factory_level",
    token_observation_length=64,
    token_max_length=64,
    token_pad_value="<PAD>",
    include_technician_fatigue_tokens=True,
    include_technician_knowledge_tokens=True,
)
```
