# knowledge-aware-technician-assignment

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
