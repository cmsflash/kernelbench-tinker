# Kimi-K2.6 L1.1 RL Stop And Reward Revision

## Summary

The first Kimi-K2.6 L1.1 RL run was stopped after the first completed update because every observed reward was `0.0`. This confirmed that the previous reward setup did not provide useful separation between malformed, non-compiling, partially correct, and fully incorrect outputs.

Artifact directory: `runs/kimi_k26_l1p1_g4_b1_100steps`

## Run Setup

| Setting | Value |
|---|---:|
| Model | `moonshotai/Kimi-K2.6:peft:131072` |
| LoRA rank | 32 |
| Problem | L1.1 |
| Batch size | 1 |
| Group size | 4 |
| Planned batches | 100 |
| Max output tokens | 120,000 |
| Constant reward filter | Enabled, but uniform groups are retained by cookbook fallback |

## Final Observed State

| Metric | Value |
|---|---:|
| Completed metric rows | 1 |
| Trace records | 6 |
| Checkpoints | 2 |
| Latest checkpoint | `000001` |
| Skipped batches | 0 |
| Uniform reward warnings | 1 |
| Completed batch reward mean | 0.0 |
| Completed batch format rate | 75.0% |
| Completed batch compile rate | 75.0% |
| Completed batch correct rate | 0.0% |
| Completed batch wall time | 293.43s |

Across all 6 trace records before stopping, format rate was 83.3%, compile rate was 66.7%, correct rate was 0.0%, and total observed reward was 0.0.

## Reward Change

The reward setup was revised before the next run:

| Condition | New reward behavior |
|---|---|
| Format failure | `-2.0` |
| Compile failure | `-1.0` |
| Partial correctness | `0.3 * tests_passed / tests_total` |
| Full correctness | `0.3`, with no speed bonus |
| Incomplete correctness with speed data | `0.3 + speed_reward` |

Focused reward tests and full unittest discovery both pass after the change.
