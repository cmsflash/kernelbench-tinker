# Selected Models Pass@4, L1.1, Concurrency 8

This pass evaluated 8 selected Tinker models on KernelBench Level 1 Problem 1 with 4 samples per model and 8 total concurrent samples.

Artifact directory: `runs/validation/l1p1_selected8_pass4_c8_/161633`

## Settings

| Setting | Value |
|---|---:|
| Models | 8 |
| Samples per model | 4 |
| Total samples | 32 |
| Concurrency | 8 |
| Max output tokens | 4,096 |
| Correctness trials | 1 |
| Performance timing | Disabled |

## Aggregate

| Metric | Value |
|---|---:|
| Wall time | 148.7s |
| Passes | 1/32 |
| Extraction | 24/32 |
| Compilation | 19/32 |
| Token-cap hits | 9/32 |
| Average tokens | 2,403 |
| Average generation | 27.9s |
| Average eval | 5.0s |

## Results

| Model | Done | Passes | Extraction | Compilation | Correctness report | Avg tokens | Wall s | Gen s | Eval s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4/4 | 0 | 4 | 4 | Output mismatch x3; shared memory exceeded x1 | 965 | 29.1 | 10.5 | 17.5 |
| `meta-llama/Llama-3.1-8B-Instruct` | 4/4 | 0 | 4 | 2 | `triton_jit` import error x1; Tensor parent error x1 | 1,302 | 29.1 | 12.2 | 14.6 |
| `deepseek-ai/DeepSeek-V3.1` | 4/4 | 0 | 4 | 2 | Output mismatch x2; invalid fullwidth pipe char x1 | 1,146 | 33.8 | 23.2 | 0.9 |
| `moonshotai/Kimi-K2.6:peft:131072` | 4/4 | 1 | 3 | 3 | Output mismatch x2; extraction failure x1 | 2,682 | 48.4 | 30.0 | 3.5 |
| `Qwen/Qwen3.6-35B-A3B` | 4/4 | 0 | 0 | 0 | Extraction failure x4 | 4,096 | 44.5 | 35.0 | 0.0 |
| `Qwen/Qwen3.5-397B-A17B` | 4/4 | 0 | 2 | 2 | Output mismatch x2; extraction failure x2 | 2,564 | 63.9 | 45.6 | 0.7 |
| `openai/gpt-oss-20b` | 4/4 | 0 | 3 | 2 | Shared memory exceeded x1; invalid decimal literal x1 | 3,762 | 31.4 | 27.2 | 1.3 |
| `openai/gpt-oss-120b` | 4/4 | 0 | 4 | 4 | Output mismatch x3; missing `forward` x1 | 2,704 | 47.8 | 39.8 | 1.5 |

## Notes

Only `moonshotai/Kimi-K2.6:peft:131072` produced a passing sample in this pass.

`Qwen/Qwen3.6-35B-A3B` was kept in the report as requested; all four of its samples hit the 4,096-token cap and failed extraction.
