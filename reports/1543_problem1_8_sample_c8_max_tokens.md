# Problem 1 8-Sample Study With Maximum Safe Token Limit

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This rerun used the same model/problem pair and concurrency, but increased the output budget to the largest safe value for the model context. The prompt was 1163 tokens, the model context limit was 65536 tokens, and the run used a 128-token buffer, giving `max_tokens=64245`.

Artifact directory: `runs/validation/problem1_8_c8_maxtok`

## Run Configuration

| Setting | Value |
|---|---:|
| Total samples | 8 |
| Concurrency | 8 |
| Max output tokens | 64245 |
| Correctness trials | 1 |
| Performance timing | disabled |

## Summary

| Metric | Value |
|---|---:|
| Wall time | 87.66s |
| Throughput | 5.48 samples/min |
| Exceptions | 0 |
| Format OK | 3/8 |
| Compiled | 2/8 |
| Correct | 0/8 |
| Hit token cap | 0/8 |
| Avg generation time | 62.10s |
| Avg eval time | 1.64s |
| Max sample wall time | 86.36s |

## Per-Sample Outcomes

| Sample | Stage | Failure | Tokens | Wall time | Generation time | Eval time |
|---:|---|---|---:|---:|---:|---:|
| 0 | Correctness | `module 'triton' has no attribute 'matmul'` | 4265 | 52.2s | 42.6s | 8.3s |
| 1 | Format extraction | `Could not extract valid kernel code from response` | 4529 | 44.2s | 44.2s | 0.0s |
| 2 | Format extraction | `Could not extract valid kernel code from response` | 8338 | 75.0s | 75.0s | 0.0s |
| 3 | Format extraction | `Could not extract valid kernel code from response` | 9801 | 86.4s | 86.4s | 0.0s |
| 4 | Format extraction | `Could not extract valid kernel code from response` | 8539 | 76.3s | 76.3s | 0.0s |
| 5 | Format extraction | `Could not extract valid kernel code from response` | 6992 | 64.3s | 64.3s | 0.0s |
| 6 | Correctness | `out of resource: shared memory, Required: 262144, Hardware limit: 166912. Reducing block sizes or num_stages may help.` | 1911 | 38.0s | 33.8s | 4.3s |
| 7 | Compile | `module 'temp_module' has no attribute 'ModelNew'` | 8159 | 74.9s | 74.3s | 0.6s |

## Interpretation

Increasing the output limit removed truncation as the main explanation. None of the 8 samples hit the new token cap. The remaining failures are model-output and code-quality failures: unparseable responses, missing `ModelNew`, invalid Triton API usage, and an over-large shared-memory kernel.
