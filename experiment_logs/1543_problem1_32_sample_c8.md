# Problem 1 32-Sample Study At Concurrency 8

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This pass sampled `Qwen/Qwen3.6-35B-A3B` on Level 1 Problem 1 with 32 total samples and 8 concurrent in-flight samples.

Artifact directory: `runs/validation/problem1_32_c8`

## Run Configuration

| Setting | Value |
|---|---:|
| Total samples | 32 |
| Concurrency | 8 |
| Max output tokens | 4096 |
| Correctness trials | 1 |
| Performance timing | disabled |

## Summary

| Metric | Value |
|---|---:|
| Wall time | 166.19s |
| Throughput | 11.55 samples/min |
| Exceptions | 0 |
| Format OK | 2/32 |
| Compiled | 2/32 |
| Correct | 0/32 |
| Avg generation time | 37.25s |
| Avg eval time | 0.76s |
| Max sample wall time | 52.80s |

## Failure Breakdown

| Failure stage | Error | Count |
|---|---|---:|
| Format extraction | `Could not extract valid kernel code from response` | 30 |
| Correctness | `Output mismatch` | 2 |

## Token-Cap Observation

| Category | Count |
|---|---:|
| Hit 4096-token cap | 28 |
| Did not hit cap | 4 |

## Interpretation

The 32-sample pass was dominated by output truncation. Most failures were not meaningful kernel failures; they were incomplete or unparseable model responses caused by the 4096-token output limit.
