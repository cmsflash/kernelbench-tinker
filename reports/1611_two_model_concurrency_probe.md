# Two-Model End-To-End Concurrency Probe

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal evaluator: `kernel-rl-evaluator/KernelEvaluator`
- Correctness trials: `1`
- Performance timing: disabled
- Evaluation cache: disabled for the final pass

## Experiment

This probe compared end-to-end model generation plus KernelBench evaluation throughput for two models at requested concurrency 4 and 8, then ran a mixed 4 + 4 condition.

Artifacts:

- Runner: `scripts/model_concurrency_probe.py`
- JSONL: `runs/validation/concurrency_sweep/two_model_l1p1_c4_c8_mixed.jsonl`
- Log: `runs/validation/concurrency_sweep/two_model_l1p1_c4_c8_mixed.log`

Max output token budgets:

| Model | Max output tokens | Note |
|---|---:|---|
| `Qwen/Qwen3.6-35B-A3B` | 64,245 | Same practical budget used in the prior max-token run |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 31,544 | The API rejected 64,245 because `1,096 + 64,245 > 32,768`; this uses a 128-token buffer |

## Summary

| Condition | Requested concurrency | Wall time | Throughput | Format OK | Compiled | Correct | Token-cap hits | Avg gen | Avg eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.6-35B-A3B` | 4 | 38.96s | 6.16/min | 0/4 | 0/4 | 0/4 | 0/4 | 32.85s | 0.00s |
| `Qwen/Qwen3.6-35B-A3B` | 8 | 56.61s | 8.48/min | 2/8 | 2/8 | 0/8 | 0/8 | 31.56s | 4.03s |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | 11.52s | 20.83/min | 4/4 | 4/4 | 0/4 | 0/4 | 8.43s | 3.08s |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 8 | 17.13s | 28.02/min | 8/8 | 8/8 | 0/8 | 0/8 | 10.86s | 3.65s |
| Mixed: `Qwen3.6` 4 + `Qwen3-30B` 4 | 8 total | 56.59s | 8.48/min | 4/8 | 4/8 | 0/8 | 0/8 | 29.37s | 0.69s |

## Mixed Breakdown

| Model | Requested concurrency | Samples | Throughput using mixed wall | Max sample wall | Format OK | Compiled | Correct | Avg gen | Avg eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3.6-35B-A3B` | 4 | 4 | 4.24/min | 56.58s | 0/4 | 0/4 | 0/4 | 49.59s | 0.00s |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 4 | 4 | 4.24/min | 11.06s | 4/4 | 4/4 | 0/4 | 9.16s | 1.39s |

## Interpretation

This pass did not reproduce the earlier result that `Qwen/Qwen3.6-35B-A3B` was best at concurrency 4. In this final cache-free pass, requested concurrency 8 had higher throughput than concurrency 4 for both individual models:

- `Qwen/Qwen3.6-35B-A3B`: 6.16/min at c4 vs 8.48/min at c8
- `Qwen/Qwen3-30B-A3B-Instruct-2507`: 20.83/min at c4 vs 28.02/min at c8

The mixed 4 + 4 run, however, was much slower than the sum of the two solo c4 rates. Expected from solo c4 rates would be roughly `6.16 + 20.83 = 26.99/min`; observed mixed throughput was `8.48/min`. The slowdown was concentrated in `Qwen/Qwen3.6-35B-A3B`, whose average generation time rose from 32.85s solo c4 to 49.59s in the mixed run. The smaller Qwen model still generated in about 9s, but total mixed throughput was bounded by the slow model's wall time.

So the observed evidence is: solo c8 did not hit the same apparent limit in this run, but mixed 4 + 4 does suggest the two models are not fully independent under concurrent load. A single pass is not enough to prove a stable shared quota, because `Qwen3.6` showed high run-to-run variance across the old and new probes.

## Per-Sample Details

### `Qwen/Qwen3.6-35B-A3B`, c4

| Sample | Failure stage | Failure reason | Output tokens | Wall | Gen | Eval |
|---:|---|---|---:|---:|---:|---:|
| 0 | format | Could not extract valid kernel code from response | 4,215 | 34.78s | 33.50s | 0.00s |
| 1 | format | Could not extract valid kernel code from response | 5,217 | 37.68s | 37.68s | 0.00s |
| 2 | format | Could not extract valid kernel code from response | 4,531 | 34.14s | 34.13s | 0.00s |
| 3 | format | Could not extract valid kernel code from response | 3,098 | 26.11s | 26.10s | 0.00s |

### `Qwen/Qwen3.6-35B-A3B`, c8

| Sample | Failure stage | Failure reason | Output tokens | Wall | Gen | Eval |
|---:|---|---|---:|---:|---:|---:|
| 0 | incorrect | `at 47:22:` | 1,727 | 38.65s | 22.07s | 16.58s |
| 1 | format | Could not extract valid kernel code from response | 2,435 | 22.07s | 22.06s | 0.00s |
| 2 | format | Could not extract valid kernel code from response | 5,322 | 38.43s | 38.43s | 0.00s |
| 3 | format | Could not extract valid kernel code from response | 7,170 | 56.60s | 56.60s | 0.00s |
| 4 | format | Could not extract valid kernel code from response | 4,439 | 33.79s | 33.79s | 0.00s |
| 5 | format | Could not extract valid kernel code from response | 4,015 | 33.79s | 33.79s | 0.00s |
| 6 | format | Could not extract valid kernel code from response | 2,435 | 23.95s | 23.95s | 0.00s |
| 7 | incorrect | Output mismatch | 1,702 | 37.46s | 21.76s | 15.69s |

### `Qwen/Qwen3-30B-A3B-Instruct-2507`, c4

| Sample | Failure stage | Failure reason | Output tokens | Wall | Gen | Eval |
|---:|---|---|---:|---:|---:|---:|
| 0 | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.52s | 8.42s | 3.10s |
| 1 | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.52s | 8.44s | 3.07s |
| 2 | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.52s | 8.41s | 3.10s |
| 3 | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.51s | 8.45s | 3.06s |

### `Qwen/Qwen3-30B-A3B-Instruct-2507`, c8

| Sample | Failure stage | Failure reason | Output tokens | Wall | Gen | Eval |
|---:|---|---|---:|---:|---:|---:|
| 0 | incorrect | Cannot call `@triton.jit` outside of the scope of a kernel | 929 | 14.77s | 11.25s | 3.51s |
| 1 | incorrect | `at 19:30:` | 956 | 14.76s | 11.14s | 3.62s |
| 2 | incorrect | Cannot call `@triton.jit` outside of the scope of a kernel | 998 | 17.12s | 11.57s | 5.55s |
| 3 | incorrect | `at 49:4:` | 812 | 11.40s | 9.37s | 2.03s |
| 4 | incorrect | Cannot call `@triton.jit` outside of the scope of a kernel | 929 | 14.75s | 11.21s | 3.54s |
| 5 | incorrect | Cannot call `@triton.jit` outside of the scope of a kernel | 929 | 14.75s | 11.13s | 3.62s |
| 6 | incorrect | Cannot call `@triton.jit` outside of the scope of a kernel | 825 | 11.40s | 9.37s | 2.03s |
| 7 | incorrect | `at 19:27:` | 1,021 | 17.11s | 11.83s | 5.27s |

### Mixed 4 + 4

| Sample | Model | Failure stage | Failure reason | Output tokens | Wall | Gen | Eval |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | `Qwen/Qwen3.6-35B-A3B` | format | Could not extract valid kernel code from response | 5,680 | 48.68s | 48.67s | 0.00s |
| 1 | `Qwen/Qwen3.6-35B-A3B` | format | Could not extract valid kernel code from response | 3,813 | 46.57s | 46.56s | 0.00s |
| 2 | `Qwen/Qwen3.6-35B-A3B` | format | Could not extract valid kernel code from response | 3,813 | 46.56s | 46.55s | 0.00s |
| 3 | `Qwen/Qwen3.6-35B-A3B` | format | Could not extract valid kernel code from response | 5,101 | 56.58s | 56.58s | 0.00s |
| 0 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.06s | 9.36s | 1.70s |
| 1 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 10.03s | 8.95s | 1.07s |
| 2 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 10.03s | 8.95s | 1.07s |
| 3 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | incorrect | out of resource: shared memory, Required: 262144, Hardware limit: 166912 | 844 | 11.06s | 9.36s | 1.70s |
