# Model-Based Eval Smoke Test

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

- Model: `Qwen/Qwen3.6-35B-A3B`
- Command shape: one sample, Level 1 Problem 1, one correctness trial, no performance timing.
- Result artifact: `runs/validation/base_qwen36_a3b_l1p1.json`
- Result: pipeline completed end-to-end, generated kernel compiled, but failed correctness with `Output mismatch`.

## Metrics

| Metric | Value |
|---|---:|
| Problems | 1 |
| Compiled | 1/1 |
| Correct | 0/1 |
| Error | `Output mismatch` |
