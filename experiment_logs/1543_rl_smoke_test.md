# One-Batch RL Smoke Test

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

- Model: `Qwen/Qwen3.6-35B-A3B`
- Problems: Level 1 Problem 1 only
- Batch size: 1
- Group size: 2
- Correctness trials: 1
- Performance timing: disabled
- Result directory: `runs/validation/train_smoke_qwen36_a3b`
- Result: one training batch completed and final Tinker checkpoint was saved.

## Final Observed Metrics

| Metric | Value |
|---|---:|
| Batch groups | 1 |
| Trajectories | 2 |
| Reward mean | 0.0 |
| Format rate | 0.0 |
| Compile rate | 0.0 |
| Correct rate | 0.0 |
| Total step time | 57.26s |
