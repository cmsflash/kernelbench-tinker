# End-To-End Model Concurrency Probe

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This probe submitted concurrent `Qwen/Qwen3.6-35B-A3B` generations for the same Level 1 Problem 1 and then evaluated any extracted kernels.

Artifact: `runs/validation/concurrency_sweep/qwen36_l1p1_sweep.jsonl`

| Requested concurrency | Wall time | Throughput | Exceptions |
|---:|---:|---:|---:|
| 1 | 27.94s | 2.15/min | 0 |
| 2 | 34.10s | 3.52/min | 0 |
| 4 | 35.08s | 6.84/min | 0 |
| 8 | 178.35s | 2.69/min | 0 |

## Interpretation

The model-based path was not a reliable Modal concurrency measurement because most generations failed to produce valid kernel code. For this end-to-end workflow, 4-way concurrency was the best observed point before generation latency regressed at 8-way.
