# Modal-Only Concurrency Probe To 64

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This probe reused the known-valid `ModelNew` and measured concurrent local requests to the Modal evaluator.

The known-valid Problem 1 kernel was created at:

- `runs/validation/concurrency_sweep/problem1_valid_kernel.py`

The implementation is a correctness baseline:

```python
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)
```

Single Modal verification:

| Metric | Value |
|---|---:|
| Format OK | true |
| Compiled | true |
| Correct | true |
| Tests passed | 1/1 |
| Modal eval time | 13.81s |
| Total eval time | 15.17s |

Artifacts:

- `runs/validation/concurrency_sweep/modal_only_l1p1_sweep_to64.jsonl`
- `runs/validation/concurrency_sweep/modal_only_l1p1_sweep_to64.log`

| Requested concurrency | Wall time | Throughput | Correct |
|---:|---:|---:|---:|
| 1 | 2.690s | 22.3/min | 1/1 |
| 2 | 1.111s | 108.0/min | 2/2 |
| 4 | 1.768s | 135.7/min | 4/4 |
| 8 | 2.764s | 173.7/min | 8/8 |
| 16 | 4.931s | 194.7/min | 16/16 |
| 32 | 15.031s | 127.7/min | 32/32 |
| 64 | 14.181s | 270.8/min | 64/64 |

## Interpretation

- The current setup sustained 64 requested concurrent evaluation calls without failures.
- The repo evaluator config uses `max_batch_size=32`.
- The Modal app class is configured with `max_containers=32`.
- Therefore, the 64-request result confirms the local/repo evaluation path handles 64 concurrent requests, but it does not prove 64 simultaneous Modal A100 containers. It likely runs as batched waves under the current `32` limits.
- To test true Modal-side parallelism above 32, raise both the local evaluator `max_batch_size` and the Modal app `max_containers`, redeploy, and rerun the same sweep.
