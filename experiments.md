# Experiments

## 2026-05-25: KernelBench Problem 1 Validation And Modal Concurrency

### Setup

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

### Model-Based Eval Smoke Test

- Model: `Qwen/Qwen3.6-35B-A3B`
- Command shape: one sample, Level 1 Problem 1, one correctness trial, no performance timing.
- Result artifact: `runs/validation/base_qwen36_a3b_l1p1.json`
- Result: pipeline completed end-to-end, generated kernel compiled, but failed correctness with `Output mismatch`.

Metrics:

| Metric | Value |
|---|---:|
| Problems | 1 |
| Compiled | 1/1 |
| Correct | 0/1 |
| Error | `Output mismatch` |

### One-Batch RL Smoke Test

- Model: `Qwen/Qwen3.6-35B-A3B`
- Problems: Level 1 Problem 1 only
- Batch size: 1
- Group size: 2
- Correctness trials: 1
- Performance timing: disabled
- Result directory: `runs/validation/train_smoke_qwen36_a3b`
- Result: one training batch completed and final Tinker checkpoint was saved.

Final observed metrics:

| Metric | Value |
|---|---:|
| Batch groups | 1 |
| Trajectories | 2 |
| Reward mean | 0.0 |
| Format rate | 0.0 |
| Compile rate | 0.0 |
| Correct rate | 0.0 |
| Total step time | 57.26s |

### End-To-End Model Concurrency Probe

This probe submitted concurrent `Qwen/Qwen3.6-35B-A3B` generations for the same Level 1 Problem 1 and then evaluated any extracted kernels.

Artifact: `runs/validation/concurrency_sweep/qwen36_l1p1_sweep.jsonl`

| Requested concurrency | Wall time | Throughput | Exceptions |
|---:|---:|---:|---:|
| 1 | 27.94s | 2.15/min | 0 |
| 2 | 34.10s | 3.52/min | 0 |
| 4 | 35.08s | 6.84/min | 0 |
| 8 | 178.35s | 2.69/min | 0 |

Interpretation: the model-based path was not a reliable Modal concurrency measurement because most generations failed to produce valid kernel code. For this end-to-end workflow, 4-way concurrency was the best observed point before generation latency regressed at 8-way.

### Valid Kernel For Modal-Only Probe

A known-valid Problem 1 kernel was created at:

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

### Modal-Only Concurrency Probe To 64

This probe reused the known-valid `ModelNew` and measured concurrent local requests to the Modal evaluator.

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

Interpretation:

- The current setup sustained 64 requested concurrent evaluation calls without failures.
- The repo evaluator config uses `max_batch_size=32`.
- The Modal app class is configured with `max_containers=32`.
- Therefore, the 64-request result confirms the local/repo evaluation path handles 64 concurrent requests, but it does not prove 64 simultaneous Modal A100 containers. It likely runs as batched waves under the current `32` limits.
- To test true Modal-side parallelism above 32, raise both the local evaluator `max_batch_size` and the Modal app `max_containers`, redeploy, and rerun the same sweep.

### Problem 1 32-Sample Study At Concurrency 8

This pass sampled `Qwen/Qwen3.6-35B-A3B` on Level 1 Problem 1 with 32 total samples and 8 concurrent in-flight samples.

Artifact directory: `runs/validation/problem1_32_c8`

Run configuration:

| Setting | Value |
|---|---:|
| Total samples | 32 |
| Concurrency | 8 |
| Max output tokens | 4096 |
| Correctness trials | 1 |
| Performance timing | disabled |

Summary:

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

Failure breakdown:

| Failure stage | Error | Count |
|---|---|---:|
| Format extraction | `Could not extract valid kernel code from response` | 30 |
| Correctness | `Output mismatch` | 2 |

Token-cap observation:

| Category | Count |
|---|---:|
| Hit 4096-token cap | 28 |
| Did not hit cap | 4 |

Interpretation: the 32-sample pass was dominated by output truncation. Most failures were not meaningful kernel failures; they were incomplete or unparseable model responses caused by the 4096-token output limit.

### Problem 1 8-Sample Study With Maximum Safe Token Limit

This rerun used the same model/problem pair and concurrency, but increased the output budget to the largest safe value for the model context. The prompt was 1163 tokens, the model context limit was 65536 tokens, and the run used a 128-token buffer, giving `max_tokens=64245`.

Artifact directory: `runs/validation/problem1_8_c8_maxtok`

Run configuration:

| Setting | Value |
|---|---:|
| Total samples | 8 |
| Concurrency | 8 |
| Max output tokens | 64245 |
| Correctness trials | 1 |
| Performance timing | disabled |

Summary:

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

Per-sample outcomes:

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

Interpretation: increasing the output limit removed truncation as the main explanation. None of the 8 samples hit the new token cap. The remaining failures are model-output and code-quality failures: unparseable responses, missing `ModelNew`, invalid Triton API usage, and an over-large shared-memory kernel.
