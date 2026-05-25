# Agent Notes

## Summary

- Modal evaluation for KernelBench Level 1 Problem 1 sustained 64 concurrent local evaluation requests with a known-valid kernel and no correctness failures; the current repo/app configuration uses `max_batch_size=32` and Modal `max_containers=32`, so 64 requested concurrency is supported by the repo evaluation path but may execute as batched waves rather than 64 simultaneous Modal containers.
- For model-based KernelBench generation/evaluation, use the maximum practical output token limit. Short limits caused `Qwen/Qwen3.6-35B-A3B` to get cut off before emitting a complete valid `<KERNEL>` block, producing formatting failures instead of meaningful kernel evaluations.
- When reporting evaluation runs, use a compact summary followed by a per-sample table. Include model, level/problem, sample count, concurrency, max output tokens, wall time, throughput, format/compile/correct counts, token-cap hits, and average generation/evaluation time. For each sample, include sample id, failure stage, failure reason, output tokens, wall seconds, generation seconds, and evaluation seconds.

Example report format:

Summary:
- Model: `Qwen/Qwen3.6-35B-A3B`
- Problem: Level 1 Problem 1
- Samples: 8
- Concurrency: 8
- Max output tokens: 64245
- Wall time: 87.7s
- Throughput: 5.48 samples/min
- Format OK: 3/8
- Compiled: 2/8
- Correct: 0/8
- Token cap hits: 0/8
- Avg generation/eval time: 62.1s / 1.6s

| Sample | Stage | Failure | Tokens | Wall s | Gen s | Eval s |
|---:|---|---|---:|---:|---:|---:|
| 0 | Correctness | `module 'triton' has no attribute 'matmul'` | 4265 | 52.2 | 42.6 | 8.3 |
| 1 | Format extraction | Could not extract valid kernel code | 4529 | 44.2 | 44.2 | 0.0 |
| 6 | Correctness | Shared memory resource exceeded | 1911 | 38.0 | 33.8 | 4.3 |
| 7 | Compile | Missing `ModelNew` | 8159 | 74.9 | 74.3 | 0.6 |
