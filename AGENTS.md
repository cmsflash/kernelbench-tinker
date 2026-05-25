# Agent Notes

## Summary

- Modal evaluation for KernelBench Level 1 Problem 1 sustained 64 concurrent local evaluation requests with a known-valid kernel and no correctness failures; the current repo/app configuration uses `max_batch_size=32` and Modal `max_containers=32`, so 64 requested concurrency is supported by the repo evaluation path but may execute as batched waves rather than 64 simultaneous Modal containers.
- For model-based KernelBench generation/evaluation, use the maximum practical output token limit. Short limits caused `Qwen/Qwen3.6-35B-A3B` to get cut off before emitting a complete valid `<KERNEL>` block, producing formatting failures instead of meaningful kernel evaluations.
- When reporting evaluation runs, use a compact summary followed by a per-sample table. Include model, level/problem, sample count, concurrency, max output tokens, wall time, throughput, format/compile/correct counts, token-cap hits, and average generation/evaluation time. For each sample, include sample id, failure stage, failure reason, output tokens, wall seconds, generation seconds, and evaluation seconds.
