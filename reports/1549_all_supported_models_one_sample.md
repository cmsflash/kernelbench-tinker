# All Supported Models One-Sample Pass

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This pass ran one sample for each model returned by Tinker's `supported_models` endpoint.

Artifact directory: `runs/validation/l1p1_all_models_c8`

## Run Configuration

| Setting | Value |
|---|---:|
| Models | 39 |
| Concurrency | 8 |
| Max output tokens | 4096 |
| Correctness trials | 1 |
| Performance timing | disabled |

## Summary

| Metric | Value |
|---|---:|
| Wall time | 200.30s |
| Throughput | 11.68 models/min |
| Exceptions | 0 |
| Format OK | 22/39 |
| Compiled | 5/39 |
| Correct | 0/39 |
| Hit token cap | 10/39 |

## Failure Stages

| Stage | Count |
|---|---:|
| Format extraction | 17 |
| Compile/eval failure | 17 |
| Compiled but incorrect | 5 |

## Compiled Models

| Model | Correct | Failure |
|---|---:|---|
| `moonshotai/Kimi-K2.6:peft:131072` | No | Output mismatch |
| `meta-llama/Llama-3.3-70B-Instruct` | No | at 13:14: a_ptr, # Pointer to first input b_ptr, # Pointer to second input out_ptr, # Pointer to output M: ... |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | No | at 4:10: def matmul_kernel( A, B, C, M, N, K ): row = tl.blockIdx.y * tl.blockDim.y + tl.threadIdx.y ^ Attr... |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | No | Output mismatch |
| `Qwen/Qwen3-4B-Instruct-2507` | No | at 36:26: for k in range(0, n, BLOCK_N): # Load a row slice of A (row, k:k+BLOCK_N) a_slice = tl.load(a_ptr... |

## Interpretation

No supported model solved L1.1 in this one-sample pass. The run separated failures into formatting failures, compile/evaluation failures, and compiled-but-incorrect kernels.
