# All Supported Models One-Sample Pass With Raw Outputs

## Setup Context

- Repository: `/Users/cmsflash/programs/kernelbench-tinker`
- Benchmark problem: KernelBench Level 1 Problem 1, square matrix multiplication
- Problem file: `KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py`
- Inputs: two random FP32 matrices of shape `(4096, 4096)`
- Reference behavior: `torch.matmul(A, B)`
- Modal GPU observed during validation: `NVIDIA A100-SXM4-40GB`

## Experiment

This pass ran one sample for each model returned by Tinker's `supported_models` endpoint after confirming that eval records preserve both raw model output and extracted/evaluated kernel code.

Artifact directory: `runs/validation/l1p1_all_models_c8_1552_raw`

## Run Configuration

| Setting | Value |
|---|---:|
| Models | 39 |
| Concurrency | 8 |
| Max output tokens | 4096 |
| Correctness trials | 1 |
| Performance timing | disabled |

## Logging Verification

| Field | Value |
|---|---:|
| Raw response logged | true |
| Kernel code logged | true |
| JSONL rows missing `response.raw` | 0 |
| JSONL rows missing `kernel_code` | 0 |

## Summary

| Metric | Value |
|---|---:|
| Wall time | 352.72s |
| Throughput | 6.63 models/min |
| Exceptions | 0 |
| Format OK | 24/39 |
| Compiled | 19/39 |
| Correct | 0/39 |
| Hit token cap | 10/39 |

## Failure Stages

| Stage | Count |
|---|---:|
| Format extraction | 15 |
| Compile/eval failure | 5 |
| Compiled but incorrect | 19 |

## Compiled Models

| Model | Correct | Failure |
|---|---:|---|
| `deepseek-ai/DeepSeek-V3.1` | No | Output mismatch |
| `moonshotai/Kimi-K2-Thinking` | No | Output mismatch |
| `moonshotai/Kimi-K2.5` | No | Output mismatch |
| `moonshotai/Kimi-K2.5:peft:131072` | No | Output mismatch |
| `moonshotai/Kimi-K2.6:peft:131072` | No | out of resource: shared memory, Required: 262144, Hardware limit: 166912. Reducing block sizes or `num_stages` may help. |
| `meta-llama/Llama-3.1-8B-Instruct` | No | 'function' object is not subscriptable |
| `meta-llama/Llama-3.3-70B-Instruct` | No | Output mismatch |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | No | at 9:10: def _matmul_kernel( A, B, C, M, N, K ): idx = tl.program_id(0) i = idx // N j = idx % N if i >= M or j >= N:... |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | No | Output mismatch |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | No | Output mismatch |
| `Qwen/Qwen3-4B-Instruct-2507` | No | at 31:20: # Initialize accumulator acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) # Loop over all K dimensions ... |
| `Qwen/Qwen3-8B-Base` | No | at 14:10: B_ptr, # Pointer to input matrix B C_ptr, # Pointer to output matrix C M: tl.constexpr, N: tl.constexpr, K:... |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | No | Output mismatch |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | No | at 27:15: # Loop over K dimension for k in range(0, N, BLOCK_SIZE): # Load A block: BLOCK_SIZE x BLOCK_SIZE k_offset ... |
| `Qwen/Qwen3.5-27B` | No | at 57:65: # Load A block: [BLOCK_SIZE_M, BLOCK_SIZE_K] a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_offs[None, :... |
| `Qwen/Qwen3.5-397B-A17B` | No | Output mismatch |
| `openai/gpt-oss-120b` | No | Cannot call @triton.jit'd outside of the scope of a kernel |
| `openai/gpt-oss-120b:peft:131072` | No | Output mismatch |
| `openai/gpt-oss-20b` | No | at 37:14: block_start_n = pid_n * BLOCK_SIZE_N # Temporary accumulator for the block acc = tl.zeros((BLOCK_SIZE_M, BL... |

## Interpretation

No supported model solved L1.1 in this pass. Preserving raw outputs should make the next diagnosis sharper because format failures can now be distinguished from parser limitations and actual model truncation/content issues.
