# Compiled Models by Generation Time

All-model pass@1 for L1.1, filtered to models that reached the correctness stage, with `Qwen/Qwen3.6-35B-A3B` kept as a reference row even though it failed format extraction. Ordered by `Gen s` ascending.

| Model | Stage | Failure | Tokens | Wall s | Gen s | Eval s |
|---|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | Correctness | Triton compile error | 547 | 6.8 | 5.0 | 0.8 |
| `meta-llama/Llama-3.1-8B-Instruct` | Correctness | Invalid kernel launch syntax | 632 | 9.3 | 6.5 | 1.8 |
| `Qwen/Qwen3-8B-Base` | Correctness | Triton compile error | 974 | 10.1 | 8.3 | 0.8 |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Correctness | Output mismatch | 983 | 12.1 | 9.0 | 2.2 |
| `moonshotai/Kimi-K2.6:peft:131072` | Correctness | Shared memory resource exceeded | 1,151 | 28.1 | 12.7 | 14.3 |
| `openai/gpt-oss-20b` | Correctness | Triton compile error | 2,381 | 15.1 | 12.7 | 0.8 |
| `deepseek-ai/DeepSeek-V3.1` | Correctness | Output mismatch | 935 | 29.6 | 13.4 | 13.7 |
| `meta-llama/Llama-3.3-70B-Instruct` | Correctness | Output mismatch | 671 | 16.3 | 14.2 | 1.1 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Correctness | Output mismatch | 867 | 19.3 | 17.0 | 1.3 |
| `Qwen/Qwen3.5-27B` | Correctness | Triton compile error | 1,756 | 21.7 | 19.8 | 0.8 |
| `openai/gpt-oss-120b:peft:131072` | Correctness | Output mismatch | 1,720 | 29.6 | 27.3 | 1.3 |
| `Qwen/Qwen3.6-35B-A3B` | Format extraction | Could not extract valid kernel code | 4,096 | 28.9 | 27.7 | 0.0 |
| `Qwen/Qwen3.5-397B-A17B` | Correctness | Output mismatch | 1,622 | 30.5 | 28.0 | 1.4 |
| `Qwen/Qwen3-4B-Instruct-2507` | Correctness | Triton compile error | 861 | 32.2 | 30.4 | 0.8 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Correctness | Triton compile error | 4,096 | 32.6 | 30.7 | 0.9 |
| `openai/gpt-oss-120b` | Correctness | Invalid Triton JIT call | 3,058 | 51.3 | 48.9 | 1.3 |
| `moonshotai/Kimi-K2.5:peft:131072` | Correctness | Output mismatch | 1,524 | 69.3 | 66.6 | 1.6 |
| `moonshotai/Kimi-K2.5` | Correctness | Output mismatch | 1,615 | 73.5 | 68.9 | 3.5 |
| `moonshotai/Kimi-K2-Thinking` | Correctness | Output mismatch | 1,225 | 77.7 | 75.4 | 1.3 |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | Correctness | Output mismatch | 1,101 | 271.4 | 250.7 | 19.8 |
