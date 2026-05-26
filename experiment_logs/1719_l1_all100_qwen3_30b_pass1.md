# L1 All-Problem Pass@1: Qwen3 30B A3B Instruct

Timestamp: 17:19 US Pacific

This experiment ran `Qwen/Qwen3-30B-A3B-Instruct-2507` on all 100 KernelBench Level 1 problems with one sample per problem.

## Configuration

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Level | 1 |
| Problems | 1-100 |
| Samples per problem | 1 |
| Total samples | 100 |
| Concurrency | 8 |
| Max output tokens | 4096 |
| Temperature | 0.0 |
| Correctness trials | 1 |
| Performance timing | Disabled |

## Artifacts

| Artifact | Path |
|---|---|
| Summary | `runs/validation/l1_all100_qwen3_30b_a3b_inst_p1_c8_/170711/summary.json` |
| Results JSONL | `runs/validation/l1_all100_qwen3_30b_a3b_inst_p1_c8_/170711/results.jsonl` |

## Summary

| Done | Passes | Extraction | Compilation | Token cap | Avg tokens | Avg wall s | Gen s | Eval s | Total wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100/100 | 4 | 96 | 62 | 5 | 1,221 | 29.5 | 12.9 | 16.6 | 384.1 |

## Stage Breakdown

| Stage | Count |
|---|---:|
| Passed | 4 |
| Correctness | 58 |
| Compilation | 34 |
| Extraction | 4 |

## Problem Lists

| Category | Problems |
|---|---|
| Passed | `5`, `12`, `27`, `96` |
| Extraction failures | `36`, `58`, `65`, `80` |
| Token-cap hits | `18`, `36`, `58`, `65`, `80` |
| Compile failures | `18`, `38`, `40`, `45`, `48`, `52`, `55`, `56`, `57`, `59`, `60`, `62`, `63`, `67`, `68`, `69`, `70`, `71`, `72`, `73`, `74`, `75`, `76`, `77`, `78`, `81`, `82`, `83`, `84`, `87`, `92`, `95`, `98`, `100` |

## Top Failure Themes

| Failure | Count |
|---|---:|
| `Evaluation failed: "attribute 'bias' already exists"` | 17 |
| `Evaluation failed: CUDA error: an illegal memory access was encountered` | 16 |
| `Output mismatch` | 11 |
| `CUDA out of memory` | 7 |
| `Could not extract valid kernel code from response` | 4 |

## Notes

The run produced 4 passing Level 1 problems out of 100. Most generated outputs were extractable, and 62 compiled, but only 4 passed correctness. Later Level 1 problems had more compilation-stage failures, especially duplicate module attributes and illegal CUDA memory access errors.
