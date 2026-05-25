# Extractor and Prompt Fix

## Summary

Implemented the extraction fallback and prompt alignment changes requested after reviewing the extraction failures from:

- `runs/validation/l1p1_selected8_pass4_c8_/161633/results.jsonl`
- `runs/validation/l1p1_all_models_c8_1552_raw/results.jsonl`

## Code Changes

- Added shared prompt module: `src/kernelbench_tinker/envs/prompts.py`
- Updated RL training environment to import the shared `DEFAULT_SYSTEM_PROMPT`
- Updated eval generation to use the same shared `DEFAULT_SYSTEM_PROMPT`
- Updated `extract_code_block()` to:
  1. Prefer `<KERNEL>...</KERNEL>` when present
  2. Strip everything through `</think>` when present
  3. Collect fenced code blocks
  4. Choose the last block containing both `class ModelNew` and `def forward`
  5. Else choose the last block containing `class ModelNew`
  6. Else choose the last block containing `def forward`
  7. Else fail extraction

## Validation

Command:

```bash
uv run python -m compileall src/kernelbench_tinker
```

Result: passed.

Saved-artifact parser replay:

| Study | Old extraction failures | New extraction failures | Recovered |
|---|---:|---:|---:|
| 8-model pass@4 | 8 | 2 | 6 |
| 39-model pass@1 | 15 | 10 | 5 |
| Combined | 23 | 12 | 11 |

## Notes

The new parser recovers one more saved failure than the earlier rough oracle count because the repo's current `format_ok` rule accepts a selected block containing either `class ModelNew` or `def forward`.

The eval path previously used only:

```text
You are an expert GPU kernel developer.
```

It now uses the same structured prompt as RL training, including the required `<KERNEL>```python ... ```</KERNEL>` response format.
