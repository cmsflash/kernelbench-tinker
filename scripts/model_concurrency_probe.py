#!/usr/bin/env python3
"""Probe end-to-end KernelBench throughput at model-generation concurrency limits."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import renderers, tokenizer_utils

from kernelbench_tinker.env import setup_environment
from kernelbench_tinker.envs.kernelbench_client import (
    KernelBenchProblem,
    evaluate_kernel_async,
)
from kernelbench_tinker.modal.evaluator import (
    ModalEvaluatorConfig,
    ModalKernelEvaluator,
    set_modal_evaluator,
)
from kernelbench_tinker.scripts.eval_kernel_rl import generate_kernel
from kernelbench_tinker.training.models import get_renderer_name_for_model


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRuntime:
    model: str
    sampling_client: tinker.SamplingClient
    renderer: renderers.Renderer


async def run_sample(
    *,
    runtime: ModelRuntime,
    problem: KernelBenchProblem,
    sample_index: int,
    max_tokens: int,
    temperature: float,
    modal_timeout: float,
) -> dict[str, Any]:
    sample_start = time.perf_counter()
    try:
        generated = await generate_kernel(
            runtime.sampling_client,
            problem,
            runtime.renderer,
            max_tokens,
            temperature,
        )
    except Exception as exc:
        wall_s = time.perf_counter() - sample_start
        return {
            "model": runtime.model,
            "sample_index": sample_index,
            "exception": repr(exc),
            "failure_stage": "generation_exception",
            "failure_reason": str(exc),
            "wall_s": wall_s,
            "generation_s": wall_s,
            "eval_s": 0.0,
            "output_tokens": 0,
            "hit_token_cap": False,
            "format_ok": False,
            "compiled": False,
            "correctness": False,
        }

    eval_start = time.perf_counter()
    eval_result = await evaluate_kernel_async(
        level=problem.level,
        problem_id=problem.problem_id,
        backend=problem.backend,
        kernel_code=generated.kernel_code,
        dataset_src=problem.dataset_src,
        num_correct_trials=1,
        measure_performance=False,
        timeout=modal_timeout,
    )
    eval_s = time.perf_counter() - eval_start
    wall_s = time.perf_counter() - sample_start

    format_ok = bool(generated.format_ok and eval_result.get("format_ok"))
    compiled = bool(eval_result.get("compiled"))
    correctness = bool(eval_result.get("correctness"))
    failure_stage = "correct"
    failure_reason = ""
    if not format_ok:
        failure_stage = "format"
        failure_reason = eval_result.get("error_message") or "Could not extract valid kernel code"
    elif not compiled:
        failure_stage = "compile_or_eval"
        failure_reason = eval_result.get("error_message") or "Compilation/evaluation failed"
    elif not correctness:
        failure_stage = "incorrect"
        failure_reason = eval_result.get("error_message") or "Output mismatch"

    return {
        "model": runtime.model,
        "sample_index": sample_index,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "wall_s": wall_s,
        "generation_s": generated.generation_s,
        "eval_s": eval_s,
        "output_tokens": generated.token_count,
        "hit_token_cap": generated.token_count >= max_tokens,
        "kernel_len": len(generated.kernel_code),
        "format_ok": format_ok,
        "response_format_ok": generated.format_ok,
        "compiled": compiled,
        "correctness": correctness,
        "tests_passed": eval_result.get("tests_passed", 0),
        "tests_total": eval_result.get("tests_total", 1),
        "modal_eval_s": eval_result.get("metadata", {}).get("modal_eval_s"),
        "total_eval_s": eval_result.get("metadata", {}).get("total_eval_s"),
        "stop_condition": generated.stop_condition,
    }


def summarize(
    *,
    label: str,
    models: list[str],
    requested_by_model: dict[str, int],
    max_tokens: int,
    wall_s: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        model_results = [r for r in results if r.get("model") == model]
        model_wall = max((float(r.get("wall_s", 0.0)) for r in model_results), default=0.0)
        by_model[model] = {
            "requested_concurrency": requested_by_model.get(model, 0),
            "samples": len(model_results),
            "throughput_per_min_using_run_wall": len(model_results) / wall_s * 60 if wall_s else 0,
            "max_sample_wall_s": model_wall,
            "exceptions": sum(1 for r in model_results if r.get("exception")),
            "format_ok": sum(1 for r in model_results if r.get("format_ok")),
            "compiled": sum(1 for r in model_results if r.get("compiled")),
            "correct": sum(1 for r in model_results if r.get("correctness")),
            "hit_token_cap": sum(1 for r in model_results if r.get("hit_token_cap")),
            "avg_generation_s": average(r.get("generation_s") for r in model_results),
            "avg_eval_s": average(r.get("eval_s") for r in model_results),
        }

    return {
        "label": label,
        "models": models,
        "requested_by_model": requested_by_model,
        "requested_total_concurrency": sum(requested_by_model.values()),
        "max_tokens": max_tokens,
        "wall_s": wall_s,
        "throughput_per_min": total / wall_s * 60 if wall_s else 0,
        "exceptions": sum(1 for r in results if r.get("exception")),
        "format_ok": sum(1 for r in results if r.get("format_ok")),
        "compiled": sum(1 for r in results if r.get("compiled")),
        "correct": sum(1 for r in results if r.get("correctness")),
        "hit_token_cap": sum(1 for r in results if r.get("hit_token_cap")),
        "avg_generation_s": average(r.get("generation_s") for r in results),
        "avg_eval_s": average(r.get("eval_s") for r in results),
        "by_model": by_model,
        "results": results,
    }


def average(values: Any) -> float:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


async def run_condition(
    *,
    label: str,
    runtimes: dict[str, ModelRuntime],
    problem: KernelBenchProblem,
    requested_by_model: dict[str, int],
    max_tokens: int,
    temperature: float,
    modal_timeout: float,
) -> dict[str, Any]:
    LOGGER.info("Starting %s: %s", label, requested_by_model)
    tasks = []
    for model, count in requested_by_model.items():
        for sample_index in range(count):
            tasks.append(
                run_sample(
                    runtime=runtimes[model],
                    problem=problem,
                    sample_index=sample_index,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    modal_timeout=modal_timeout,
                )
            )

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - start
    summary = summarize(
        label=label,
        models=list(requested_by_model),
        requested_by_model=requested_by_model,
        max_tokens=max_tokens,
        wall_s=wall_s,
        results=results,
    )
    LOGGER.info(
        "Finished %s: wall=%.2fs throughput=%.2f/min exceptions=%d format=%d compiled=%d correct=%d",
        label,
        summary["wall_s"],
        summary["throughput_per_min"],
        summary["exceptions"],
        summary["format_ok"],
        summary["compiled"],
        summary["correct"],
    )
    return summary


def make_runtime(service_client: tinker.ServiceClient, model: str) -> ModelRuntime:
    renderer_name = get_renderer_name_for_model(model)
    tokenizer = tokenizer_utils.get_tokenizer(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return ModelRuntime(
        model=model,
        sampling_client=service_client.create_sampling_client(base_model=model),
        renderer=renderer,
    )


async def main_async(args: argparse.Namespace) -> None:
    setup_environment()
    set_modal_evaluator(
        ModalKernelEvaluator(
            ModalEvaluatorConfig(enabled=True, gpu_type=args.modal_gpu_type, timeout=args.modal_timeout)
        )
    )

    service_client = tinker.ServiceClient(base_url=args.base_url)
    models = args.models
    runtimes = {model: make_runtime(service_client, model) for model in models}
    problem = KernelBenchProblem(
        level=args.level,
        problem_id=args.problem_id,
        backend=args.backend,
        dataset_src=args.dataset_src,
        prompt_option=args.prompt_option,
        prompt_precision=args.precision,
        prompt_include_hardware=args.prompt_include_hardware,
        prompt_gpu_name=args.prompt_gpu_name,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for model in models:
            for concurrency in (4, 8):
                summary = await run_condition(
                    label=f"{short_model_name(model)}_c{concurrency}",
                    runtimes=runtimes,
                    problem=problem,
                    requested_by_model={model: concurrency},
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    modal_timeout=args.modal_timeout,
                )
                f.write(json.dumps(summary, default=str) + "\n")
                f.flush()

        mixed = await run_condition(
            label="mixed_4_plus_4",
            runtimes=runtimes,
            problem=problem,
            requested_by_model={models[0]: 4, models[1]: 4},
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            modal_timeout=args.modal_timeout,
        )
        f.write(json.dumps(mixed, default=str) + "\n")


def short_model_name(model: str) -> str:
    return (
        model.lower()
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "")
        .replace("-", "_")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs=2,
        default=["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3-30B-A3B-Instruct-2507"],
    )
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem-id", type=int, default=1)
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--dataset-src", default="huggingface")
    parser.add_argument("--prompt-option", default="one_shot")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--prompt-include-hardware", action="store_true")
    parser.add_argument("--prompt-gpu-name")
    parser.add_argument("--max-tokens", type=int, default=64245)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--modal-gpu-type", default="A100")
    parser.add_argument("--modal-timeout", type=float, default=120.0)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("runs/validation/concurrency_sweep/two_model_l1p1_c4_c8_mixed.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
