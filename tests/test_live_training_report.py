from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_training_report.py"
SPEC = importlib.util.spec_from_file_location("live_training_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
live_training_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live_training_report)


def write_jsonl(path: Path, records: list[dict], extra: str = "") -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records) + extra,
        encoding="utf-8",
    )


class LiveTrainingReportTests(unittest.TestCase):
    def test_collect_status_handles_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status = live_training_report.collect_status(Path(temp_dir) / "missing_run")

        self.assertEqual(status["state"], "waiting")
        self.assertEqual(status["metrics"]["completed_updates"], 0)
        self.assertEqual(status["traces"]["total"], 0)
        self.assertEqual(status["checkpoints"]["count"], 0)

    def test_collect_status_summarizes_synthetic_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_name": "moonshotai/Kimi-K2.6:peft:131072",
                        "max_tokens": 120000,
                        "remove_constant_reward_groups": True,
                        "dataset_builder": {
                            "level": 1,
                            "start_problem": 1,
                            "end_problem": 1,
                            "batch_size": 1,
                            "group_size": 4,
                            "num_epochs": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                run_dir / "metrics.jsonl",
                [
                    {
                        "step": 0,
                        "reward/mean": 0.8,
                        "kernel/format_rate": 1.0,
                        "kernel/compile_rate": 0.5,
                        "kernel/correct_rate": 0.25,
                    }
                ],
            )
            write_jsonl(
                run_dir / "traces.jsonl",
                [
                    {
                        "timestamp": 10,
                        "level": 1,
                        "problem_id": 1,
                        "reward": 0.3,
                        "response": {"raw": "abc", "kernel": "class ModelNew: pass", "format_ok": True},
                        "eval_result": {"compiled": True, "correctness": True, "speedup": 2.0},
                        "metrics": {"format_ok": 1.0, "compiled": 1.0, "correctness": 1.0},
                    },
                    {
                        "timestamp": 11,
                        "level": 1,
                        "problem_id": 1,
                        "reward": 0.0,
                        "response": {"raw": "bad", "kernel": "", "format_ok": False},
                        "eval_result": {
                            "compiled": False,
                            "correctness": False,
                            "error_message": "format failure",
                        },
                        "metrics": {"format_ok": 0.0, "compiled": 0.0, "correctness": 0.0},
                    },
                ],
                extra="{partial",
            )
            write_jsonl(run_dir / "checkpoints.jsonl", [{"name": "000001"}, {"name": "final"}])
            (run_dir / "logs.log").write_text(
                "Batch 2: All groups filtered out, skipping\nERROR example\n",
                encoding="utf-8",
            )

            status = live_training_report.collect_status(run_dir)

        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["expected_batches"], 100)
        self.assertEqual(status["metrics"]["completed_updates"], 1)
        self.assertEqual(status["traces"]["total"], 2)
        self.assertEqual(status["traces"]["correct_rate"], 0.5)
        self.assertEqual(status["traces"]["mean_speedup"], 2.0)
        self.assertEqual(status["logs"]["skipped_batches"], 1)
        self.assertEqual(status["parse_errors"]["traces_jsonl"], 1)

    def test_collect_status_reads_existing_smoke_artifacts(self) -> None:
        run_dir = ROOT / "runs" / "validation" / "train_smoke_qwen36_a3b"
        status = live_training_report.collect_status(run_dir)

        self.assertGreaterEqual(status["metrics"]["completed_updates"], 1)
        self.assertGreaterEqual(status["traces"]["total"], 1)
        self.assertGreaterEqual(status["checkpoints"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
