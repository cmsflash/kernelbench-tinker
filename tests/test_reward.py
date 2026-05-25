from __future__ import annotations

import unittest

from kernelbench_tinker.training.reward import (
    RewardConfig,
    compile_reward,
    compute_reward,
    correctness_reward,
    speed_reward,
)


def eval_result(**overrides):
    result = {
        "format_ok": True,
        "compiled": True,
        "correctness": False,
        "tests_passed": 0,
        "tests_total": 5,
        "speedup": None,
        "runtime_ms": None,
        "baseline_runtime_ms": None,
        "error_message": None,
        "code_length": 1000,
        "metadata": {},
    }
    result.update(overrides)
    return result


class RewardTests(unittest.TestCase):
    def test_format_failure_gets_negative_reward(self) -> None:
        reward = compute_reward(eval_result(format_ok=False, compiled=False))

        self.assertEqual(reward, -2.0)

    def test_compile_failure_gets_negative_reward(self) -> None:
        result = eval_result(compiled=False)

        self.assertEqual(compile_reward(result, RewardConfig()), -1.0)
        self.assertEqual(compute_reward(result), -1.0)

    def test_correctness_is_partial_by_default(self) -> None:
        result = eval_result(tests_passed=2, tests_total=5)

        self.assertEqual(correctness_reward(result, RewardConfig()), 0.4)
        self.assertAlmostEqual(compute_reward(result), 0.12)

    def test_full_correctness_does_not_add_speed_reward(self) -> None:
        result = eval_result(
            correctness=True,
            tests_passed=5,
            tests_total=5,
            speedup=3.0,
        )

        self.assertEqual(speed_reward(result, RewardConfig()), 2.0)
        self.assertEqual(compute_reward(result), 0.3)

    def test_incomplete_correctness_can_add_speed_on_max_correctness_reward(self) -> None:
        result = eval_result(
            correctness=False,
            tests_passed=3,
            tests_total=5,
            speedup=2.5,
        )

        self.assertEqual(speed_reward(result, RewardConfig()), 1.5)
        self.assertEqual(compute_reward(result), 1.8)


if __name__ == "__main__":
    unittest.main()
