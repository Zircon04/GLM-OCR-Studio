#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as jr


class MemoryProfileTests(unittest.TestCase):
    def test_floor_is_the_larger_of_reserve_and_ten_percent(self):
        total = 16 * 1024 ** 3
        self.assertEqual(jr.admission_floor(total, 3), 3 * 1024 ** 3)
        self.assertEqual(jr.admission_floor(total, 1), int(total * 0.10))

    def test_user_reserve_is_not_lowered(self):
        total = 16 * 1024 ** 3
        self.assertGreater(jr.admission_floor(total, 7), 3 * 1024 ** 3)

    def test_missing_telemetry_does_not_allow_the_block(self):
        self.assertIsNone(jr.memory_allows_block(3, available=None, total=0))
        original = jr.memory_snapshot
        jr.memory_snapshot = lambda: None
        try:
            self.assertIsNone(jr.memory_allows_block(3))
        finally:
            jr.memory_snapshot = original

    def test_cancel_during_wait_returns_immediately(self):
        jr.cancelled = True
        try:
            self.assertFalse(jr.wait_for_memory())
        finally:
            jr.cancelled = False


if __name__ == "__main__":
    unittest.main()
