#!/usr/bin/env python3
"""Пауза во время ожидания памяти не запускает следующий блок."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as jr


class PauseTests(unittest.TestCase):
    def test_cancelled_wait_does_not_say_memory_is_free(self):
        jr.cancelled = True
        try:
            self.assertFalse(jr.wait_for_memory())
        finally:
            jr.cancelled = False


if __name__ == "__main__":
    unittest.main()
