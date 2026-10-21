#!/usr/bin/env python3
import unittest
from pathlib import Path


class JobOptimizationOrderTests(unittest.TestCase):
    def test_optimize_happens_after_export_and_before_cleanup(self):
        text = Path(__file__).resolve().parents[1].joinpath("manager/job_runner.py").read_text(encoding="utf-8")
        export_at = text.index('stage="export"')
        optimize_at = text.index("optimize_tree(")
        cleanup_at = text.index("finalize_book_dir(", optimize_at)
        self.assertLess(export_at, optimize_at)
        self.assertLess(optimize_at, cleanup_at)


if __name__ == "__main__":
    unittest.main()
