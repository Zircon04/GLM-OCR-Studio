#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as jr


class ExportSelectionTests(unittest.TestCase):
    def test_default_three_formats(self):
        self.assertEqual(jr.parse_export_formats("md,split_md,searchable_pdf"), ["md", "split_md", "searchable_pdf"])

    def test_empty_and_unknown_are_rejected(self):
        with self.assertRaises(ValueError):
            jr.parse_export_formats("  ")
        with self.assertRaises(ValueError):
            jr.parse_export_formats("html,docx")


if __name__ == "__main__":
    unittest.main()
