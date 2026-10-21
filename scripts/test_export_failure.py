#!/usr/bin/env python3
import unittest
from pathlib import Path


class ExportFailureTests(unittest.TestCase):
    def test_failed_export_returns_before_cleanup(self):
        text = Path(__file__).resolve().parents[1].joinpath("manager/job_runner.py").read_text(encoding="utf-8")
        export_error = text.split("Экспорт завершился с кодом", 1)[1]
        self.assertIn("return code or 1", export_error.split("def ", 1)[0])
        self.assertLess(text.index("Экспорт завершился с кодом"), text.index("optimize_tree("))


if __name__ == "__main__":
    unittest.main()
