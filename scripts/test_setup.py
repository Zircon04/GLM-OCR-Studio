#!/usr/bin/env python3
import unittest
from pathlib import Path


class SetupTests(unittest.TestCase):
    def test_setup_reconciles_existing_environments(self):
        text = Path(__file__).resolve().parents[1].joinpath("setup.command").read_text(encoding="utf-8")
        self.assertIn("ensurepip", text)
        self.assertIn("ensure_env", text)
        self.assertNotIn("rm -rf .venv", text)


if __name__ == "__main__":
    unittest.main()
