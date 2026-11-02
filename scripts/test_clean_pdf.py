#!/usr/bin/env python3
import unittest
from pathlib import Path


class CleanPdfTests(unittest.TestCase):
    def test_formula_fallback_and_font_subset_remain(self):
        text = Path(__file__).resolve().parents[1].joinpath("manager/reconstruct_pdf.py").read_text(encoding="utf-8")
        self.assertIn("get_pixmap", text)
        self.assertIn("subset_fonts", text)
        searchable = Path(__file__).resolve().parents[1].joinpath("manager/searchable_pdf.py").read_text(encoding="utf-8")
        self.assertIn("subset_fonts", searchable)
        self.assertIn("jpeg_quality: int | None = 0", searchable)


if __name__ == "__main__":
    unittest.main()
