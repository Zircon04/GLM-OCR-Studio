#!/usr/bin/env python3
"""Regression tests for real formula rendering and lossless visual fallback."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import reconstruct_pdf as pdf


class FormulaTests(unittest.TestCase):
    def test_basic_fraction_is_rendered_as_image(self):
        data = pdf.formula_png(r"$$f_0=\frac{1}{2\pi\sqrt{LC}}$$")
        self.assertTrue(data and data.startswith(b"\x89PNG"), "Matplotlib renderer unavailable")

    def test_unrenderable_formula_preserves_original_region(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "scan.pdf"
            target = root / "clean.pdf"
            metadata = root / "result.json"
            document = pymupdf.open()
            page = document.new_page(width=500, height=500)
            page.draw_rect(pymupdf.Rect(50, 50, 450, 150), color=(0, 0, 1), fill=(0, 0, 1))
            document.save(source)
            document.close()
            content = r"$$\begin{aligned} a&=b \\ c&=d \end{aligned}$$"
            metadata.write_text(json.dumps([[{"label": "formula", "content": content, "bbox_2d": [100,100,900,300]}]]))
            font_dir = pdf.FONT_DIR if pdf.FONT_DIR.exists() else Path("/usr/share/fonts/truetype/dejavu")
            with patch.object(pdf, "formula_png", return_value=None), patch.object(pdf, "FONT_DIR", font_dir):
                pdf.build(source, metadata, root, target)
            with pymupdf.open(target) as output:
                self.assertTrue(output[0].get_images(), "Unsupported formula replaced with raw LaTeX instead of original crop")
                pixel = output[0].get_pixmap().pixel(250, 100)
                self.assertGreater(pixel[2], 230)
                self.assertLess(pixel[0], 20)
                self.assertIn("aligned", output[0].get_text(), "Searchable LaTeX was lost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
