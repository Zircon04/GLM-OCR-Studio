#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
from export_results import rewrite_image_links


class ExportAssetTests(unittest.TestCase):
    def test_links_are_relative_to_each_output_and_math_is_kept(self):
        with tempfile.TemporaryDirectory() as raw:
            book = Path(raw) / "Book"
            (book / "imgs").mkdir(parents=True)
            image = book / "imgs" / "рис 1.png"
            image.write_bytes(b"x")
            markdown = "$$\nimgs/рис 1.png\n$$\n\n![](imgs/рис%201.png)\n"
            export = Path(raw) / "exports" / "Book.md"
            export.parent.mkdir()
            rewritten = rewrite_image_links(markdown, book, export)
            self.assertIn("imgs/рис 1.png", rewritten.split("$$")[1])
            self.assertIn("../Book/imgs/", rewritten)
            chapter = Path(raw) / "exports" / "Book_chapters" / "01.md"
            chapter.parent.mkdir()
            chapter_text = rewrite_image_links("![](imgs/рис%201.png)\n", book, chapter)
            self.assertTrue(chapter_text.startswith("![](../../Book/imgs/"))


if __name__ == "__main__":
    unittest.main()
