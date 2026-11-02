#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as jr


class FinalizationTests(unittest.TestCase):
    def test_enhanced_markdown_becomes_main_and_both_image_dirs_stay(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Book"
            (root / "imgs").mkdir(parents=True)
            (root / "imgs_enhanced").mkdir()
            (root / "imgs" / "a.png").write_bytes(b"png")
            (root / "imgs_enhanced" / "a.png").write_bytes(b"enh")
            (root / "Book.md").write_text("old", encoding="utf-8")
            (root / "Book_enhanced.md").write_text("new ![](imgs_enhanced/a.png)", encoding="utf-8")
            (root / "notes.json").write_text("{}", encoding="utf-8")
            jr.finalize_book_dir(root, "Book")
            self.assertEqual((root / "Book.md").read_text(encoding="utf-8"), "new ![](imgs_enhanced/a.png)")
            self.assertFalse((root / "Book_enhanced.md").exists())
            self.assertTrue((root / "imgs" / "a.png").is_file())
            self.assertTrue((root / "imgs_enhanced" / "a.png").is_file())
            self.assertFalse((root / "notes.json").exists())


if __name__ == "__main__":
    unittest.main()
