#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
from PIL import Image

from optimize_images import optimize_tree, rewrite_names


class ImageOptimizationTests(unittest.TestCase):
    def test_formula_text_is_not_rewritten(self):
        text = "$$\nimgs_enhanced/a.png\n$$\n\n![](imgs_enhanced/a.png)\n"
        changed = rewrite_names(text, {"imgs_enhanced/a.png": "imgs_enhanced/a.webp"})
        self.assertIn("$$\nimgs_enhanced/a.png\n$$", changed)
        self.assertIn("![](imgs_enhanced/a.webp)", changed)

    def test_smaller_lossless_webp_replaces_enhanced_png(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            folder = root / "Book" / "imgs_enhanced"
            folder.mkdir(parents=True)
            image = Image.new("RGBA", (48, 48), (10, 20, 30, 255))
            png = folder / "a.png"
            image.save(png)
            original = image.tobytes()
            (root / "Book" / "Book.md").write_text("![](imgs_enhanced/a.png)\n", encoding="utf-8")
            stats = optimize_tree(root)
            self.assertGreaterEqual(stats["saved_bytes"], 0)
            webp = folder / "a.webp"
            if webp.exists():
                self.assertFalse(png.exists())
                with Image.open(webp) as packed:
                    self.assertEqual(packed.convert("RGBA").tobytes(), original)
                self.assertIn("a.webp", (root / "Book" / "Book.md").read_text(encoding="utf-8"))
            else:
                self.assertTrue(png.exists())


if __name__ == "__main__":
    unittest.main()
