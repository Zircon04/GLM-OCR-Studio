#!/usr/bin/env python3
"""Lossless post-enhancement image optimization regressions."""
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))


class OptimizationTests(unittest.TestCase):
    def test_small_png_is_never_replaced_by_a_larger_file(self):
        from optimize_images import optimize_image
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tiny.png"
            Image.new("RGB", (1, 1), "white").save(source, optimize=True)
            before = source.read_bytes()
            output = optimize_image(source)
            self.assertEqual(output, source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list(source.parent.glob("*.part")), [])

    def test_sixteen_bit_png_is_not_quantized(self):
        from optimize_images import optimize_image
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "depth.png"
            Image.new("I;16", (24, 24), 12345).save(source)
            before = source.read_bytes()
            self.assertEqual(optimize_image(source), source)
            self.assertEqual(source.read_bytes(), before)

    def test_existing_different_webp_is_not_overwritten(self):
        from optimize_images import optimize_image
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "figure.png"
            Image.new("RGB", (80, 80), "white").save(source, compress_level=0)
            target = source.with_name(source.name + ".lossless.webp")
            Image.new("RGB", (80, 80), "black").save(target, lossless=True)
            before = target.read_bytes()
            optimize_image(source)
            self.assertEqual(target.read_bytes(), before)

    def test_reference_style_links_do_not_lose_their_original_image(self):
        from optimize_images import optimize_book_images
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); book = root / "book"
            images = book / "imgs_enhanced"; images.mkdir(parents=True)
            source = images / "diagram.png"
            Image.new("RGB", (400, 300), "white").save(source, compress_level=0)
            markdown = book / "book.md"
            markdown.write_text("![inline](imgs_enhanced/diagram.png)\n![ref][x]\n[x]: imgs_enhanced/diagram.png\n")
            optimize_book_images(book, root)
            self.assertTrue(source.exists(), "Unsupported reference syntax must not become a broken link")
            self.assertNotIn(".lossless.webp", markdown.read_text())

    def test_failed_link_write_retains_original_and_allows_retry(self):
        from optimize_images import optimize_book_images
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); book = root / "book"
            images = book / "imgs_enhanced"; images.mkdir(parents=True)
            source = images / "diagram.png"
            image = Image.new("RGB", (400, 300), "white")
            image.save(source, compress_level=0)
            markdown = book / "book.md"; markdown.write_text("![A](imgs_enhanced/diagram.png)\n")
            with patch("optimize_images._atomic_text", side_effect=OSError("disk write interrupted")):
                with self.assertRaises(OSError):
                    optimize_book_images(book, root)
            self.assertTrue(source.exists())
            self.assertTrue(source.with_name(source.name + ".lossless.webp").exists())
            optimize_book_images(book, root)
            self.assertFalse(source.exists())
            self.assertIn(".lossless.webp", markdown.read_text())

    def test_book_rewrites_export_links_before_deleting_old_images(self):
        from optimize_images import optimize_book_images
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book = root / "book"; images = book / "imgs_enhanced"; images.mkdir(parents=True)
            original = images / "рис 1.png"
            image = Image.new("RGB", (480, 320), "white")
            ImageDraw.Draw(image).line((10, 10, 450, 300), fill="black", width=2)
            image.save(original, compress_level=0)
            (book / "book_enhanced.md").write_text("![A](<imgs_enhanced/рис 1.png>)\n$$x=1$$\n")
            chapters = root / "exports/book_chapters"; chapters.mkdir(parents=True)
            chapter = chapters / "01.md"
            chapter.write_text("![A](../../book/imgs_enhanced/рис%201.png?view=1#diagram)\n")
            stats = optimize_book_images(book, root)
            self.assertGreater(stats["saved_bytes"], 0)
            self.assertFalse(original.exists())
            text = chapter.read_text()
            self.assertIn(".lossless.webp?view=1#diagram", text)
            self.assertIn("$$x=1$$", (book / "book_enhanced.md").read_text())
            from urllib.parse import unquote
            target = text.split("](", 1)[1].split("?", 1)[0]
            with Image.open(chapter.parent / unquote(target)) as decoded:
                self.assertEqual(decoded.tobytes(), image.tobytes())

    def test_smaller_lossless_result_preserves_pixels_dimensions_and_alpha(self):
        from optimize_images import optimize_image
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "схема 1.png"
            image = Image.new("RGBA", (480, 320), (17, 29, 41, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 450, 290), outline=(0, 0, 0, 255), width=2)
            draw.line((20, 150, 450, 150), fill=(255, 0, 0, 255), width=1)
            image.save(source, compress_level=0)
            before = source.stat().st_size
            output = optimize_image(source)
            self.assertLess(output.stat().st_size, before)
            self.assertTrue(source.exists(), "References must be updated before deleting original")
            with Image.open(output) as restored:
                self.assertEqual(restored.size, image.size)
                self.assertEqual(restored.convert("RGBA").tobytes(), image.tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
