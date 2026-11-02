#!/usr/bin/env python3
"""Image enhancement and exports must survive final workspace cleanup."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import export_results as export
import job_runner as runner


class FinalizationTests(unittest.TestCase):
    def fixture(self, root):
        book = root / "scan"
        (book / "imgs").mkdir(parents=True)
        (book / "imgs_enhanced").mkdir()
        (book / "imgs/original.png").write_bytes(b"original")
        (book / "imgs_enhanced/restored.png").write_bytes(b"restored")
        (book / "scan.md").write_text("# Book\n![Original](imgs/original.png)\n")
        enhanced = "# Book\n![Restored](imgs_enhanced/restored.png)\n\n$$x=1$$\n"
        (book / "scan_enhanced.md").write_text(enhanced)
        (book / "scan.json").write_text("[[]]")
        return book, enhanced

    def test_final_markdown_uses_restoration_and_keeps_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book, enhanced = self.fixture(root)
            runner.finalize_book_dir(book, "scan")
            self.assertEqual((book / "scan.md").read_text(), enhanced)
            self.assertTrue((book / "imgs_enhanced/restored.png").is_file())
            self.assertEqual((book / "imgs/original.png").read_bytes(), b"original")
            self.assertFalse((book / "scan_enhanced.md").exists())
            self.assertFalse((book / "scan.json").exists())

    def test_export_links_survive_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book, _ = self.fixture(root)
            argv = ["export_results.py", "--result", str(book), "--source", str(root / "scan.pdf"),
                    "--output-root", str(root), "--formats", "md,split_md"]
            with patch.object(sys, "argv", argv):
                self.assertEqual(export.main(), 0)
            # Legacy renderer is still supported internally, not as a CLI export.
            (root / "exports/legacy.html").write_text(export.markdown_to_html(
                (book / "scan_enhanced.md").read_text(), "scan", asset_root=book,
                output_dir=root / "exports"))
            runner.finalize_book_dir(book, "scan")
            outputs = list((root / "exports").rglob("*.md")) + list((root / "exports").glob("*.html"))
            for output in outputs:
                text = output.read_text()
                paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) if output.suffix == ".md" else re.findall(r'src="([^"]+)"', text)
                self.assertTrue(paths)
                for path in paths:
                    self.assertTrue((output.parent / unquote(path)).is_file(), f"Broken after cleanup: {output}: {path}")



if __name__ == "__main__":
    unittest.main(verbosity=2)
