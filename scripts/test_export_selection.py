#!/usr/bin/env python3
"""Reduced export selection and original-scan preservation regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('TMPDIR', str(ROOT.parent)))


class ExportSelectionTests(unittest.TestCase):
    def run_export(self, root, formats=None):
        command = [sys.executable, str(ROOT / 'manager/export_results.py'),
                   '--result', str(root / 'scan'), '--source', str(root / 'scan.pdf'),
                   '--output-root', str(root)]
        if formats is not None:
            command += ['--formats', formats]
        return subprocess.run(command, text=True, capture_output=True)

    def fixture(self, root):
        result = root / 'scan'
        result.mkdir()
        (result / 'scan.md').write_text('# Scan\nSearchable colour scan\n')
        (result / 'scan.json').write_text(json.dumps([[{
            'label': 'text', 'content': 'Searchable colour scan',
            'bbox_2d': [50, 50, 950, 180],
        }]]))
        return result

    def test_searchable_export_preserves_original_raster_pixels_and_colours(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root = Path(temp)
            self.fixture(root)
            # High-frequency, non-grey scan catches JPEG damage and colour loss.
            width, height = 96, 80
            pixels = bytes((x * 31 + y * 17 + channel * 83) % 256
                           for y in range(height) for x in range(width) for channel in range(3))
            raster = pymupdf.Pixmap(pymupdf.csRGB, width, height, pixels, False)
            with pymupdf.open() as doc:
                page = doc.new_page(width=width, height=height)
                page.insert_image(page.rect, pixmap=raster)
                doc.save(root / 'scan.pdf')
            original_bytes = (root / 'scan.pdf').read_bytes()
            process = self.run_export(root, 'searchable_pdf')
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            with pymupdf.open(root / 'scan.pdf') as source, pymupdf.open(root / 'exports/scan_searchable.pdf') as output:
                self.assertEqual(output[0].rect, source[0].rect)
                self.assertEqual(output[0].get_pixmap().samples, source[0].get_pixmap().samples)
                src_image = pymupdf.Pixmap(source, source[0].get_images()[0][0])
                out_image = pymupdf.Pixmap(output, output[0].get_images()[0][0])
                self.assertEqual((out_image.width, out_image.height, out_image.n), (width, height, 3))
                self.assertEqual(out_image.samples, src_image.samples)
                self.assertIn('Searchable colour scan', output[0].get_text())
                self.assertTrue(output[0].search_for('Searchable'))
            self.assertEqual((root / 'scan.pdf').read_bytes(), original_bytes)

    def test_cli_defaults_to_exactly_three_supported_exports(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root = Path(temp)
            self.fixture(root)
            with pymupdf.open() as doc:
                doc.new_page()
                doc.save(root / 'scan.pdf')
            process = self.run_export(root)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            events = [json.loads(line.removeprefix('GLMOCR_EVENT '))
                      for line in process.stdout.splitlines() if line.startswith('GLMOCR_EVENT ')]
            started = next(event for event in events if event['type'] == 'export_started')
            self.assertEqual(started['formats'], ['md', 'split_md', 'searchable_pdf'])
            self.assertEqual({p.name for p in (root / 'exports').iterdir()},
                             {'scan.md', 'scan_chapters', 'scan_searchable.pdf'})

    def test_gui_exposes_only_supported_exports_enabled_by_default(self):
        source = (ROOT / 'gui/main.swift').read_text()
        selected = source.split('private func selectedExportFormats()', 1)[1].split('private func formatTitle', 1)[0]
        self.assertEqual(re.findall(r'formats.append\("([^"]+)"\)', selected),
                         ['md', 'split_md', 'searchable_pdf'])
        for obsolete in ('htmlCheckbox', 'docxCheckbox', 'cleanPDFCheckbox', 'zipCheckbox',
                         '#selector(openHTML)', '#selector(openDOCX)', '#selector(openZIP)'):
            self.assertNotIn(obsolete, source)
        enabled = re.search(r'for checkbox in \[([^\]]+)\]\s*\{\s*checkbox\?\.state = \.on', source)
        self.assertIsNotNone(enabled)
        self.assertEqual([name.strip() for name in enabled[1].split(',')],
                         ['fullMDCheckbox', 'splitMDCheckbox', 'searchablePDFCheckbox'])
        buttons = source.split('for (name, icon, selector) in [', 1)[1].split('] {', 1)[0]
        self.assertEqual(re.findall(r'#selector\(([^)]+)\)', buttons),
                         ['openMarkdown', 'openSplitMarkdown', 'openSearchablePDF'])
        self.assertIn('openExport("_searchable.pdf")', source)
        self.assertIn('"webp"', source.split('private func supported(', 1)[1])

    def test_gui_reserves_optimization_tail_and_decodes_preview_with_imageio(self):
        source = (ROOT / 'gui/main.swift').read_text()
        for event in ('optimization_started', 'optimization_progress', 'optimization_done'):
            self.assertIn('case "' + event + '"', source)
        self.assertIn('stage == "optimize"', source)
        self.assertIn('let optimizationShare = 0.03', source)
        export_done = source.split('case "export_done":', 1)[1].split('case ', 1)[0]
        self.assertNotIn('0.99', export_done)
        self.assertIn('exportEnd', export_done)
        self.assertIn('CGImageSourceCreateWithURL', source)
        self.assertIn('CGImageSourceCreateImageAtIndex', source)
        self.assertIn('originalImage.image = previewImage(original)', source)
        self.assertIn('enhancedImage.image = previewImage(enhanced)', source)

    def test_cli_rejects_unsupported_or_empty_selection_without_success(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            root = Path(temp)
            self.fixture(root)
            for formats in ('html', 'docx', 'clean_pdf', 'zip', 'unknown', 'md,unknown', '', ', ,'):
                with self.subTest(formats=formats):
                    process = self.run_export(root, formats)
                    self.assertNotEqual(process.returncode, 0, process.stdout)
                    self.assertNotIn('"type": "export_done"', process.stdout)
                    self.assertFalse((root / 'exports').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
