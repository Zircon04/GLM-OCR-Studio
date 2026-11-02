#!/usr/bin/env python3
"""Standalone export regressions: .venv-tests/bin/python scripts/test_export_assets.py."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('TMPDIR', str(ROOT.parent)))
sys.path.insert(0, str(ROOT / 'manager'))
from export_results import markdown_to_html, rebase_markdown_assets


class ExportAssetsTests(unittest.TestCase):
    def test_preserves_math_literals_nonimage_text_and_external_urls(self):
        unchanged = r'''LaTeX: $\frac{a}{b}$ and \(\alpha + \beta\).
$$
\begin{aligned} x &= \text{imgs/one.png} \\
y &= 2
\end{aligned}
$$
\[ ![not an image](imgs/math.png) \]
Prose imgs/plain.png and [ordinary](imgs/plain.png).
`![inline](imgs/code.png)` and ``<img src="imgs/code.png">``
```markdown
![fenced](imgs/code.png)
```
~~~html
<img src='imgs/code.png'>
~~~
    ![indented](imgs/code.png)
<!-- <img src="imgs/comment.png"> -->
<pre><img src="imgs/code.png"></pre>
\![escaped](imgs/escape.png)
![remote](https://example.com/imgs/a.png?x=1#part)
![protocol](//example.com/imgs/a.png)
![absolute](/var/assets/image.png)
![windows](C:\assets\image.png)
![data](data:image/png;base64,ABC)
<img src='/absolute/image.png'>
'''
        root, output = Path('/work/Исходник с пробелами'), Path('/work/exports')
        self.assertEqual(rebase_markdown_assets(unchanged, root, output), unchanged)

    def test_html_attributes_only_rewrite_real_src(self):
        text = '''<img alt="literal src='imgs/not-real.png' ![example](imgs/not-real.png)" data-src="imgs/lazy.png" src='./imgs/real.png' title="imgs/title.png">'''
        actual = rebase_markdown_assets(text, Path('/work/result'), Path('/work/exports'))
        self.assertEqual(actual, text.replace("src='./imgs/real.png'", "src='../result/imgs/real.png'"))

    def test_encoded_paths_queries_and_html_api(self):
        text = '![x](imgs/a%20b.png?size=2#crop)\n<img src="./imgs/рисунок.png">'
        actual = rebase_markdown_assets(text, Path('/work/result'), Path('/work/exports/deeper'))
        self.assertIn('../../result/imgs/a%20b.png?size=2#crop', actual)
        self.assertNotIn('%2520', actual)
        for kwargs in ({}, {'asset_root': Path('/work/result'), 'output_dir': Path('/work/exports')}):
            rendered = markdown_to_html('![x](./imgs/a.png)', 'result', **kwargs)
            self.assertIn('src="../result/imgs/a.png"', rendered)

    def test_asset_map_rewrites_only_mapped_images_preserving_literals(self):
        root = Path('/work/scan with spaces')
        source = root / 'imgs/one crop.png'
        optimized = root / 'imgs_enhanced/one crop.webp'
        text = ('![crop](<imgs/one%20crop.png?size=2#detail>)\n'
                '<img src="imgs/one%20crop.png?size=2#detail">\n'
                '![unmapped](imgs/other.png)\n'
                '`![literal](imgs/one%20crop.png)`\n'
                '$$ ![math](imgs/one%20crop.png) \\alpha $$\n')
        actual = rebase_markdown_assets(text, root, Path('/work/exports'),
                                       asset_map={source.resolve(): optimized.resolve()})
        expected = text.replace('<imgs/one%20crop.png?size=2#detail>',
                                '<../scan%20with%20spaces/imgs_enhanced/one%20crop.webp?size=2#detail>')
        expected = expected.replace('src="imgs/one%20crop.png?size=2#detail"',
                                    'src="../scan%20with%20spaces/imgs_enhanced/one%20crop.webp?size=2#detail"')
        self.assertEqual(actual, expected)

    def test_exported_images_resolve_from_each_output_directory(self):
        with tempfile.TemporaryDirectory(prefix='export-assets-', dir=SCRATCH) as temp:
            root = Path(temp) / 'Выход с пробелами'
            result = root / 'Документ тест' / 'nested OCR'
            result.mkdir(parents=True)
            for name in ('imgs/рисунок 1.png', 'imgs/second.png', 'imgs_enhanced/crop.png'):
                asset = result / name
                asset.parent.mkdir(exist_ok=True)
                asset.write_bytes(b'asset fixture')
            markdown = (
                '# Chapter one\n'
                '![figure](<imgs/рисунок 1.png>)\n'
                '![second](./imgs/second.png "Caption")\n'
                '<img alt="crop" src="./imgs_enhanced/crop.png">\n'
                '# Chapter two\n'
                '![again](imgs/second.png)\n'
            )
            (result / 'ocr.md').write_text(markdown, encoding='utf-8')
            (result / 'ocr.json').write_text(json.dumps({}), encoding='utf-8')
            process = subprocess.run([
                sys.executable, str(ROOT / 'manager/export_results.py'),
                '--result', str(result.parent), '--source', str(root / 'unused.pdf'),
                '--output-root', str(root), '--formats', 'md,split_md',
            ], text=True, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            exports = root / 'exports'
            # Keep low-level HTML rebasing regression coverage, not a CLI format.
            (exports / 'legacy.html').write_text(markdown_to_html(
                markdown, result.parent.name, asset_root=result, output_dir=exports), encoding='utf-8')
            outputs = list(exports.glob('*.md')) + list(exports.glob('*_chapters/*.md')) + list(exports.glob('*.html'))
            self.assertEqual(len(outputs), 4)
            for output in outputs:
                text = output.read_text(encoding='utf-8')
                urls = re.findall(r'(?:src=[\"\'])(.*?)[\"\']', text)
                if output.suffix == '.md':
                    urls += re.findall(r'!\[[^\]]*\]\(<?([^\s>)]+)', text)
                self.assertTrue(urls, output)
                for url in urls:
                    path = output.parent / unquote(urlsplit(html.unescape(url)).path)
                    self.assertTrue(path.is_file(), f'{output}: {url} -> {path}')
            self.assertEqual((result / 'ocr.md').read_text(encoding='utf-8'), markdown)
            self.assertFalse((exports / 'imgs').exists(), 'Do not copy asset directories')


if __name__ == '__main__':
    unittest.main(verbosity=2)
