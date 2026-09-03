#!/usr/bin/env python3
"""Return a compact JSON description of one GLM-OCR result for the native GUI."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def first_file(root: Path, pattern: str, exclude: str = "") -> Path | None:
    files = [p for p in root.rglob(pattern) if p.is_file() and ".job" not in p.parts]
    if exclude:
        files = [p for p in files if exclude not in p.name]
    return sorted(files, key=lambda p: (len(p.parts), p.name))[0] if files else None


def inspect(root: Path) -> dict:
    state_path = root / "job-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    final_dir = Path(state.get("final_dir", root))
    if not final_dir.exists():
        final_dir = root
    enhanced_md = first_file(final_dir, "*_enhanced.md")
    markdown_path = enhanced_md or first_file(final_dir, "*.md", "_enhanced")
    exact_json = final_dir / f"{final_dir.name}.json"
    json_candidates = [p for p in final_dir.rglob("*.json") if "_model" not in p.name and "manifest" not in p.name]
    json_path = exact_json if exact_json.exists() else (sorted(json_candidates, key=lambda p: len(p.parts))[0] if json_candidates else None)
    html_path = first_file(root / "exports", "*.html") if (root / "exports").exists() else None
    exports = sorted(str(p) for p in (root / "exports").glob("*") if p.is_file()) if (root / "exports").exists() else []

    pages: list = []
    if json_path:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            pages = raw
    labels: Counter[str] = Counter()
    formulas: list[dict] = []
    tables: list[dict] = []
    blocks: list[dict] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, list):
            continue
        for block in page:
            if not isinstance(block, dict):
                continue
            label = str(block.get("label") or block.get("native_label") or "unknown")
            content = str(block.get("content", ""))
            labels[label] += 1
            short = re.sub(r"\s+", " ", content).strip()
            blocks.append({"page": page_index + 1, "label": label, "content": short[:500], "bbox": block.get("bbox_2d")})
            if label in {"formula", "equation", "isolate_formula"} or re.search(r"\$\$|\\(?:frac|sum|begin|int|nabla)", content):
                formulas.append({"page": page_index + 1, "content": content})
            if label == "table" or "<table" in content.lower():
                tables.append({"page": page_index + 1, "content": content})

    pairs: list[dict] = []
    manifest = final_dir / "image-enhancement-manifest.json"
    if manifest.exists():
        try:
            for pair in json.loads(manifest.read_text(encoding="utf-8")):
                pairs.append({"original": pair.get("original", ""), "enhanced": pair.get("enhanced", ""), "method": pair.get("method", "")})
        except Exception:
            pass
    if not pairs:
        for image in final_dir.rglob("imgs/*"):
            if image.is_file():
                pairs.append({"original": str(image), "enhanced": "", "method": ""})

    return {
        "root": str(root), "final_dir": str(final_dir),
        "source": state.get("source", ""), "title": state.get("result_name", final_dir.name),
        "pages": state.get("total_pages", len(pages)), "status": state.get("status", "complete"),
        "markdown": str(markdown_path) if markdown_path else "",
        "json": str(json_path) if json_path else "",
        "html": str(html_path) if html_path else "",
        "exports": exports, "labels": dict(labels), "formulas": formulas,
        "tables": tables, "blocks": blocks, "images": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(inspect(Path(args.path).expanduser().resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
