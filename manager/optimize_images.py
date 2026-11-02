#!/usr/bin/env python3
"""Lossless, bounded per-image optimization before per-book final cleanup."""
from __future__ import annotations

import argparse
import json
import os
import re
import html
from urllib.parse import unquote
from pathlib import Path

from PIL import Image, features


def _identical(image: Image.Image, candidate: Path) -> bool:
    with Image.open(candidate) as restored:
        return (not getattr(restored, "is_animated", False)
                and restored.size == image.size
                and restored.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()
                and restored.info.get("icc_profile", b"") == image.info.get("icc_profile", b""))


def optimize_image(source: Path, *, allow_webp: bool = True) -> Path:
    """Return smaller verified pixels; retain source if changing its filename.

    Existing files are never overwritten by a WebP conversion. Atomic in-place
    PNG recompression is safe before link commit because pixels do not change.
    """
    if source.is_symlink() or source.suffix.lower() != ".png":
        return source
    initial_size = source.stat().st_size
    png_part = source.with_name(source.name + ".optimize-png.part")
    webp_part = source.with_name(source.name + ".optimize-webp.part")
    webp = source.with_name(source.name + ".lossless.webp")
    try:
        with Image.open(source) as image:
            if getattr(image, "is_animated", False) or image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                return source
            image.load()
            metadata = {key: image.info[key] for key in ("icc_profile", "exif", "dpi") if key in image.info}
            # Preserve palette transparency rather than flattening the image.
            if "transparency" in image.info:
                metadata["transparency"] = image.info["transparency"]
            image.save(png_part, format="PNG", optimize=True, compress_level=9, **metadata)
            best = png_part if png_part.stat().st_size < initial_size and _identical(image, png_part) else source
            if allow_webp and features.check("webp"):
                if webp.exists():
                    candidate = webp if _identical(image, webp) else None
                else:
                    options = {key: image.info[key] for key in ("icc_profile", "exif") if key in image.info}
                    try:
                        image.save(webp_part, format="WEBP", lossless=True, exact=True, quality=100, method=6, **options)
                        candidate = webp_part if _identical(image, webp_part) else None
                    except (OSError, ValueError):
                        candidate = None  # e.g. dimensions outside the WebP limit
                if candidate is not None and candidate.stat().st_size < best.stat().st_size:
                    best = candidate
            if best == source or initial_size - best.stat().st_size < max(128, initial_size // 100):
                return source
            if best == png_part:
                os.replace(png_part, source)
                return source
            if best == webp_part:
                os.replace(webp_part, webp)
            return webp
    finally:
        png_part.unlink(missing_ok=True)
        webp_part.unlink(missing_ok=True)


def _emit(kind: str, **payload) -> None:
    print("GLMOCR_EVENT " + json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def _atomic_text(path: Path, content: str) -> None:
    part = path.with_name(path.name + ".optimize.part")
    try:
        part.write_text(content, encoding="utf-8")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def optimize_book_images(result: Path, output_root: Path) -> dict[str, int]:
    from export_results import rebase_markdown_assets
    result = result.resolve()
    output_root = output_root.resolve()
    if result == output_root or not result.is_relative_to(output_root):
        raise ValueError("Image result must be a book directory inside output-root")
    files = []
    for name in ("imgs", "imgs_enhanced"):
        directory = result / name
        if directory.is_dir() and not directory.is_symlink():
            files.extend(p for p in directory.rglob("*")
                         if p.is_file() and not p.is_symlink()
                         and p.resolve().is_relative_to(result)
                         and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
    files.sort()
    before = sum(p.stat().st_size for p in files)
    mapping = {}
    warnings = 0
    _emit("optimization_started", total=len(files))
    for index, source in enumerate(files, 1):
        try:
            optimized = optimize_image(source, allow_webp="imgs_enhanced" in source.relative_to(result).parts)
            if optimized != source:
                mapping[source.resolve()] = optimized.resolve()
        except Exception as exc:
            warnings += 1
            _emit("log", level="warning", message=f"Изображение сохранено без оптимизации: {source.name}: {exc}")
        _emit("optimization_progress", done=index, total=len(files))
    # New images are committed before links, all links before deletion. If a
    # write fails or the process is terminated, both old and new paths work.
    markdowns = list(result.rglob("*.md"))
    exports = output_root / "exports"
    if exports.is_dir() and not exports.is_symlink():
        markdowns.extend(exports.rglob("*.md"))
    contents = {}
    for path in sorted(set(markdowns)):
        if path.is_symlink() or not path.resolve().is_relative_to(output_root):
            continue
        contents[path] = path.read_text(encoding="utf-8")
    planned = {p: rebase_markdown_assets(text, p.parent, p.parent, asset_map=mapping)
               for p, text in contents.items()}
    def mentioned(name: str, texts) -> bool:
        pattern = re.compile(re.escape(name) + r"(?![\w.-])")
        return any(pattern.search(re.sub(r"\\(.)", r"\1", unquote(html.unescape(text)))) for text in texts)
    # Reference-style links, srcset and literals are deliberately not rewritten.
    # If any old filename remains, keep that filename and recompress its PNG
    # instead of creating a broken reference or retaining duplicate images.
    for source, target in list(mapping.items()):
        if mentioned(source.name, planned.values()):
            del mapping[source]
            optimize_image(source, allow_webp=False)
            if target not in files and not mentioned(target.name, contents.values()):
                target.unlink(missing_ok=True)
    for path, content in contents.items():
        changed = rebase_markdown_assets(content, path.parent, path.parent, asset_map=mapping)
        if changed != content:
            _atomic_text(path, changed)
    for source in mapping:
        source.unlink(missing_ok=True)
    after_files = {mapping.get(p.resolve(), p.resolve()) for p in files}
    after = sum(p.stat().st_size for p in after_files if p.exists())
    stats = {"images": len(files), "bytes_before": before, "bytes_after": after,
             "saved_bytes": max(0, before - after), "warnings": warnings}
    _emit("optimization_done", **stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    try:
        optimize_book_images(Path(args.result), Path(args.output_root))
    except Exception as exc:
        _emit("error", message=f"Оптимизация не завершена, исходные изображения сохранены: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
