#!/usr/bin/env python3
"""Safely enhance images extracted by GLM-OCR without modifying originals."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image
from realesrgan_mlx.pipeline_mlx import make_upsampler

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def classical_restore(image: np.ndarray) -> np.ndarray:
    """Mild denoise + local contrast + unsharp mask; deliberately conservative."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3]
    else:
        rgb, alpha = image[:, :, :3], None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(l_chan)
    contrast = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.35, blurred, -0.35, 0)
    out = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    if alpha is not None:
        out = np.dstack((out, alpha))
    return out


def find_images(result_dir: Path) -> list[Path]:
    found: list[Path] = []
    for imgs_dir in result_dir.rglob("imgs"):
        if not imgs_dir.is_dir() or "imgs_enhanced" in imgs_dir.parts or ".job" in imgs_dir.parts:
            continue
        found.extend(p for p in imgs_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--mode", choices=("safe", "hybrid"), default="hybrid")
    parser.add_argument("--small-edge", type=int, default=900)
    parser.add_argument("--max-pixels", type=int, default=1_200_000)
    parser.add_argument("--tile", type=int, default=256)
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.is_dir():
        print(f"ERROR: папка результата не найдена: {result_dir}", flush=True)
        return 2

    images = find_images(result_dir)
    print(f"ENHANCE_TOTAL {len(images)}", flush=True)
    if not images:
        print("ENHANCE_DONE 0 0", flush=True)
        return 0

    upsampler = None
    if args.mode == "hybrid":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        upsampler = make_upsampler("RealESRGAN_x2plus", tile=args.tile)

    enhanced_count = 0
    upscaled_count = 0
    replacements: dict[str, str] = {}
    manifest: list[dict[str, object]] = []

    for index, src in enumerate(images, 1):
        imgs_dir = next(parent for parent in src.parents if parent.name == "imgs")
        relative = src.relative_to(imgs_dir)
        dst_dir = imgs_dir.parent / "imgs_enhanced" / relative.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = (dst_dir / relative.name).with_suffix(".png")
        method = "noise-reduction + sharpening"
        try:
            with Image.open(src) as pil:
                original = np.asarray(pil.convert("RGB"))
            restored = classical_restore(original)
            height, width = restored.shape[:2]
            should_upscale = (
                upsampler is not None
                and min(width, height) < args.small_edge
                and width * height <= args.max_pixels
            )
            if should_upscale:
                temp = dst_dir / (dst.stem + ".pre_esrgan.png")
                Image.fromarray(restored).save(temp)
                output, _ = upsampler.enhance(np.asarray(Image.open(temp)), outscale=2)
                temp.unlink(missing_ok=True)
                restored = output
                method += " + Real-ESRGAN x2 (MLX)"
                upscaled_count += 1
            Image.fromarray(restored).save(dst, optimize=True)
            enhanced_count += 1
            replacements[f"imgs/{relative.as_posix()}"] = f"imgs_enhanced/{relative.with_suffix('.png').as_posix()}"
            manifest.append({
                "original": str(src), "enhanced": str(dst), "method": method,
                "original_size": [width, height],
                "enhanced_size": [int(restored.shape[1]), int(restored.shape[0])],
            })
            print(f"ENHANCE_PROGRESS {index} {len(images)} {src.name} {method}", flush=True)
        except Exception as exc:
            print(f"ENHANCE_WARNING {src}: {exc}", flush=True)

    for markdown in result_dir.rglob("*.md"):
        if markdown.name.endswith("_enhanced.md"):
            continue
        content = markdown.read_text(encoding="utf-8")
        changed = content
        for old, new in replacements.items():
            changed = changed.replace(old, new)
        if changed != content:
            enhanced_markdown = markdown.with_name(markdown.stem + "_enhanced.md")
            enhanced_markdown.write_text(changed, encoding="utf-8")

    manifest_path = result_dir / "image-enhancement-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ENHANCE_DONE {enhanced_count} {upscaled_count}", flush=True)
    print(f"Metal device: {mx.default_device()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
