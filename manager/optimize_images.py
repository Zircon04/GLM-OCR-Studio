#!/usr/bin/env python3
"""Перепаковка картинок без смены пикселей после экспорта одного документа.

Исходные PNG остаются PNG. У улучшенных PNG пробуется меньший lossless WebP.
Новый файл пишется рядом, Markdown обновляется атомарно, и только потом
удаляется заменённый PNG. Если ссылка на старое имя где-то осталась, имя
не меняется: PNG только перепаковывается.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

Progress = Callable[[int, int, str], None]
_FENCE = re.compile(r"(```.*?```|\$\$.*?\$\$|`[^`\n]*`)", re.DOTALL)
_MD_TARGET = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")
_HTML_SRC = re.compile(r"(<img\b[^>]*\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)


def _emit(progress: Progress | None, done: int, total: int, name: str) -> None:
    if progress is not None:
        progress(done, total, name)


def _icc(image: Image.Image) -> bytes | None:
    profile = image.info.get("icc_profile")
    return profile if isinstance(profile, bytes) else None


def _rgba(image: Image.Image) -> bytes:
    return image.convert("RGBA").tobytes()


def _animated(image: Image.Image) -> bool:
    return bool(getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) > 1)


def _supported(image: Image.Image) -> bool:
    if _animated(image):
        return False
    return image.mode in {"1", "L", "LA", "P", "RGB", "RGBA"}


def _same_pixels(left: Image.Image, right: Image.Image) -> bool:
    return left.size == right.size and _rgba(left) == _rgba(right) and _icc(left) == _icc(right)


def _save_png(image: Image.Image, dest: Path) -> None:
    extra: dict[str, Any] = {}
    profile = _icc(image)
    if profile:
        extra["icc_profile"] = profile
    image.save(dest, format="PNG", optimize=True, **extra)


def _save_webp(image: Image.Image, dest: Path) -> None:
    extra: dict[str, Any] = {"lossless": True, "quality": 100, "method": 6}
    profile = _icc(image)
    if profile:
        extra["icc_profile"] = profile
    image.save(dest, format="WEBP", **extra)


def _repack_png(path: Path) -> int:
    """Меньший PNG с теми же пикселями. Возвращает сэкономленные байты."""
    if Image is None or path.suffix.lower() != ".png":
        return 0
    before = path.stat().st_size
    try:
        with Image.open(path) as image:
            image.load()
            if not _supported(image):
                return 0
            snapshot = image.copy()
        temp = path.with_name(path.name + ".repack.png")
        _save_png(snapshot, temp)
        with Image.open(temp) as packed, Image.open(path) as original:
            original.load()
            packed.load()
            if temp.stat().st_size >= before or not _same_pixels(original, packed):
                temp.unlink(missing_ok=True)
                return 0
        temp.replace(path)
        return before - path.stat().st_size
    except Exception:
        return 0


def _protected_sub(text: str, pattern: re.Pattern[str], replacer) -> str:
    pieces = _FENCE.split(text)
    out: list[str] = []
    for index, piece in enumerate(pieces):
        if index % 2 == 1:
            out.append(piece)
        else:
            out.append(pattern.sub(replacer, piece))
    return "".join(out)


def rewrite_names(text: str, replacements: dict[str, str]) -> str:
    """Меняет только пути картинок. Формулы, код и ограждения не трогает."""

    def swap(raw: str) -> str:
        query = ""
        cleaned = raw.replace("\\", "/")
        if "?" in cleaned:
            cleaned, query = cleaned.split("?", 1)
            query = "?" + query
        for old, new in replacements.items():
            if old in cleaned:
                return cleaned.replace(old, new, 1) + query
        return raw

    def md(match: re.Match[str]) -> str:
        return match.group(1) + swap(match.group(2)) + match.group(3)

    def html(match: re.Match[str]) -> str:
        return match.group(1) + swap(match.group(2)) + match.group(3)

    return _protected_sub(_protected_sub(text, _MD_TARGET, md), _HTML_SRC, html)


def _markdown_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*.md"):
        if not path.is_file() or ".job" in path.parts:
            continue
        found.append(path)
    return found


def _still_mentioned(root: Path, needle: str) -> bool:
    for path in _markdown_files(root):
        if needle in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def _atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".opt.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def optimize_tree(root: Path, progress: Progress | None = None) -> dict[str, int]:
    """Оптимизирует картинки внутри результата. Корень включает и exports."""
    root = Path(root)
    if Image is None or not root.is_dir():
        return {"saved_bytes": 0, "converted": 0, "repacked": 0}
    images = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png" and ".job" not in path.parts
        and any(part in {"imgs", "imgs_enhanced"} for part in path.parts)
    ]
    total = len(images)
    saved = 0
    converted = 0
    repacked = 0
    planned: list[tuple[Path, Path, str, str]] = []
    for index, path in enumerate(images, 1):
        _emit(progress, index, total, path.name)
        enhanced = "imgs_enhanced" in path.parts
        if not enhanced:
            gained = _repack_png(path)
            saved += gained
            if gained:
                repacked += 1
            continue
        webp = path.with_suffix(".webp")
        if webp.exists() and webp.stat().st_size > 0:
            # Чужой отличающийся WebP не перезаписываем.
            try:
                with Image.open(path) as src, Image.open(webp) as dst:
                    src.load()
                    dst.load()
                    if not _same_pixels(src, dst):
                        gained = _repack_png(path)
                        saved += gained
                        if gained:
                            repacked += 1
                        continue
            except Exception:
                continue
        try:
            with Image.open(path) as image:
                image.load()
                if not _supported(image):
                    continue
                snapshot = image.copy()
            _save_webp(snapshot, webp)
            with Image.open(path) as src, Image.open(webp) as dst:
                src.load()
                dst.load()
                if webp.stat().st_size >= path.stat().st_size or not _same_pixels(src, dst):
                    webp.unlink(missing_ok=True)
                    gained = _repack_png(path)
                    saved += gained
                    if gained:
                        repacked += 1
                    continue
        except Exception:
            webp.unlink(missing_ok=True)
            continue
        old = path.relative_to(root).as_posix()
        new = webp.relative_to(root).as_posix()
        planned.append((path, webp, old, new))

    # Ссылки обновляются после того, как новые файлы уже лежат на диске.
    replacements: dict[str, str] = {}
    for _src, _dst, old, new in planned:
        old_name = "imgs_enhanced/" + Path(old).name
        new_name = "imgs_enhanced/" + Path(new).name
        replacements[old_name] = new_name
        replacements[old] = new
    if replacements:
        for markdown in _markdown_files(root):
            original = markdown.read_text(encoding="utf-8")
            changed = rewrite_names(original, replacements)
            if changed != original:
                _atomic_write(markdown, changed)

    for src, dst, old, new in planned:
        short_old = "imgs_enhanced/" + src.name
        if _still_mentioned(root, short_old) or _still_mentioned(root, old):
            dst.unlink(missing_ok=True)
            gained = _repack_png(src)
            saved += gained
            if gained:
                repacked += 1
            continue
        before = src.stat().st_size
        src.unlink(missing_ok=True)
        saved += max(0, before - dst.stat().st_size)
        converted += 1
    return {"saved_bytes": saved, "converted": converted, "repacked": repacked}
