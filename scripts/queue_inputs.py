#!/usr/bin/env python3
"""Входы ночной очереди: PDF, папка и ZIP, с проверкой уже распакованного."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

DOCUMENT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
MANIFEST_NAME = "inputs-ready.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _documents(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES and path.name != MANIFEST_NAME
    )


def write_manifest(unpacked: Path, source: Path | None) -> Path:
    files = []
    for path in _documents(unpacked):
        files.append({"path": path.relative_to(unpacked).as_posix(), "sha256": sha256(path), "size": path.stat().st_size})
    payload = {"source": str(source) if source else "", "files": files}
    target = unpacked / MANIFEST_NAME
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return target


def manifest_matches(unpacked: Path) -> bool:
    target = unpacked / MANIFEST_NAME
    if not target.is_file():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    listed = {item["path"]: item for item in payload.get("files", []) if isinstance(item, dict)}
    actual = {path.relative_to(unpacked).as_posix(): path for path in _documents(unpacked)}
    if set(listed) != set(actual):
        return False
    for relative, path in actual.items():
        item = listed[relative]
        if int(item.get("size", -1)) != path.stat().st_size:
            return False
        if item.get("sha256") != sha256(path):
            return False
    return True


def prepare_input(source: Path, work: Path) -> Path:
    """Возвращает папку с документами. ZIP распаковывается, уже проверенная копия переиспользуется."""
    source = source.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if manifest_matches(source):
            return source
        if any(_documents(source)):
            write_manifest(source, source)
            return source
        raise FileNotFoundError(f"В папке нет документов: {source}")
    if source.suffix.lower() == ".zip":
        unpacked = work / source.stem
        if unpacked.is_dir() and manifest_matches(unpacked):
            return unpacked
        if unpacked.exists():
            raise FileExistsError(f"Папка распаковки уже есть и не совпадает с журналом: {unpacked}")
        unpacked.mkdir(parents=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(unpacked)
        if not _documents(unpacked):
            raise FileNotFoundError(f"В архиве нет документов: {source}")
        write_manifest(unpacked, source)
        return unpacked
    if source.suffix.lower() in DOCUMENT_SUFFIXES:
        return source
    raise ValueError(f"Неизвестный вход: {source}")
