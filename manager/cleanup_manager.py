#!/usr/bin/env python3
"""Narrow, guarded cleanup helpers for GLM OCR Local.

This tool never touches source documents, model weights, virtual environments,
or the application bundle. Whole-result deletion is intentionally handled by
the native app through the macOS Trash.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

INSTALL_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = (INSTALL_DIR / "results").resolve()
LOG_FILE = (INSTALL_DIR / "logs/mlx-vlm.log").resolve()


def allocated_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_blocks * 512
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in dirs + files:
            item = root_path / name
            try:
                total += item.lstat().st_blocks * 512
            except OSError:
                pass
    return total


def guarded_result(raw: str) -> Path:
    requested = Path(raw).expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("Символические ссылки не очищаются")
    path = requested.resolve(strict=True)
    if path == RESULTS_DIR or path.parent != RESULTS_DIR:
        raise ValueError("Разрешена очистка только отдельной папки задания внутри results")
    if not path.is_dir():
        raise ValueError("Папка результата не найдена")
    return path


def state_for(path: Path) -> dict:
    state_path = path / "job-state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def support_targets(path: Path) -> list[Path]:
    targets: list[Path] = []
    job = path / ".job"
    if job.is_dir() and not job.is_symlink():
        targets.append(job)
    for candidate in path.rglob("layout_vis"):
        if not candidate.is_dir() or candidate.is_symlink() or ".job" in candidate.parts:
            continue
        targets.append(candidate)
    for candidate in path.rglob(".DS_Store"):
        if candidate.is_file() and ".job" not in candidate.parts:
            targets.append(candidate)
    # Ensure nested matches are not counted or removed twice.
    unique: list[Path] = []
    for candidate in sorted(set(targets), key=lambda value: len(value.parts)):
        if not any(parent == candidate or parent in candidate.parents for parent in unique):
            unique.append(candidate)
    return unique


def inspect(path: Path) -> dict:
    targets = support_targets(path)
    state = state_for(path)
    return {
        "path": str(path),
        "status": state.get("status", "unknown"),
        "total_bytes": allocated_size(path),
        "support_bytes": sum(allocated_size(item) for item in targets),
        "support_targets": [str(item) for item in targets],
    }


def trim(path: Path) -> dict:
    state = state_for(path)
    if state.get("status") != "complete":
        raise ValueError("Служебные файлы можно удалить только у завершённого задания")
    targets = support_targets(path)
    reclaimed = sum(allocated_size(item) for item in targets)
    removed: list[str] = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        removed.append(str(target))
    state["support_files_removed"] = True
    state_path = path / "job-state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "reclaimed_bytes": reclaimed, "removed": removed}


def clear_logs() -> dict:
    if LOG_FILE.exists() and not LOG_FILE.is_symlink():
        reclaimed = allocated_size(LOG_FILE)
        LOG_FILE.write_bytes(b"")
    else:
        reclaimed = 0
    return {"path": str(LOG_FILE), "reclaimed_bytes": reclaimed}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "trim"):
        command = sub.add_parser(name)
        command.add_argument("path")
    sub.add_parser("clear-logs")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            result = inspect(guarded_result(args.path))
        elif args.command == "trim":
            result = trim(guarded_result(args.path))
        else:
            result = clear_logs()
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
