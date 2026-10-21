#!/usr/bin/env python3
"""Последовательная очередь документов. Второй экземпляр не стартует.

Это помощник рядом с приложением, не часть окна. Питание и свободное место
проверяются перед каждым документом. Удаляется только временная копия уже
завершённого входа, не исходный файл и не активный результат.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable

from queue_inputs import prepare_input

LOCK_NAME = "night-queue.lock"
STATE_NAME = "night-queue-state.json"


def disk_free(path: Path) -> int:
    return shutil.disk_usage(path).free


def power_plugged() -> bool:
    """На Mac смотрит pmset. Вне Mac очередь не останавливается из-за батареи."""
    try:
        import subprocess
        result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return True
    text = (result.stdout or "") + (result.stderr or "")
    if "AC Power" in text or "charged" in text and "Battery Power" not in text:
        return True
    return "Battery Power" not in text


def should_pause(plugged: bool, free_bytes: int, stop_below: int) -> str | None:
    if not plugged:
        return "Нет питания от сети"
    if free_bytes < stop_below:
        return f"Свободно {free_bytes} байт, порог {stop_below}"
    return None


def acquire_lock(lock: Path) -> int:
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    return fd


def release_lock(fd: int, lock: Path) -> None:
    try:
        os.close(fd)
    finally:
        lock.unlink(missing_ok=True)


def _state_path(work: Path) -> Path:
    return work / STATE_NAME


def load_state(work: Path) -> dict:
    path = _state_path(work)
    if not path.is_file():
        return {"done": [], "paused": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(work: Path, state: dict) -> None:
    path = _state_path(work)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def output_is_complete(output: Path) -> bool:
    state = output / "job-state.json"
    if not state.is_file():
        return False
    try:
        payload = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete"


def run_queue(
    inputs: list[Path],
    output_root: Path,
    work: Path,
    runner: Callable[[Path, Path], int],
    *,
    stop_below: int = 15 * 1024 ** 3,
    plugged: Callable[[], bool] = power_plugged,
    free: Callable[[Path], int] = disk_free,
) -> int:
    work.mkdir(parents=True, exist_ok=True)
    lock = work / LOCK_NAME
    try:
        fd = acquire_lock(lock)
    except FileExistsError:
        print("Очередь уже запущена", flush=True)
        return 2
    state = load_state(work)
    done = set(state.get("done") or [])
    try:
        for source in inputs:
            reason = should_pause(plugged(), free(output_root), stop_below)
            if reason:
                state["paused"] = reason
                save_state(work, state)
                print(f"Пауза: {reason}", flush=True)
                return 75
            prepared = prepare_input(source, work / "inputs")
            key = str(source.resolve())
            output = output_root / source.stem
            if key in done or output_is_complete(output):
                if key not in done:
                    done.add(key)
                    state["done"] = sorted(done)
                    save_state(work, state)
                continue
            if output.exists() and not output_is_complete(output):
                # Незавершённый результат не удаляем: обработчик продолжит его сам.
                pass
            code = runner(prepared if prepared.is_dir() else prepared, output)
            if code != 0:
                state["paused"] = f"Документ {source.name} остановился с кодом {code}"
                state["done"] = sorted(done)
                save_state(work, state)
                return code
            done.add(key)
            state["done"] = sorted(done)
            state["paused"] = None
            save_state(work, state)
            copied = work / "inputs" / source.stem
            if source.suffix.lower() == ".zip" and copied.is_dir() and copied.resolve() != source.resolve():
                shutil.rmtree(copied)
        return 0
    finally:
        release_lock(fd, lock)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--stop-below-gb", type=int, default=15)
    args = parser.parse_args()

    def runner(source: Path, output: Path) -> int:
        import subprocess
        root = Path(__file__).resolve().parent.parent
        python = root / ".venv-sdk" / "bin" / "python"
        if not python.is_file():
            python = Path(os.environ.get("PYTHON", "python3"))
        return subprocess.call([
            str(python), str(root / "manager" / "job_runner.py"),
            "--input", str(source), "--output", str(output),
            "--enhance", "hybrid", "--exports",
        ])

    return run_queue(
        [Path(item) for item in args.inputs],
        Path(args.output),
        Path(args.work),
        runner,
        stop_below=args.stop_below_gb * 1024 ** 3,
    )


if __name__ == "__main__":
    raise SystemExit(main())
