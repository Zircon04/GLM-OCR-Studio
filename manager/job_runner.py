#!/usr/bin/env python3
"""Resumable local GLM-OCR job runner for the macOS GUI."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf

INSTALL_DIR = Path(__file__).resolve().parent.parent
GLMOCR = INSTALL_DIR / ".venv-sdk/bin/glmocr"
CONFIG = INSTALL_DIR / "config/local-mlx.yaml"
ENHANCE_PYTHON = INSTALL_DIR / ".venv-enhance/bin/python"
ENHANCE_SCRIPT = INSTALL_DIR / "enhancement/enhance_results.py"
EXPORT_SCRIPT = INSTALL_DIR / "manager/export_results.py"
LIBRARY_SCRIPT = INSTALL_DIR / "manager/library_index.py"
SERVER_LOG = INSTALL_DIR / "logs/mlx-vlm.log"
PROFILE_FILE = INSTALL_DIR / "config/performance-profile.env"
EVENT_PREFIX = "GLMOCR_EVENT "

cancelled = False
active_child: subprocess.Popen[bytes] | None = None


def emit(kind: str, **payload: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def safe_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).rstrip(" .")
    return value or "result"


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_profile() -> dict[str, int | str]:
    values: dict[str, str] = {}
    if PROFILE_FILE.exists():
        for line in PROFILE_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return {
        "name": values.get("PROFILE", "balanced"),
        "workers": int(values.get("SDK_WORKERS", "6")),
        "pages": int(values.get("PAGE_QUEUE", "8")),
        "regions": int(values.get("REGION_QUEUE", "160")),
        "memory": int(values.get("TOTAL_MEMORY_GB", "16")),
        "reserved": int(values.get("RESERVED_MEMORY_GB", "5")),
    }


def chunk_size_for(profile: dict[str, int | str]) -> int:
    name, memory = str(profile["name"]), int(profile["memory"])
    base = {"low": 5, "auto": 8, "balanced": 10, "high": 12, "max": 14, "extreme": 18, "fast": 12}.get(name, 10)
    if memory <= 8:
        return min(base, 5)
    if memory <= 16:
        return base
    if memory <= 24:
        return int(base * 1.5)
    return min(32, base * 2)


def memory_free_percent() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, timeout=4
        )
        match = re.search(r"free percentage:\s*(\d+)%", result.stdout + result.stderr)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def wait_for_memory() -> None:
    profile = read_profile()
    reserve = int(profile.get("reserved", 5))
    total = max(1, int(profile.get("memory", 16)))
    minimum_free = max(10, int(reserve / total * 100))
    while not cancelled:
        free = memory_free_percent()
        if free is None or free >= minimum_free:
            return
        emit("memory_wait", free_percent=free, minimum_percent=minimum_free,
             message="Память занята; следующий блок подождёт, чтобы macOS оставалась отзывчивой")
        time.sleep(5)


def local_server_alive() -> bool:
    pid_file = INSTALL_DIR / "glm-ocr-server.pid"
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        process = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=3
        )
        if process.returncode == 0 and str(INSTALL_DIR / ".venv-mlx/bin/mlx_vlm.server") in process.stdout:
            return True
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["/usr/bin/curl", "--silent", "--fail", "--connect-timeout", "1", "--max-time", "3",
             "http://127.0.0.1:8080/health"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def handle_signal(_signum: int, _frame: Any) -> None:
    global cancelled
    cancelled = True
    if active_child and active_child.poll() is None:
        active_child.send_signal(signal.SIGINT)


def server_log_offset() -> int:
    try:
        return SERVER_LOG.stat().st_size
    except OSError:
        return 0


def count_new_fragments(offset: int) -> tuple[int, int]:
    try:
        with SERVER_LOG.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
            return data.count(b"Request completed: endpoint=/chat/completions"), offset + len(data)
    except OSError:
        return 0, offset


def run_streaming(command: list[str], stage: str, state: dict[str, Any], state_path: Path) -> int:
    global active_child
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    active_child = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, cwd=INSTALL_DIR
    )
    assert active_child.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(active_child.stdout, selectors.EVENT_READ)
    pending = ""
    last_heartbeat = 0.0
    fragments = 0
    log_offset = server_log_offset()
    queue_pages = "0/0"
    queue_regions = "0/0"
    units_done = 0
    units_total = 0
    segment_started_at = time.time()
    processing_base = float(state.get("processing_seconds", 0.0))
    while active_child.poll() is None or selector.get_map():
        for key, _ in selector.select(timeout=0.5):
            data = os.read(key.fileobj.fileno(), 65536)
            if not data:
                selector.unregister(key.fileobj)
                continue
            pending += data.decode("utf-8", errors="replace").replace("\r", "\n")
            parts = pending.split("\n")
            pending = parts.pop()
            for line in parts:
                if line.startswith(EVENT_PREFIX):
                    print(line, flush=True)
                    continue
                if line.startswith("ENHANCE_TOTAL "):
                    emit("enhancement_total", total=int(line.split()[1]))
                    continue
                if line.startswith("ENHANCE_PROGRESS "):
                    values = line.split(" ", 4)
                    emit("enhancement_progress", done=int(values[1]), total=int(values[2]), image=values[3])
                    continue
                match = re.search(r"Q1:(\d+/\d+) Q2:(\d+/\d+)", line)
                if match:
                    queue_pages, queue_regions = match.group(1), match.group(2)
                progress_match = re.search(r"Parsing:\s*\d+%.*?\b(\d+)/(\d+)\s*\[", line)
                if progress_match:
                    units_done, units_total = int(progress_match.group(1)), int(progress_match.group(2))
                elif any(word in line for word in ("Error", "ERROR", "Failed", "WARNING")):
                    emit("log", level="warning", message=line.strip()[-1000:])
        now = time.time()
        if now - last_heartbeat >= 1:
            state["active_elapsed"] = processing_base + now - segment_started_at
            increment, log_offset = count_new_fragments(log_offset)
            fragments += increment
            state["fragments"] = int(state.get("fragments", 0)) + increment
            state["updated_at"] = now
            atomic_json(state_path, state)
            payload = progress_payload(state, units_done if stage == "ocr" else 0)
            emit(
                "heartbeat", stage=stage, fragments=fragments,
                total_fragments=state["fragments"], page_queue=queue_pages,
                region_queue=queue_regions, elapsed=int(state.get("active_elapsed", now - state["started_at"])),
                memory_free_percent=memory_free_percent(), units_done=units_done,
                units_total=units_total, **payload,
            )
            last_heartbeat = now
        if cancelled and active_child.poll() is None:
            active_child.send_signal(signal.SIGINT)
    code = active_child.wait()
    active_child = None
    return code


def split_pdf(source: Path, chunks_dir: Path, chunk_size: int) -> tuple[int, list[dict[str, Any]]]:
    doc = pymupdf.open(source)
    total = len(doc)
    chunks: list[dict[str, Any]] = []
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for index, start in enumerate(range(0, total, chunk_size)):
        end = min(total, start + chunk_size)
        chunk_id = f"chunk_{index:04d}"
        pdf_path = chunks_dir / f"{chunk_id}.pdf"
        if not pdf_path.exists():
            part = pymupdf.open()
            part.insert_pdf(doc, from_page=start, to_page=end - 1)
            part.save(pdf_path, garbage=3, deflate=True)
            part.close()
        chunks.append({"id": chunk_id, "start": start, "end": end, "input": str(pdf_path), "status": "pending"})
    doc.close()
    return total, chunks


def find_artifact(directory: Path, suffix: str, exclude: str = "") -> Path | None:
    matches = [p for p in directory.rglob(f"*{suffix}") if not exclude or exclude not in p.name]
    return sorted(matches, key=lambda p: len(p.parts))[0] if matches else None


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    return value


def image_job_stamp(state: dict[str, Any], output_root: Path) -> str:
    """Stable per-job prefix used by image files and Markdown links."""
    started = float(state.get("started_at", 0) or 0)
    date = datetime.fromtimestamp(started).strftime("%Y%m%d") if started else datetime.now().strftime("%Y%m%d")
    identity = f"{state.get('source', '')}|{output_root.name}|{started:.6f}"
    return f"{date}-{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:8]}"


def merge_chunks(state: dict[str, Any], output_root: Path) -> Path:
    final_dir = output_root / state["result_name"]
    final_dir.mkdir(parents=True, exist_ok=True)
    final_imgs = final_dir / "imgs"
    final_layout = final_dir / "layout_vis"
    merged_json: list[Any] = []
    merged_model: list[Any] = []
    markdown_parts: list[str] = []
    job_stamp = image_job_stamp(state, output_root)

    for chunk_index, chunk in enumerate(state["chunks"]):
        if chunk["status"] != "done":
            continue
        chunk_output = Path(chunk["output"])
        json_path = find_artifact(chunk_output, ".json", "_model")
        model_path = find_artifact(chunk_output, "_model.json")
        md_path = find_artifact(chunk_output, ".md", "_enhanced")
        replacements: dict[str, str] = {}
        imgs_dir = next((p for p in chunk_output.rglob("imgs") if p.is_dir()), None)
        if imgs_dir:
            final_imgs.mkdir(parents=True, exist_ok=True)
            for image in imgs_dir.rglob("*"):
                if not image.is_file():
                    continue
                match = re.search(r"cropped_page(\d+)_idx(\d+)", image.stem)
                local_page = int(match.group(1)) if match else 0
                local_idx = int(match.group(2)) if match else 0
                global_page = chunk["start"] + local_page
                file_hash = hashlib.sha1(image.read_bytes()).hexdigest()[:8]
                new_name = f"figure_{job_stamp}_p{global_page + 1:04d}_b{chunk_index + 1:03d}_i{local_idx + 1:04d}_{file_hash}{image.suffix.lower()}"
                shutil.copy2(image, final_imgs / new_name)
                for old in (f"imgs/{image.name}", f"./imgs/{image.name}", str(image)):
                    replacements[old] = f"imgs/{new_name}"
        layout_dir = next((p for p in chunk_output.rglob("layout_vis") if p.is_dir()), None)
        if layout_dir:
            final_layout.mkdir(parents=True, exist_ok=True)
            images = sorted(p for p in layout_dir.iterdir() if p.is_file())
            for local_page, image in enumerate(images):
                shutil.copy2(image, final_layout / f"layout_page{chunk['start'] + local_page:04d}{image.suffix.lower()}")
        if json_path:
            data = replace_strings(json.loads(json_path.read_text(encoding="utf-8")), replacements)
            if isinstance(data, list):
                merged_json.extend(data)
        if model_path:
            data = replace_strings(json.loads(model_path.read_text(encoding="utf-8")), replacements)
            if isinstance(data, list):
                merged_model.extend(data)
        if md_path:
            text = md_path.read_text(encoding="utf-8", errors="replace")
            for old, new in replacements.items():
                text = text.replace(old, new)
            markdown_parts.append(f"<!-- Страницы {chunk['start'] + 1}-{chunk['end']} -->\n\n{text.strip()}")

    base = state["result_name"]
    (final_dir / f"{base}.json").write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")
    if merged_model:
        (final_dir / f"{base}_model.json").write_text(json.dumps(merged_model, ensure_ascii=False, indent=2), encoding="utf-8")
    (final_dir / f"{base}.md").write_text("\n\n---\n\n".join(markdown_parts), encoding="utf-8")
    return final_dir


def copy_direct_result(state: dict[str, Any], output_root: Path) -> Path:
    final_dir = output_root / state["result_name"]
    return final_dir


def progress_payload(state: dict[str, Any], active_pages: int = 0) -> dict[str, Any]:
    done_pages = sum(chunk["end"] - chunk["start"] for chunk in state["chunks"] if chunk["status"] == "done")
    if active_pages and isinstance(state.get("current_chunk"), int):
        chunk = state["chunks"][state["current_chunk"]]
        done_pages += min(active_pages, chunk["end"] - chunk["start"])
    total = state["total_pages"]
    elapsed = max(1, int(state.get("active_elapsed", time.time() - state["started_at"])))
    eta = int(elapsed / done_pages * (total - done_pages)) if done_pages else None
    return {"pages_done": done_pages, "pages_total": total, "fraction": done_pages / max(1, total), "eta_seconds": eta}


def run_job(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Файл или папка не найдены: {source}")
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "job-state.json"
    profile = read_profile()
    result_name = safe_name(source.stem if source.is_file() else source.name)
    export_formats = args.export_formats if args.exports else ""

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if Path(state.get("source", "")).resolve() != source:
            raise ValueError("Папка задания относится к другому исходному файлу")
        if state.get("status") == "complete":
            final_dir = Path(state.get("final_dir", output_root / state.get("result_name", source.stem)))
            if not final_dir.exists():
                raise FileNotFoundError("Готовый результат был удалён")
            emit("already_complete", message="Задание уже завершено", final_dir=str(final_dir), output=str(output_root))
            emit("complete", final_dir=str(final_dir), output=str(output_root), **progress_payload(state))
            return 0
        emit("resumed", message="Продолжаю ранее начатое задание", state_path=str(state_path))
        # A resumed job follows the options currently selected in the GUI.
        state["enhancement"] = args.enhance
        state["exports"] = args.exports
        state["export_formats"] = export_formats
    else:
        state = {
            "version": 1, "source": str(source), "output": str(output_root),
            "result_name": result_name, "status": "running", "stage": "ocr",
            "started_at": time.time(), "updated_at": time.time(), "fragments": 0,
            "processing_seconds": 0.0, "active_elapsed": 0.0,
            "profile": profile, "enhancement": args.enhance, "exports": args.exports,
            "export_formats": export_formats,
        }
        if source.suffix.lower() == ".pdf":
            selected_chunk_size = args.chunk_size or chunk_size_for(profile)
            total, chunks = split_pdf(source, output_root / ".job/chunks", selected_chunk_size)
            state.update({"total_pages": total, "chunks": chunks, "chunk_size": selected_chunk_size})
        else:
            state.update({"total_pages": 1, "chunks": [{"id": "document", "start": 0, "end": 1, "input": str(source), "status": "pending"}], "chunk_size": 1})
        atomic_json(state_path, state)

    emit("job_started", source=str(source), total_pages=state["total_pages"], chunks=len(state["chunks"]), profile=profile)
    state["status"] = "running"
    state.pop("error", None)
    run_started_at = time.time()
    previous_processing = float(state.get("processing_seconds", 0.0))
    work_root = output_root / ".job/results"
    work_root.mkdir(parents=True, exist_ok=True)
    for chunk_index, chunk in enumerate(state["chunks"]):
        if cancelled:
            break
        if chunk["status"] == "done":
            continue
        wait_for_memory()
        chunk_output = work_root / chunk["id"]
        chunk_output.mkdir(parents=True, exist_ok=True)
        chunk["status"] = "running"
        chunk["output"] = str(chunk_output)
        state["current_chunk"] = chunk_index
        state["updated_at"] = time.time()
        atomic_json(state_path, state)
        emit("chunk_started", index=chunk_index + 1, count=len(state["chunks"]), page_start=chunk["start"] + 1, page_end=chunk["end"], **progress_payload(state))
        command = [
            str(GLMOCR), "parse", chunk["input"], "--config", str(CONFIG), "--output", str(chunk_output),
            "--set", "pipeline.max_workers", str(profile["workers"]),
            "--set", "pipeline.page_maxsize", str(profile["pages"]),
            "--set", "pipeline.region_maxsize", str(profile["regions"]),
        ]
        code = run_streaming(command, "ocr", state, state_path)
        state["processing_seconds"] = previous_processing + time.time() - run_started_at
        state["active_elapsed"] = state["processing_seconds"]
        previous_processing = state["processing_seconds"]
        run_started_at = time.time()
        if code != 0:
            server_lost = not cancelled and not local_server_alive()
            resumable = cancelled or server_lost
            chunk["status"] = "paused" if resumable else "error"
            state["status"] = "paused" if resumable else "error"
            state["error"] = "Локальный сервер недоступен — задание можно продолжить" if server_lost else f"OCR завершился с кодом {code}"
            atomic_json(state_path, state)
            emit("server_lost" if server_lost else ("job_paused" if cancelled else "error"), message=state["error"], **progress_payload(state))
            return 75 if server_lost else (130 if cancelled else code)
        chunk["status"] = "done"
        chunk["completed_at"] = time.time()
        final_dir = merge_chunks(state, output_root)
        state["final_dir"] = str(final_dir)
        atomic_json(state_path, state)
        emit("chunk_completed", index=chunk_index + 1, count=len(state["chunks"]), **progress_payload(state))

    if cancelled:
        state["status"] = "paused"
        atomic_json(state_path, state)
        return 130

    final_dir = merge_chunks(state, output_root)
    state["final_dir"] = str(final_dir)
    atomic_json(state_path, state)
    emit("stage", stage="assembled", message="Markdown и JSON готовы")
    if args.enhance != "off":
        state["stage"] = "enhancement"
        atomic_json(state_path, state)
        emit("stage", stage="enhancement", message="Улучшаю извлечённые изображения")
        code = run_streaming([str(ENHANCE_PYTHON), str(ENHANCE_SCRIPT), str(final_dir), "--mode", args.enhance], "enhancement", state, state_path)
        if cancelled:
            state["status"] = "paused"; atomic_json(state_path, state)
            emit("job_paused", message="Улучшение изображений приостановлено", **progress_payload(state))
            return 130
        if code != 0:
            emit("log", level="warning", message="OCR готов, но улучшение изображений завершилось с ошибкой")

    if export_formats:
        state["stage"] = "export"
        atomic_json(state_path, state)
        emit("stage", stage="export", message="Создаю выбранные форматы")
        code = run_streaming([
            str(INSTALL_DIR / ".venv-sdk/bin/python"), str(EXPORT_SCRIPT),
            "--result", str(final_dir), "--source", str(source), "--output-root", str(output_root),
            "--formats", export_formats,
        ], "export", state, state_path)
        if cancelled:
            state["status"] = "paused"; atomic_json(state_path, state)
            emit("job_paused", message="Экспорт приостановлен", **progress_payload(state))
            return 130
        if code != 0:
            emit("log", level="warning", message="Основной результат готов, но часть экспортов не создана")

    state["stage"] = "complete"
    state["status"] = "complete"
    state["completed_at"] = time.time()
    state["updated_at"] = time.time()
    atomic_json(state_path, state)
    subprocess.run([str(INSTALL_DIR / ".venv-sdk/bin/python"), str(LIBRARY_SCRIPT), "index", str(output_root)], check=False)
    emit("complete", final_dir=str(final_dir), output=str(output_root), **progress_payload(state))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enhance", choices=("off", "safe", "hybrid"), default="hybrid")
    parser.add_argument("--exports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-formats", default="html,docx,searchable_pdf,clean_pdf,zip")
    parser.add_argument("--chunk-size", type=int, help="Размер возобновляемого блока PDF (по умолчанию автоматически)")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        return run_job(args)
    except Exception as exc:
        emit("error", message=str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
