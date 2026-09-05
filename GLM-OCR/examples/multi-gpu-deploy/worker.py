"""Worker process — runs inside a subprocess with CUDA_VISIBLE_DEVICES
already set to a single GPU."""

import sys
import json
from pathlib import Path
from typing import Any, Dict, List

from gpu_utils import _print_err
from engine import write_progress


def run_worker(args) -> None:
    """Process a shard of files using the GLM-OCR pipeline.

    ``cuda_visible_devices="0"`` always refers to the intended physical GPU
    because the parent process restricts visibility via CUDA_VISIBLE_DEVICES.
    """
    with open(args.filelist, "r") as f:
        files = json.load(f)

    if not files:
        write_progress(args.progress_file, 0, 0, status="done")
        return

    total = len(files)
    completed = 0
    failed = 0
    failed_files: List[Dict[str, str]] = []

    write_progress(args.progress_file, 0, total, 0, "loading_model")

    try:
        from glmocr.api import GlmOcr
        from glmocr.utils.logging import configure_logging

        configure_logging(level=args.log_level or "WARNING")

        glm_kwargs: Dict[str, Any] = {
            "ocr_api_port": args.port,
            "cuda_visible_devices": "0",
        }
        if args.config:
            glm_kwargs["config_path"] = args.config

        with GlmOcr(**glm_kwargs) as parser:
            write_progress(args.progress_file, 0, total, 0, "running")

            no_save = getattr(args, "no_save", False)

            for result in parser.parse(files, stream=True, save_layout_visualization=not no_save):
                completed += 1

                if not no_save:
                    try:
                        save_dir = args.output
                        if args.input_root and result.original_images:
                            try:
                                rel = Path(
                                    result.original_images[0]
                                ).parent.relative_to(args.input_root)
                                if str(rel) != ".":
                                    save_dir = str(Path(args.output) / rel)
                            except ValueError:
                                pass

                        result.save(output_dir=save_dir)
                    except Exception as e:
                        failed += 1
                        src = (
                            result.original_images[0]
                            if result.original_images
                            else "unknown"
                        )
                        failed_files.append({"file": src, "error": str(e)})

                write_progress(
                    args.progress_file, completed, total, failed, "running"
                )

    except Exception as e:
        import traceback

        _print_err(f"[GPU {args.gpu_id}] Worker error: {e}")
        traceback.print_exc(file=sys.stderr)
        write_progress(
            args.progress_file, completed, total, failed, f"error: {e}"
        )
        _save_failed_list(args.progress_file, failed_files)
        return

    status = "done" if failed == 0 else "done_with_errors"
    write_progress(args.progress_file, completed, total, failed, status)
    if not getattr(args, "no_save", False):
        _save_failed_list(args.progress_file, failed_files)


def _save_failed_list(
    progress_file: str, failed_files: List[Dict[str, str]]
) -> None:
    if not failed_files:
        return
    path = progress_file.replace(".json", "_failed.json")
    try:
        with open(path, "w") as f:
            json.dump(failed_files, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
