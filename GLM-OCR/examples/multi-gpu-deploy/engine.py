"""Engine service management and progress tracking."""

import os
import sys
import json
import time
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================
# Engine Service Management
# =========================================================================

def build_engine_cmd(
    engine: str,
    model: str,
    port: int,
    extra_args: str = "",
    engine_log_level: str = "warning",
) -> List[str]:
    """Build command to start sglang or vLLM service.

    Default speculative-decoding flags are included for each engine so that
    GLM-OCR runs with MTP (multi-token prediction) out of the box.  Pass
    ``extra_args`` to override or extend these defaults.
    """
    if engine == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model",
            model,
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--served-model-name",
            "glm-ocr",
        ]
    elif engine == "vllm":
        cmd = [
            "vllm",
            "serve",
            model,
            "--port",
            str(port),
            "--allowed-local-media-path",
            "/",
            "--speculative-config",
            '{"method": "mtp", "num_speculative_tokens": 1}',
            "--served-model-name",
            "glm-ocr",
            "--uvicorn-log-level",
            engine_log_level.lower(),
        ]
    else:
        raise ValueError(f"Unknown engine: {engine}")

    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def start_engine(
    engine: str,
    model: str,
    gpu_id: int,
    port: int,
    log_dir: str,
    extra_args: str = "",
    engine_log_level: str = "warning",
) -> Tuple[subprocess.Popen, Path, Any]:
    """Start an engine service on a specific GPU.

    Returns (process, log_path, log_file_handle).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if engine == "vllm":
        env["VLLM_LOGGING_LEVEL"] = engine_log_level.upper()

    cmd = build_engine_cmd(engine, model, port, extra_args, engine_log_level)
    log_path = Path(log_dir) / f"engine_gpu{gpu_id}_port{port}.log"
    log_fh = open(log_path, "w")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log_path, log_fh


def wait_for_service(
    port: int,
    proc: subprocess.Popen,
    timeout: int = 600,
    interval: int = 5,
) -> Tuple[bool, int]:
    """Wait for a service to become ready by polling /v1/models.

    Returns (success, elapsed_seconds).
    """
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/v1/models"
    # Bypass any HTTP proxy for localhost connections
    no_proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(no_proxy_handler)
    start = time.time()

    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False, int(time.time() - start)
        try:
            req = urllib.request.Request(url, method="GET")
            with opener.open(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, int(time.time() - start)
        except Exception:
            pass
        time.sleep(interval)

    return False, int(time.time() - start)


# =========================================================================
# Progress Tracking
# =========================================================================

def write_progress(
    path: str,
    completed: int,
    total: int,
    failed: int = 0,
    status: str = "running",
) -> None:
    """Atomically write progress to a JSON file."""
    data = {
        "completed": completed,
        "total": total,
        "failed": failed,
        "status": status,
        "timestamp": time.time(),
    }
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def read_progress(path: str) -> Optional[Dict]:
    """Read progress from a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
