"""GPU detection, port checking, file collection, and sharding utilities."""

import os
import sys
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".pdf"}

DEFAULT_BASE_PORT = 8080
DEFAULT_MIN_FREE_MB = 16000


def _print_err(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


# =========================================================================
# GPU Detection
# =========================================================================

def get_gpu_info() -> List[Dict[str, Any]]:
    """Query GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "id": int(parts[0]),
                        "name": parts[1],
                        "total_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                        "used_mb": int(parts[4]),
                    }
                )
        return gpus
    except FileNotFoundError:
        _print_err("[ERROR] nvidia-smi not found. Is the NVIDIA driver installed?")
        return []
    except Exception as e:
        _print_err(f"[ERROR] Failed to query GPU info: {e}")
        return []


def filter_available_gpus(
    gpus: List[Dict],
    min_free_mb: int,
    gpu_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Filter GPUs that have enough free VRAM."""
    available = []
    for gpu in gpus:
        if gpu_ids is not None and gpu["id"] not in gpu_ids:
            continue
        if gpu["free_mb"] >= min_free_mb:
            available.append(gpu)
        else:
            _print_err(
                f"  [SKIP] GPU {gpu['id']} ({gpu['name']}): "
                f"{gpu['free_mb']}MB free < {min_free_mb}MB required"
            )
    return available


# =========================================================================
# Port Checking
# =========================================================================

def is_port_available(port: int) -> bool:
    """Check if a TCP port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def find_available_ports(base_port: int, count: int) -> List[int]:
    """Find *count* available ports starting from *base_port*.

    Skips any port that is already in use.
    """
    ports: List[int] = []
    port = base_port
    max_port = base_port + count * 10
    while len(ports) < count and port < max_port:
        if is_port_available(port):
            ports.append(port)
        else:
            _print_err(f"  [SKIP] Port {port} is occupied, trying next...")
        port += 1
    return ports


# =========================================================================
# File Collection and Sharding
# =========================================================================

def collect_files(input_path: str) -> List[str]:
    """Collect all supported image/PDF files from input path (recursive)."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            return [str(path.absolute())]
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.is_dir():
        seen: set = set()
        files: List[str] = []
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                abs_p = str(p.absolute())
                if abs_p not in seen:
                    seen.add(abs_p)
                    files.append(abs_p)
        if not files:
            raise ValueError(f"No image/PDF files found in: {input_path}")
        return files
    raise ValueError(f"Path does not exist: {input_path}")


def shard_files(files: List[str], n_shards: int) -> List[List[str]]:
    """Distribute files across shards using round-robin."""
    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for i, f in enumerate(files):
        shards[i % n_shards].append(f)
    return shards
