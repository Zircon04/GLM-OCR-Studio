"""P7: M5 Max benchmarks — per-variant latency + peak memory, and the
tile-vs-no-tile memory story for a large image.

Runs on the default device (GPU — the on-device target), fp16 weights from
dist/. Numbers are wall-clock for a full RealESRGANer.enhance() including
pre/post processing. Writes docs/REPORT.md.

Usage:  python scripts/benchmark.py
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realesrgan_mlx.config import VARIANTS  # noqa: E402
from realesrgan_mlx.utils.realesrganer import RealESRGANer  # noqa: E402
from realesrgan_mlx.utils.weights import build_model  # noqa: E402

DIST = "dist"


def _peak_mb() -> float:
    for fn in ("get_peak_memory",):
        if hasattr(mx, fn):
            return getattr(mx, fn)() / 1e6
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / 1e6
    return float("nan")


def _reset_peak():
    for fn in ("reset_peak_memory",):
        if hasattr(mx, fn):
            getattr(mx, fn)(); return
    if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _time_enhance(up: RealESRGANer, img: np.ndarray, runs: int = 3) -> tuple[float, float, tuple]:
    out, _ = up.enhance(img)  # warmup (compiles kernels)
    _reset_peak()
    t0 = time.perf_counter()
    for _ in range(runs):
        out, _ = up.enhance(img)
    dt = (time.perf_counter() - t0) / runs
    return dt, _peak_mb(), out.shape


def main() -> None:
    rng = np.random.default_rng(0)
    lines = ["# Real-ESRGAN-MLX — Benchmarks (M5 Max, fp16, GPU)\n",
             "Wall-clock per `enhance()` (incl. pre/post), mean of 3 runs.\n"]

    # 1) per-variant on a 256x256 input
    lines.append("\n## Per-variant — 256×256 input, no tiling\n")
    lines.append("| Variant | Arch | Scale | Output | Latency | Peak mem |")
    lines.append("|---|---|---|---|---|---|")
    img256 = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    for name, cfg in VARIANTS.items():
        wdir = os.path.join(DIST, cfg.hf_name)
        if not os.path.isdir(wdir):
            continue
        up = RealESRGANer(build_model(name, weights_dir=wdir), cfg.netscale, tile=0)
        dt, mem, shp = _time_enhance(up, img256)
        lines.append(f"| {name} | {cfg.arch} | ×{cfg.netscale} | {shp[1]}×{shp[0]} | "
                     f"{dt * 1000:.0f} ms | {mem:.0f} MB |")
        print(lines[-1])

    # 2) tile vs no-tile on a large input (x4plus, the heaviest)
    lines.append("\n## Tiling — x4plus on a large input (memory story)\n")
    lines.append("| Input | Mode | Output | Latency | Peak mem |")
    lines.append("|---|---|---|---|---|")
    cfg = VARIANTS["RealESRGAN_x4plus"]
    wdir = os.path.join(DIST, cfg.hf_name)
    big = (rng.random((1024, 1024, 3)) * 255).astype(np.uint8)
    model = build_model("RealESRGAN_x4plus", weights_dir=wdir)
    for mode, tile in [("no-tile", 0), ("tile=256", 256), ("tile=128", 128)]:
        up = RealESRGANer(model, cfg.netscale, tile=tile, tile_pad=16)
        try:
            dt, mem, shp = _time_enhance(up, big, runs=1)
            row = f"| 1024×1024 | {mode} | {shp[1]}×{shp[0]} | {dt * 1000:.0f} ms | {mem:.0f} MB |"
        except Exception as e:  # noqa: BLE001
            row = f"| 1024×1024 | {mode} | — | OOM/err | {type(e).__name__} |"
        lines.append(row); print(row)

    os.makedirs("docs", exist_ok=True)
    with open("docs/REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote docs/REPORT.md")


if __name__ == "__main__":
    main()
