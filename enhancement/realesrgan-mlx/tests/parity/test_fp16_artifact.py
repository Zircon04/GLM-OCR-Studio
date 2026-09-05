"""P4: validate the published fp16 safetensors artifacts against fp32 goldens.

Confirms (a) materialization worked (no silent-zero tensors) and (b) fp16 weights
stay within the fp16 full-forward gate (< 1e-2). Loads via the real build_model()
path from dist/. Skips when dist/ or goldens/ are absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

from realesrgan_mlx.config import VARIANTS
from realesrgan_mlx.utils.weights import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST_DIR = os.path.join(ROOT, "dist")
GOLDENS_DIR = os.path.join(ROOT, "goldens")

FP16_TOL = 1e-2


@pytest.mark.parametrize("name", list(VARIANTS.keys()))
def test_fp16_artifact_parity(name):
    cfg = VARIANTS[name]
    wdir = os.path.join(DIST_DIR, cfg.hf_name)
    golden = os.path.join(GOLDENS_DIR, f"{name}.npz")
    if not (os.path.isdir(wdir) and os.path.exists(golden)):
        pytest.skip(f"missing dist/golden for {name}")

    g = np.load(golden)
    x_nchw = g["input_nchw"].astype(np.float32)
    y_ref = g["output_nchw"].astype(np.float32)

    model = build_model(name, weights_dir=wdir)

    with mx.stream(mx.cpu):
        x = mx.array(np.transpose(x_nchw, (0, 2, 3, 1)))
        y = model(x)
        mx.eval(y)
    y_nchw = np.transpose(np.array(y).astype(np.float32), (0, 3, 1, 2))

    max_abs = float(np.max(np.abs(y_nchw - y_ref)))
    assert not np.allclose(y_nchw, 0.0), f"{name}: all-zero output (materialize bug?)"
    print(f"\n{name:32s} fp16 max_abs={max_abs:.3e} (tol {FP16_TOL:.0e})")
    assert max_abs < FP16_TOL, f"{name} fp16 diverges: {max_abs:.3e}"


def test_dni_denoise_blend_changes_output():
    """realesr-general-x4v3: denoise_strength<1 must alter the output."""
    name = "realesr-general-x4v3"
    cfg = VARIANTS[name]
    wdir = os.path.join(DIST_DIR, cfg.hf_name)
    golden = os.path.join(GOLDENS_DIR, f"{name}.npz")
    if not (os.path.isdir(wdir) and os.path.exists(golden)):
        pytest.skip("missing general dist/golden")

    g = np.load(golden)
    x = mx.array(np.transpose(g["input_nchw"].astype(np.float32), (0, 2, 3, 1)))

    with mx.stream(mx.cpu):
        y1 = np.array(build_model(name, weights_dir=wdir, denoise_strength=1.0)(x))
        y0 = np.array(build_model(name, weights_dir=wdir, denoise_strength=0.0)(x))
    diff = float(np.max(np.abs(y1 - y0)))
    print(f"\ndni blend max_abs(s=1 vs s=0)={diff:.3e}")
    assert diff > 1e-3, "denoise blend had no effect"
