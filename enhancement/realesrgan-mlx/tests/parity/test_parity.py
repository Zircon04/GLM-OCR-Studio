"""PT <-> MLX full-forward parity, gated on mx.cpu fp32.

Apple-GPU conv accumulation is tf32-like, so correctness is gated on CPU where
MLX matches PyTorch bit-for-bit-ish. Thresholds (mlx-porting): single layer
< 1e-4, full forward < 1e-2; > 1e-1 is a port bug. Running fp32 here, so we
expect far tighter than the fp16 gate.

Needs the `[parity]` extra (torch) plus local weights/ + goldens/ (gitignored;
produced by scripts/gen_goldens.py). Skips cleanly when those are absent.

  P2 Gate A: RRDBNet  (x4plus, anime_6B, x2plus)
  P3 Gate B: SRVGGNetCompact (animevideov3, general-x4v3)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx

from realesrgan_mlx.archs import RRDBNet, SRVGGNetCompact
from realesrgan_mlx.config import VARIANTS, VariantConfig
from realesrgan_mlx.utils.convert import convert_state_dict

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(ROOT, "weights")
GOLDENS_DIR = os.path.join(ROOT, "goldens")

FULL_FORWARD_TOL = 1e-2


def _build_mlx(cfg: VariantConfig):
    if cfg.arch == "RRDBNet":
        return RRDBNet(3, 3, scale=cfg.netscale, num_feat=cfg.num_feat,
                       num_block=cfg.num_block, num_grow_ch=cfg.num_grow_ch)
    return SRVGGNetCompact(3, 3, num_feat=cfg.num_feat, num_conv=cfg.num_conv,
                           upscale=cfg.netscale, act_type=cfg.act_type)


def _load_pt_state_numpy(pth: str) -> dict[str, np.ndarray]:
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ck else ("params" if "params" in ck else None)
    sd = ck[key] if key else ck
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


@pytest.mark.parametrize("name", list(VARIANTS.keys()))
def test_full_forward_parity(name):
    cfg = VARIANTS[name]
    pth = os.path.join(WEIGHTS_DIR, f"{name}.pth")
    golden = os.path.join(GOLDENS_DIR, f"{name}.npz")
    if not (os.path.exists(pth) and os.path.exists(golden)):
        pytest.skip(f"missing weights/golden for {name}")

    g = np.load(golden)
    x_nchw = g["input_nchw"].astype(np.float32)
    y_ref_nchw = g["output_nchw"].astype(np.float32)

    model = _build_mlx(cfg)
    weights = convert_state_dict(_load_pt_state_numpy(pth))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()

    # Gate correctness on CPU (GPU conv accumulation is tf32-like).
    with mx.stream(mx.cpu):
        x_nhwc = mx.array(np.transpose(x_nchw, (0, 2, 3, 1)))
        y = model(x_nhwc)
        mx.eval(y)
    y_nchw = np.transpose(np.array(y), (0, 3, 1, 2))

    assert y_nchw.shape == y_ref_nchw.shape, (y_nchw.shape, y_ref_nchw.shape)
    max_abs = float(np.max(np.abs(y_nchw - y_ref_nchw)))
    print(f"\n{name:32s} max_abs={max_abs:.3e} (tol {FULL_FORWARD_TOL:.0e})")
    assert max_abs < FULL_FORWARD_TOL, f"{name} diverges: max_abs={max_abs:.3e}"
