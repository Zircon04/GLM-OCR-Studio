"""P5 pipeline gates.

  Gate C: tiled enhance ~= untiled enhance (valid-region placement; differences
          only at tile borders from finite tile_pad context).
  Gate D: MLX enhance(tile=0) ~= a PyTorch reference enhance (same reflect
          pre/mod pad + model + crop), validating the pad/crop wrapper. The
          model itself is already parity-locked (~3e-6) in test_parity.py.

Needs torch + local dist/ weights. Skips cleanly otherwise.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

import mlx.core as mx

from realesrgan_mlx.config import VARIANTS
from realesrgan_mlx.utils.realesrganer import RealESRGANer
from realesrgan_mlx.utils.weights import build_model

torch = pytest.importorskip("torch")

# basicsr functional_tensor shim (for the torch reference model)
import torchvision.transforms.functional as _tvf  # noqa: E402

_shim = types.ModuleType("torchvision.transforms.functional_tensor")
_shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules.setdefault("torchvision.transforms.functional_tensor", _shim)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST_DIR = os.path.join(ROOT, "dist")
WEIGHTS_DIR = os.path.join(ROOT, "weights")


def _have(variant):
    return (os.path.isdir(os.path.join(DIST_DIR, VARIANTS[variant].hf_name))
            and os.path.exists(os.path.join(WEIGHTS_DIR, f"{variant}.pth")))


def _rand_img(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


@pytest.mark.parametrize("variant", ["RealESRGAN_x4plus", "realesr-animevideov3"])
def test_gate_c_single_tile_is_exact(variant):
    """The rigorous tiling-correctness invariant: one tile covering the whole
    image (tile_pad clamped at all borders) must equal untiled BIT-EXACTLY.
    Proves the split/clamp/valid-region/placement math is correct, independent
    of receptive-field context effects."""
    if not _have(variant):
        pytest.skip(f"missing dist/weights for {variant}")
    cfg = VARIANTS[variant]
    img = _rand_img(70, 90, seed=1)
    model = build_model(variant, weights_dir=os.path.join(DIST_DIR, cfg.hf_name))
    with mx.stream(mx.cpu):
        full, _ = RealESRGANer(model, cfg.netscale, tile=0).enhance(img)
        one, _ = RealESRGANer(model, cfg.netscale, tile=200, tile_pad=10).enhance(img)
    d = int(np.abs(full.astype(np.int16) - one.astype(np.int16)).max())
    print(f"\n{variant}: single-tile vs untiled max={d} (uint8)")
    assert full.shape == one.shape
    assert d == 0, f"single-tile must be exact; got max={d}"


def test_gate_c_tile_pad_reduces_drift():
    """Multi-tile drift is the inherent tile_pad-context approximation (matches
    upstream). It must shrink as tile_pad grows — proving the knob works and the
    drift is bounded seam-band context, not a placement bug. (x4plus = largest RF.)"""
    variant = "RealESRGAN_x4plus"
    if not _have(variant):
        pytest.skip("missing dist/weights")
    cfg = VARIANTS[variant]
    img = _rand_img(70, 90, seed=1)
    model = build_model(variant, weights_dir=os.path.join(DIST_DIR, cfg.hf_name))
    with mx.stream(mx.cpu):
        full, _ = RealESRGANer(model, cfg.netscale, tile=0).enhance(img)
        d10 = np.abs(full.astype(np.int16)
                     - RealESRGANer(model, cfg.netscale, tile=32, tile_pad=10).enhance(img)[0]).mean()
        d24 = np.abs(full.astype(np.int16)
                     - RealESRGANer(model, cfg.netscale, tile=32, tile_pad=24).enhance(img)[0]).mean()
    print(f"\n{variant}: tile drift mean tile_pad=10 -> {d10:.3f}, tile_pad=24 -> {d24:.3f}")
    assert d24 < d10, "larger tile_pad should reduce seam drift"


def test_gate_c_srvgg_small_rf_tight():
    """SRVGGNetCompact has a small receptive field, so even tiny tiles drift
    little — assert a tight interior bound."""
    variant = "realesr-animevideov3"
    if not _have(variant):
        pytest.skip("missing dist/weights")
    cfg = VARIANTS[variant]
    img = _rand_img(70, 90, seed=1)
    model = build_model(variant, weights_dir=os.path.join(DIST_DIR, cfg.hf_name))
    with mx.stream(mx.cpu):
        full, _ = RealESRGANer(model, cfg.netscale, tile=0).enhance(img)
        tiled, _ = RealESRGANer(model, cfg.netscale, tile=32, tile_pad=10).enhance(img)
    d = np.abs(full.astype(np.int16) - tiled.astype(np.int16))
    print(f"\n{variant}: tiled-vs-untiled mean={d.mean():.3f} max={d.max()}")
    assert d.mean() < 0.5


def _torch_reference_enhance(variant, img):
    """numpy reflect pre/mod pad -> basicsr torch model -> crop, in NCHW."""
    from basicsr.archs.rrdbnet_arch import RRDBNet as PtRRDB
    from basicsr.archs.srvgg_arch import SRVGGNetCompact as PtSRVGG
    cfg = VARIANTS[variant]
    if cfg.arch == "RRDBNet":
        m = PtRRDB(3, 3, scale=cfg.netscale, num_feat=cfg.num_feat,
                   num_block=cfg.num_block, num_grow_ch=cfg.num_grow_ch)
    else:
        m = PtSRVGG(3, 3, num_feat=cfg.num_feat, num_conv=cfg.num_conv,
                    upscale=cfg.netscale, act_type=cfg.act_type)
    ck = torch.load(os.path.join(WEIGHTS_DIR, f"{variant}.pth"),
                    map_location="cpu", weights_only=True)
    sd = ck.get("params_ema", ck.get("params", ck))
    m.load_state_dict(sd, strict=True); m.eval()

    s = cfg.netscale
    pre_pad = 10
    mod_scale = {2: 2, 1: 4}.get(s)
    a = img.astype(np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    t = torch.nn.functional.pad(t, (0, pre_pad, 0, pre_pad), "reflect")
    mph = mpw = 0
    if mod_scale:
        h, w = t.shape[2], t.shape[3]
        mph = (mod_scale - h % mod_scale) % mod_scale
        mpw = (mod_scale - w % mod_scale) % mod_scale
        t = torch.nn.functional.pad(t, (0, mpw, 0, mph), "reflect")
    with torch.no_grad():
        o = m(t)
    h, w = o.shape[2], o.shape[3]
    if mod_scale:
        o = o[:, :, 0:h - mph * s, 0:w - mpw * s]
    h, w = o.shape[2], o.shape[3]
    o = o[:, :, 0:h - pre_pad * s, 0:w - pre_pad * s]
    o = np.clip(o[0].permute(1, 2, 0).numpy(), 0, 1)
    return (o * 255).round().astype(np.uint8)


@pytest.mark.parametrize("variant", ["RealESRGAN_x2plus", "realesr-general-x4v3"])
def test_gate_d_pipeline_matches_pytorch(variant):
    if not _have(variant):
        pytest.skip(f"missing dist/weights for {variant}")
    cfg = VARIANTS[variant]
    img = _rand_img(48, 64, seed=2)

    ref = _torch_reference_enhance(variant, img)
    model = build_model(variant, weights_dir=os.path.join(DIST_DIR, cfg.hf_name))
    with mx.stream(mx.cpu):
        out, _ = RealESRGANer(model, cfg.netscale, tile=0).enhance(img)

    assert out.shape == ref.shape, (out.shape, ref.shape)
    d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    mse = np.mean((out.astype(np.float32) - ref.astype(np.float32)) ** 2)
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"\n{variant}: MLX-vs-PyTorch enhance  mean={d.mean():.4f}  max={d.max()}  PSNR={psnr:.1f}dB")
    assert psnr > 50.0, f"pipeline diverges from PyTorch: PSNR={psnr:.1f}dB"
