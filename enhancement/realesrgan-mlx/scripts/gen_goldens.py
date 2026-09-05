"""P1: generate the PyTorch parity oracle.

Requires the `[parity]` extra (torch + basicsr). For each variant:
  1. build the basicsr arch from config.py, load the .pth (params_ema|params)
  2. eval() on CPU fp32
  3. run a fixed seeded NCHW input, capturing the full-forward output + a deep
     intermediate (via forward hook) for isolation if full parity fails
  4. save goldens/<variant>.npz  (input_nchw, output_nchw, intermediates, fp32)

Parity is gated on mx.cpu fp32 (Apple-GPU conv accumulation is tf32-like), so we
keep everything fp32 here. Run:  python scripts/gen_goldens.py
"""

from __future__ import annotations

# --- basicsr/torchvision compat shim (functional_tensor moved) --------------
import sys
import types

import torchvision.transforms.functional as _tvf

_shim = types.ModuleType("torchvision.transforms.functional_tensor")
_shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules.setdefault("torchvision.transforms.functional_tensor", _shim)
# ---------------------------------------------------------------------------

import os

import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realesrgan_mlx.config import VARIANTS, VariantConfig  # noqa: E402

WEIGHTS_DIR = "weights"
GOLDENS_DIR = "goldens"
SEED = 1234
IN_HW = 32  # small input; must be divisible by the largest input-unshuffle factor


def build_pt(cfg: VariantConfig) -> torch.nn.Module:
    if cfg.arch == "RRDBNet":
        return RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=cfg.netscale,
            num_feat=cfg.num_feat, num_block=cfg.num_block, num_grow_ch=cfg.num_grow_ch,
        )
    if cfg.arch == "SRVGGNetCompact":
        return SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=cfg.num_feat,
            num_conv=cfg.num_conv, upscale=cfg.netscale, act_type=cfg.act_type,
        )
    raise ValueError(cfg.arch)


def load_weights(model: torch.nn.Module, pth: str) -> None:
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ck else ("params" if "params" in ck else None)
    sd = ck[key] if key else ck
    model.load_state_dict(sd, strict=True)


def deep_hook_target(model: torch.nn.Module, cfg: VariantConfig):
    """A representative deep submodule to snapshot for isolation."""
    if cfg.arch == "RRDBNet":
        return ("conv_body", model.conv_body)
    # SRVGGNetCompact: the first conv's output (body[0])
    return ("body0", model.body[0])


def main() -> None:
    os.makedirs(GOLDENS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    for name, cfg in VARIANTS.items():
        pth = os.path.join(WEIGHTS_DIR, f"{name}.pth")
        if not os.path.exists(pth):
            print(f"!! missing {pth}; skip"); continue

        model = build_pt(cfg)
        load_weights(model, pth)
        model.eval()

        x = rng.random((1, 3, IN_HW, IN_HW), dtype=np.float32)  # NCHW, [0,1)
        xt = torch.from_numpy(x)

        captured = {}
        hname, hmod = deep_hook_target(model, cfg)
        h = hmod.register_forward_hook(
            lambda m, i, o: captured.__setitem__(hname, o.detach().cpu().numpy())
        )
        with torch.no_grad():
            y = model(xt).detach().cpu().numpy()
        h.remove()

        out = {"input_nchw": x, "output_nchw": y, **captured}
        np.savez(os.path.join(GOLDENS_DIR, f"{name}.npz"), **out)
        print(f"OK  {name:32s} in{tuple(x.shape)} -> out{tuple(y.shape)} "
              f"| {hname}{tuple(captured[hname].shape)} | "
              f"out[min={y.min():.4f} max={y.max():.4f}]")


if __name__ == "__main__":
    main()
