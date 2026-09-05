"""P4: convert upstream .pth -> MLX fp16 safetensors + config.json.

For each variant:
  * load weights/<variant>.pth  (params_ema | params)
  * convert_state_dict: conv (O,I,kH,kW)->(O,kH,kW,I); 1-D bias/PReLU passthrough
  * cast to fp16, mx.eval EVERY tensor (lazy -> silent zeros otherwise)
  * write dist/<hf_name>/{model.safetensors, config.json}

realesr-general-x4v3 also gets model_wdn.safetensors (the denoise sibling) so the
pipeline can dni-blend at load time for a denoise-strength slider.

Published dtype is fp16 (matches upstream half-precision default; every variant
stays < 100 MB for the ADR-0005 lazy-download pattern). Parity is validated
separately against the fp32 golden in tests.

Run:  python scripts/convert_to_mlx.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realesrgan_mlx.config import VARIANTS, VariantConfig  # noqa: E402
from realesrgan_mlx.utils.convert import convert_state_dict  # noqa: E402

WEIGHTS_DIR = "weights"
DIST_DIR = "dist"


def load_pt_numpy(pth: str) -> dict[str, np.ndarray]:
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ck else ("params" if "params" in ck else None)
    sd = ck[key] if key else ck
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def to_fp16_materialized(sd_np: dict[str, np.ndarray]) -> dict[str, mx.array]:
    out = convert_state_dict(sd_np)
    out = {k: v.astype(mx.float16) for k, v in out.items()}
    mx.eval(out)  # materialize — guards against lazy zeros on save
    return out


def write_variant(cfg: VariantConfig) -> None:
    outdir = os.path.join(DIST_DIR, cfg.hf_name)
    os.makedirs(outdir, exist_ok=True)

    pth = os.path.join(WEIGHTS_DIR, f"{cfg.name}.pth")
    weights = to_fp16_materialized(load_pt_numpy(pth))
    st_path = os.path.join(outdir, "model.safetensors")
    mx.save_safetensors(st_path, weights, metadata={"format": "mlx"})

    extra = {}
    if cfg.wdn_url:
        wdn_pth = os.path.join(WEIGHTS_DIR, "realesr-general-wdn-x4v3.pth")
        if os.path.exists(wdn_pth):
            wdn = to_fp16_materialized(load_pt_numpy(wdn_pth))
            mx.save_safetensors(os.path.join(outdir, "model_wdn.safetensors"),
                                wdn, metadata={"format": "mlx"})
            extra["has_wdn_denoise"] = True

    cfg_json = {k: v for k, v in asdict(cfg).items()}
    cfg_json.update(extra)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)

    size_mb = os.path.getsize(st_path) / 1e6
    print(f"OK  {cfg.hf_name:28s} {len(weights):3d} tensors  {size_mb:6.1f} MB"
          + ("  +wdn" if extra.get("has_wdn_denoise") else ""))


def main() -> None:
    os.makedirs(DIST_DIR, exist_ok=True)
    for cfg in VARIANTS.values():
        write_variant(cfg)


if __name__ == "__main__":
    main()
