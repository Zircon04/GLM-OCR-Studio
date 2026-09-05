"""P7: publish the fp16 dist/ artifacts to mlx-community.

Creates mlx-community/<hf_name> for each variant, writes a model card, and
uploads dist/<hf_name>/ (model.safetensors [+ model_wdn] + config.json + README).

Run after scripts/convert_to_mlx.py. Requires HF auth (xocialize @ mlx-community).
  python scripts/publish_hf.py            # all variants
  python scripts/publish_hf.py x4plus     # filter by substring
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi, create_repo, upload_folder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realesrgan_mlx.config import VARIANTS, VariantConfig  # noqa: E402

ORG = "mlx-community"
DIST = "dist"

PARITY = {  # cpu-fp32 full-forward, fp16-vs-golden
    "RealESRGAN_x4plus": ("3.8e-6", "1.3e-3"),
    "RealESRGAN_x4plus_anime_6B": ("3.0e-6", "6.1e-4"),
    "RealESRGAN_x2plus": ("2.4e-6", "5.5e-4"),
    "realesr-animevideov3": ("3.9e-6", "1.5e-3"),
    "realesr-general-x4v3": ("4.1e-6", "2.0e-3"),
}


def model_card(cfg: VariantConfig) -> str:
    fp32, fp16 = PARITY[cfg.name]
    dn = ("\n- **Denoise strength**: this model ships with a paired `model_wdn.safetensors`; "
          "`build_model(..., denoise_strength=s)` blends `s*model + (1-s)*wdn` "
          "(`1.0` keeps detail, `0.0` = max denoise).\n" if cfg.wdn_url else "\n")
    return f"""---
license: bsd-3-clause
library_name: mlx
pipeline_tag: image-to-image
tags:
- mlx
- super-resolution
- real-esrgan
- image-to-image
- apple-silicon
---

# {cfg.hf_name} (MLX)

Apple **MLX** fp16 port of Real-ESRGAN **{cfg.name}** ({cfg.arch}, ×{cfg.netscale}),
for super-resolution on Apple Silicon. Converted from the official
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) release checkpoint
(BSD-3-Clause).

## Usage

```bash
pip install realesrgan-mlx   # https://github.com/xocialize/realesrgan-mlx
realesrgan-mlx -i input.png -o out/ -n {cfg.name}
```

```python
from realesrgan_mlx.pipeline_mlx import make_upsampler, upscale_image
up = make_upsampler("{cfg.name}", tile=256)   # tile>0 caps memory on large images
out = upscale_image("input.png", up)
```

`make_upsampler` downloads these weights automatically.

## Details

- **Architecture**: {cfg.arch} (`num_feat={cfg.num_feat}`{
    f", `num_block={cfg.num_block}`, `num_grow_ch={cfg.num_grow_ch}`"
    if cfg.arch == "RRDBNet"
    else f", `num_conv={cfg.num_conv}`, `act={cfg.act_type}`"})
- **Scale**: ×{cfg.netscale} · **Precision**: fp16
- **Parity vs PyTorch**: full-forward {fp32} (CPU fp32); fp16 vs fp32 golden {fp16}.{dn}
## License

BSD-3-Clause (upstream Real-ESRGAN, Xintao Wang et al.).
"""


def publish(cfg: VariantConfig) -> None:
    repo_id = f"{ORG}/{cfg.hf_name}"
    folder = os.path.join(DIST, cfg.hf_name)
    if not os.path.isdir(folder):
        print(f"!! missing {folder}; run convert_to_mlx.py"); return
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(model_card(cfg))
    create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model",
                  commit_message=f"Add {cfg.hf_name} MLX fp16 (Real-ESRGAN port)")
    print(f"OK  https://huggingface.co/{repo_id}")


def main() -> None:
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    HfApi().whoami()  # fail fast if not authed
    for cfg in VARIANTS.values():
        if flt and flt not in cfg.name:
            continue
        publish(cfg)


if __name__ == "__main__":
    main()
