"""High-level pipeline + CLI.

    realesrgan-mlx -i input.png -o out/ --variant RealESRGAN_x4plus \\
        --outscale 4 --tile 0 --denoise_strength 0.5

--denoise_strength only applies to realesr-general-x4v3 (dni blend of general +
wdn weights at load time). --tile > 0 enables tiled inference for large images.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from .config import DEFAULT_VARIANT, VARIANTS
from .utils.realesrganer import RealESRGANer
from .utils.weights import build_model


def make_upsampler(variant=DEFAULT_VARIANT, denoise_strength=1.0, tile=0,
                   tile_pad=10, pre_pad=10, weights_dir=None) -> RealESRGANer:
    cfg = VARIANTS[variant]
    model = build_model(variant, weights_dir=weights_dir, denoise_strength=denoise_strength)
    return RealESRGANer(model, cfg.netscale, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad)


def upscale_image(path: str, upsampler: RealESRGANer, outscale: float | None = None):
    img = np.asarray(Image.open(path))
    out, _mode = upsampler.enhance(img, outscale=outscale)
    return out


def cli_main() -> None:
    p = argparse.ArgumentParser(description="Real-ESRGAN MLX upscaler")
    p.add_argument("-i", "--input", required=True, help="input image or directory")
    p.add_argument("-o", "--output", default="results", help="output directory")
    p.add_argument("-n", "--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    p.add_argument("-s", "--outscale", type=float, default=None,
                   help="final upsampling scale (default: the variant's net scale)")
    p.add_argument("-t", "--tile", type=int, default=0, help="tile size, 0=off")
    p.add_argument("--tile_pad", type=int, default=10)
    p.add_argument("--pre_pad", type=int, default=10)
    p.add_argument("-dn", "--denoise_strength", type=float, default=1.0,
                   help="realesr-general-x4v3 only: 1.0=keep noise, 0.0=max denoise")
    p.add_argument("--weights_dir", default=None)
    args = p.parse_args()

    up = make_upsampler(args.variant, args.denoise_strength, args.tile,
                        args.tile_pad, args.pre_pad, args.weights_dir)

    os.makedirs(args.output, exist_ok=True)
    inputs = ([args.input] if os.path.isfile(args.input)
              else [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))])
    for path in inputs:
        try:
            out = upscale_image(path, up, outscale=args.outscale)
        except Exception as e:  # noqa: BLE001
            print(f"skip {path}: {e}"); continue
        stem = os.path.splitext(os.path.basename(path))[0]
        dst = os.path.join(args.output, f"{stem}_{args.variant}.png")
        Image.fromarray(out).save(dst)
        print(f"{path} -> {dst}  {out.shape}")


if __name__ == "__main__":
    cli_main()
