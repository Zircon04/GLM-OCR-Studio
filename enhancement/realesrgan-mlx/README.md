# realesrgan-mlx

Apple **MLX** port of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — practical
image & video super-resolution on Apple Silicon. **BSD-3-Clause.**

A net-new clean port (not a fork of the code-less `themindstudio` `.npz` blob): both
upstream architectures, **five published variants**, **tile-based inference for large
images**, a denoise-strength control, and a PyTorch parity oracle (all 5 variants match
the reference to **~3e-6** on CPU fp32).

## Variants

| Variant | Arch | Scale | Best for | fp16 size |
|---|---|---|---|---|
| `RealESRGAN_x4plus` | RRDBNet-23 | ×4 | general photos | 33.5 MB |
| `RealESRGAN_x4plus_anime_6B` | RRDBNet-6 | ×4 | anime / illustration | 9.0 MB |
| `RealESRGAN_x2plus` | RRDBNet-23 | ×2 | general photos | 33.5 MB |
| `realesr-animevideov3` | SRVGGNetCompact-16 | ×4 | anime video | 1.2 MB |
| `realesr-general-x4v3` | SRVGGNetCompact-32 | ×4 | general + denoise slider | 2.4 MB |

(`RealESRGAN_x4plus` is the default variant.)

## Install

```bash
pip install -e .            # runtime: mlx, numpy, pillow, huggingface_hub (no torch)
pip install -e ".[parity]"  # + torch/basicsr/realesrgan to regenerate the parity oracle
pip install -e ".[dev]"     # + pytest (test runner only)
```

## Usage

### CLI

```bash
# 4× upscale a photo
realesrgan-mlx -i input.png -o results/ -n RealESRGAN_x4plus

# anime, with tiling for a large frame (caps peak memory)
realesrgan-mlx -i frame.png -o results/ -n realesr-animevideov3 -t 256

# general model with denoise strength + a custom final scale
realesrgan-mlx -i photo.jpg -o results/ -n realesr-general-x4v3 -dn 0.5 -s 3
```

Flags: `-n` variant · `-s` final outscale · `-t` tile size (0 = off) · `--tile_pad` ·
`--pre_pad` · `-dn` denoise strength (general-x4v3 only; `1.0` keeps detail/noise, `0.0` =
max denoise) · `--weights_dir` local override. `-i` accepts a single image or a directory.

### Python

```python
from realesrgan_mlx.pipeline_mlx import make_upsampler, upscale_image

up = make_upsampler("RealESRGAN_x4plus", tile=256)
out = upscale_image("input.png", up)          # HxWx3 uint8 ndarray, 4× larger
```

`make_upsampler(...)` resolves weights in this order: explicit `weights_dir` →
`$REALESRGAN_MLX_WEIGHTS_DIR/<hf_name>` → local `dist/<hf_name>` → HF hub
`mlx-community/Real-ESRGAN-<variant>` (downloaded + cached on first use).

## How it works

- **RRDBNet / SRVGGNetCompact** are translated 1:1 from `basicsr` (NHWC, MLX conv
  layout). The verified `(C,r,r)` pixel-shuffle ordering avoids the classic NHWC
  checkerboard bug.
- **Weights** are converted from the upstream `.pth` to fp16 `safetensors` (conv
  `(O,I,kH,kW)→(O,kH,kW,I)`, every tensor materialized). `realesr-general-x4v3` blends
  its `wdn` sibling at load time via `dni_weight` for the `-dn` denoise slider.
- **Tiling** ports the upstream `tile_process`: non-overlapping tiles with `tile_pad`
  context, valid-region placement. A single tile covering the whole image is
  **bit-exact** with untiled inference.

## Parity

| Variant | Full-forward vs PyTorch (CPU fp32) | fp16 vs golden |
|---|---|---|
| x4plus | 3.8e-6 | 1.3e-3 |
| x4plus_anime_6B | 3.0e-6 | 6.1e-4 |
| x2plus | 2.4e-6 | 5.5e-4 |
| animevideov3 | 3.9e-6 | 1.5e-3 |
| general-x4v3 | 4.1e-6 | 2.0e-3 |

Pipeline (MLX vs PyTorch `enhance`): 59–68 dB PSNR. See [`docs/REPORT.md`](docs/REPORT.md)
for M5 Max latency / memory benchmarks and [`docs/PREFLIGHT.md`](docs/PREFLIGHT.md) for
the full port plan.

## Development

```bash
pytest tests/smoke           # shapes + spatial-op invariants (no weights)
python scripts/gen_goldens.py        # regenerate parity oracle  (needs [parity])
python scripts/convert_to_mlx.py     # .pth -> fp16 dist/ artifacts
python scripts/benchmark.py          # M5 Max latency/memory -> docs/REPORT.md
python scripts/publish_hf.py         # push dist/ artifacts to mlx-community (network + HF auth)
pytest tests/parity          # full parity (needs weights/ + goldens/ + dist/)
```

## License

BSD-3-Clause, inherited from upstream Real-ESRGAN (Xintao Wang et al.). Weights are the
official Real-ESRGAN release checkpoints, converted to MLX fp16.
