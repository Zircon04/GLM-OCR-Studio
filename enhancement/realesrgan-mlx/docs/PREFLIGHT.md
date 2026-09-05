# Real-ESRGAN → MLX — PREFLIGHT

**Working dir:** `DEV_INT/realesrgan-mlx` · **Remote (planned):** `xocialize/realesrgan-mlx`
**Publish (planned):** `mlx-community/Real-ESRGAN-{variant}` (+ collection)
**Upstream:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) @ master · **License:** BSD-3-Clause ✅
**Started:** 2026-06-05 · **Tier:** 2 (single-stack CNN; no mlx-forge needed)

---

## CONFIRM gates (mlx-porting)

1. **License** ✅ BSD-3-Clause — code *and* release weights. Commercial-OK. No RAIL/NC strings.
2. **Port status** — Only prior art is [`themindstudio/RealESRGAN-x4plus-mlx`](https://huggingface.co/themindstudio/RealESRGAN-x4plus-mlx): a **code-less 67 MB `.npz` blob**, single x4plus variant, no Swift loader, no `config.json`, no tiling, ~6 months stale. **No `mlx-community` port; no MLX-Swift port.** → net-new clean port, wide-open lane.
3. **Config truth** — pinned in `realesrgan_mlx/config.py` from upstream `inference_realesrgan.py`, cross-checked vs `basicsr/archs/{rrdbnet,srvgg}_arch.py`. Defaults match the *trained* checkpoints.
4. **Tier** — Tier 2. Pure CNN: conv2d / leaky-relu / prelu / pixel-shuffle / nearest-interp. No attention, scheduler, or diffusion. Weight conversion = key-rename + conv transpose.

## Scope (this round)

- **Python MLX port, all 5 variants.** Swift = deferred phase 8.
- Two architectures are the only real work; each variant is a config over one of them.

| Variant | Arch | netscale | key config |
|---|---|---|---|
| RealESRGAN_x4plus | RRDBNet | 4 | num_block=23 |
| RealESRGAN_x4plus_anime_6B | RRDBNet | 4 | num_block=6 |
| RealESRGAN_x2plus | RRDBNet | 2 | num_block=23 (pixel_unshuffle x2 at input) |
| realesr-animevideov3 | SRVGGNetCompact | 4 | num_conv=16, prelu |
| realesr-general-x4v3 | SRVGGNetCompact | 4 | num_conv=32, prelu, + wdn denoise blend |

## Architecture notes

**RRDBNet** (`archs/rrdbnet_arch.py`): conv_first → `num_block`× RRDB (each = 3× ResidualDenseBlock, ×0.2 residual; RDB = 5 dense convs + leaky-relu(0.2), ×0.2 residual) → conv_body + global residual → two **fixed nearest-×2** upsamples (conv_up1/up2) → conv_hr → conv_last. The x2 variant feeds `pixel_unshuffle(x, 2)` first so the two ×2 upsamples net ×2; the x1 variant uses unshuffle ×4.

**SRVGGNetCompact** (`archs/srvgg_arch.py`): a flat `body` list interleaving conv + activation, ending in conv→`out*upscale²`, then **PixelShuffle** + a **nearest-resized input residual**. PReLU is `num_parameters=num_feat` (per-channel), each activation its own instance.

## Numerics / MLX traps (from skill + Forge ecosystem)

- **Conv layout** PyTorch `(O,I,kH,kW)` → MLX `(O,kH,kW,I)`; **NHWC** data everywhere.
- **Pixel-shuffle ordering = pitfall #7 ("the recurring one").** NHWC must split channels `(C,r,r)` to match `torch.nn.PixelShuffle`. `(r,r,C)` gives right shapes / wrong content (showed up as max_abs ~2.27 in the Forge EfRLFN port). We reuse the verified helper — see `utils/spatial.py`, mirrored from `ForgeTraining/Python/models/efrlfn_mlx.py::_pixel_shuffle_nhwc`.
- **Parity on `mx.cpu` fp32** — Apple-GPU conv accumulation is tf32-like; gate correctness on CPU.
- **Materialize before save** — `mx.eval` every tensor before `mx.save_safetensors` (lazy → silent zeros).
- **Checkerboard smoke** — decode random noise through the full upsampler before shipping.

## Forge ecosystem context (ForgeOptimizer recon, 2026-06-05)

- **ADR-0007** chose vendored Real-ESRGAN **CoreML** (`realesrgan_x{2,4}.mlpackage`) for the `ForgeUpscaler.ExportTier`, explicitly *rejecting* the themindstudio MLX blob (no Swift loader). It lists a **revisit trigger**: *"a new MLX-Swift Real-ESRGAN port lands … weights ≤100 MB and a Swift example."*
- **This port is that trigger.** Producing `mlx-community/Real-ESRGAN-*` (safetensors + config + an MLX-Swift conformer) enables swapping `RealESRGAN_CoreML` → a native `RealESRGAN_MLX` behind the existing `ExportTier` protocol — "one new conformer away." All fp16 variants are well under 100 MB.
- **SRVGGNetCompact is the legacy Forge playback model that EfRLFN (Phase C.2, MIT) replaced.** The EfRLFN MLX-Python/Swift pair is the direct structural template for our SRVGGNetCompact half (shared pixel-shuffle + nearest-residual shape). Converter template: `ForgeTraining/Scripts/convert_efrlfn_to_mlx.py`. Swift load pattern: `MLX.loadArrays` → `ModuleParameters.unflattened` → `Module.update(verify:.noUnusedKeys)`.
- **Phase 8 Swift target** = conform to `ForgeUpscaler.ExportTier` (`name`, `scaleFactor`, `inputTileSize`, `tileOverlap`, `upscale(_:)`), flexible input shape (unlike the fixed 128×128 CoreML package → can use the plan-spec 256/32 tiling).

## Plan / phases

- **P0 — Scaffold** ✅ (this commit): repo, config registry, isomorphic arch translations, spatial helpers, smoke tests, docs.
- **P1 — PyTorch oracle**: `[parity]` venv, download 5 `.pth`, CPU fp32 goldens (per-layer + full-forward) → `goldens/<variant>.npz`.
- **P2 — RRDBNet parity → Gate A**: x4plus full-forward max_abs < 1e-2 on `mx.cpu`; then anime_6B + x2plus.
- **P3 — SRVGGNetCompact parity → Gate B**: animevideov3 + general-x4v3.
- **P4 — Weight conversion**: `.pth` → fp16 safetensors + config.json; `mx.eval` all; dni blend for general. `build_model()` load path + env override.
- **P5 — Pipeline + tiling → Gates C/D**: port `RealESRGANer` (pre_pad/mod_pad, `tile_process`, alpha/gray, outscale) + CLI. Gate C: tiled == untiled. Gate D: e2e PSNR vs PyTorch RealESRGANer (>50 dB on cpu).
- **P6 — Quantize (optional)**: fp16 ships; int8 sample, gate on per-pass cosine.
- **P7 — Publish**: `mlx-community/Real-ESRGAN-*` + GitHub + README + M5 Max benchmarks (fp16, tile vs no-tile, vs themindstudio).
- **P8 — DEFERRED: MLX-Swift `ExportTier` conformer** for ForgeUpscaler (this port = the parity oracle).

## Open decisions

- **P4 weight dtype**: lean **fp16** published (matches upstream half-precision default; all variants < 100 MB for the ADR-0005 lazy-download pattern) + an fp32 option for the parity oracle. Confirm at P4.
