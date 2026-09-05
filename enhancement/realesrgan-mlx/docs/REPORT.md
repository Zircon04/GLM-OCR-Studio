# Real-ESRGAN-MLX — Benchmarks (M5 Max, fp16, GPU)

Wall-clock per `enhance()` (incl. pre/post), mean of 3 runs.


## Per-variant — 256×256 input, no tiling

| Variant | Arch | Scale | Output | Latency | Peak mem |
|---|---|---|---|---|---|
| RealESRGAN_x4plus | RRDBNet | ×4 | 1024×1024 | 1364 ms | 4631 MB |
| RealESRGAN_x4plus_anime_6B | RRDBNet | ×4 | 1024×1024 | 415 ms | 4604 MB |
| RealESRGAN_x2plus | RRDBNet | ×2 | 512×512 | 471 ms | 1524 MB |
| realesr-animevideov3 | SRVGGNetCompact | ×4 | 1024×1024 | 71 ms | 970 MB |
| realesr-general-x4v3 | SRVGGNetCompact | ×4 | 1024×1024 | 118 ms | 1843 MB |

## Tiling — x4plus on a large input (memory story)

| Input | Mode | Output | Latency | Peak mem |
|---|---|---|---|---|
| 1024×1024 | no-tile | 4096×4096 | 17074 ms | 15912 MB |
| 1024×1024 | tile=256 | 4096×4096 | 25962 ms | 5341 MB |
| 1024×1024 | tile=128 | 4096×4096 | 39453 ms | 2084 MB |
