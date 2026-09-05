# Multi-GPU Deployment for GLM-OCR

Automatically launch sglang/vLLM inference services across multiple GPUs, distribute image files evenly, and run the GLM-OCR pipeline in parallel for maximum throughput.

Each GPU hosts both an inference server (sglang or vLLM) and a layout detection model, forming a self-contained processing unit with zero cross-GPU communication.

## Features

- **Auto GPU detection** — discovers all available GPUs and filters by free VRAM
- **Dynamic port allocation** — automatically skips occupied ports
- **Fault tolerance** — failed GPUs are skipped, files are redistributed to healthy GPUs
- **Global progress bar** — real-time `tqdm` progress across all GPUs
- **Graceful shutdown** — `Ctrl+C` cleanly terminates all subprocesses; double `Ctrl+C` force-kills
- **Centralized logging** — all engine/worker logs saved under `logs/<timestamp>/`
- **Speculative decoding** — MTP enabled by default for both sglang and vLLM

## Quick Start

```bash
# Use all available GPUs with sglang (default)
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output -m /path/to/GLM-OCR

# Specify GPUs and use vLLM
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output --engine vllm --gpus 0,1,2,3

# Custom model path and VRAM threshold
python examples/multi-gpu-deploy/launch.py -i ./images -o ./output -m /path/to/GLM-OCR --min-free-mb 20000
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `-i`, `--input` | *required* | Input image file or directory (recursive) |
| `-o`, `--output` | `./output` | Output directory for results |
| `-m`, `--model` | `zai-org/GLM-OCR` | Model name or local path |
| `--engine` | `sglang` | Inference engine: `sglang` or `vllm` |
| `--gpus` | `auto` | GPU IDs (comma-separated) or `auto` for all available |
| `--base-port` | `8080` | Base port for engine services |
| `--min-free-mb` | `16000` | Minimum free GPU memory in MB to use a GPU |
| `--timeout` | `600` | Engine startup timeout in seconds |
| `--engine-args` | *none* | Extra arguments passed to the engine |
| `-c`, `--config` | *none* | Path to a custom glmocr config YAML |
| `--log-level` | `WARNING` | Log level for worker processes |


## Examples

### Basic usage

```bash
python examples/multi-gpu-deploy/launch.py -i /data/documents -o /data/results
```

### Use vLLM with specific GPUs

```bash
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --engine vllm \
  --gpus 0,2,4,6
```

### Custom engine arguments

```bash
# sglang with custom memory fraction
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --engine-args "--mem-fraction-static 0.85"
```

### Custom config YAML

```bash
python examples/multi-gpu-deploy/launch.py \
  -i /data/documents \
  -o /data/results \
  --config my_config.yaml
```

## Logs

All logs are saved under `logs/<timestamp>/`:

| File | Content |
|---|---|
| `main.log` | Coordinator stdout/stderr |
| `engine_gpu<N>_port<P>.log` | Engine service output for each GPU |
| `worker_gpu<N>.log` | Worker process output for each GPU |
| `failed_files.json` | Aggregated list of failed files (if any) |

## Troubleshooting

**Q: Some ports are occupied, will it still work?**

Yes. The launcher automatically scans for available ports starting from `--base-port` and skips any that are in use.

**Q: A GPU runs out of memory mid-processing. What happens?**

The worker on that GPU will fail, but other GPUs continue processing. Failed files are logged in `failed_files.json` for later re-processing.

**Q: How do I re-run only the failed files?**

Copy the failed files to a directory and run the launcher again pointing to that directory.

## File Structure

```
examples/multi-gpu-deploy/
├── launch.py        # Entry point and CLI argument parser
├── coordinator.py   # Orchestration: GPU detection, engine/worker lifecycle
├── engine.py        # Engine service management and progress tracking
├── worker.py        # Worker process: GLM-OCR pipeline execution
├── gpu_utils.py     # GPU detection, port checking, file sharding
├── README.md        # This file (English)
└── README_zh.md     # Chinese documentation
```
