#!/usr/bin/env python3
"""
Multi-GPU Launcher for GLM-OCR

Automatically launches sglang/vLLM services across multiple GPUs, distributes
files evenly, and runs the GLM-OCR pipeline in parallel for maximum throughput.

Each GPU hosts both a sglang/vLLM inference server and a layout detection model,
forming a self-contained processing unit with zero cross-GPU communication.

Usage:
    python examples/multi-gpu-deploy/launch.py -i ./images -o ./output
    python examples/multi-gpu-deploy/launch.py -i ./docs -o ./results --engine vllm --gpus 0,1,2,3
    python examples/multi-gpu-deploy/launch.py -i ./pdfs -o ./out --engine-args "--mem-fraction-static 0.85"
"""

import sys
import argparse
from pathlib import Path

from gpu_utils import DEFAULT_BASE_PORT, DEFAULT_MIN_FREE_MB, _print_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU launcher for GLM-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/multi-gpu-deploy/launch.py -i ./images -o ./output
  python examples/multi-gpu-deploy/launch.py -i ./docs -o ./results --engine vllm --gpus 0,1,2,3
  python examples/multi-gpu-deploy/launch.py -i ./pdfs -o ./out --engine-args "--mem-fraction-static 0.85"
  python examples/multi-gpu-deploy/launch.py -i ./imgs -o ./out --min-free-mb 20000 --timeout 900
        """,
    )

    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--gpu-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--filelist", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--progress-file", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-root", type=str, default=None, help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--input", "-i", type=str, help="Input image file or directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="zai-org/GLM-OCR",
        help="Model name or path (default: zai-org/GLM-OCR)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="sglang",
        choices=["sglang", "vllm"],
        help="Inference engine (default: sglang)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPU IDs, comma-separated, or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_BASE_PORT,
        help=f"Base port for engine services (default: {DEFAULT_BASE_PORT})",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=DEFAULT_MIN_FREE_MB,
        help=f"Min free GPU memory in MB (default: {DEFAULT_MIN_FREE_MB})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Engine startup timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--engine-args",
        type=str,
        default=None,
        help='Extra args for engine '
        '(e.g. "--mem-fraction-static 0.85")',
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to glmocr config YAML",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for workers (default: WARNING)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write any result files",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        from worker import run_worker

        run_worker(args)
    else:
        if not args.input:
            _print_err("Error: --input/-i is required")
            sys.exit(1)
        Path(args.output).mkdir(parents=True, exist_ok=True)

        from coordinator import Coordinator

        coordinator = Coordinator(args)
        coordinator.run()


if __name__ == "__main__":
    main()
