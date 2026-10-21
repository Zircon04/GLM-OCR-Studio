#!/bin/bash
# Окружение SDK без MLX: Linux или WSL, карта вроде GTX 1060.
# Детектор вёрстки ставится на CPU, чтобы не занимать 6 ГБ видеокарты.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY_BIN="${PY_BIN:-python3}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Нужен Python 3.10–3.12. На Ubuntu: sudo apt install python3 python3-venv"
  exit 1
fi

if [[ ! -d .venv-sdk ]]; then
  "$PY_BIN" -m venv .venv-sdk
fi
# shellcheck disable=SC1091
source .venv-sdk/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/sdk.txt
pip install "glmocr[layout] @ file://${ROOT}/GLM-OCR"
# CPU-сборка PyTorch. CUDA-колёса свежего torch часто собраны без Pascal (sm_61),
# а детектору видеокарта всё равно не нужна: её занимает Ollama.
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "SDK готов: $ROOT/.venv-sdk"
echo "Дальше: ./start-glm-ocr-nvidia.sh"
