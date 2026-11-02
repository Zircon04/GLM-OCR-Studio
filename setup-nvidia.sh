#!/bin/bash
# Окружения для NVIDIA на Linux или в WSL2. НЕ ПРОВЕРЯЛОСЬ НА РЕАЛЬНОЙ КАРТЕ.
#
# Бэкенд выбирается по compute capability (scripts/nvidia-backend.sh):
#   RTX 20 и новее (>= 7.5) -> cuda:   .venv-sdk с CUDA-torch (детектор вёрстки на GPU)
#                                     + .venv-vllm с vLLM (модель на CUDA/тензорных ядрах)
#   GTX 10xx/Pascal (< 7.5) -> ollama: .venv-sdk с CPU-torch, модель в Ollama
# Переопределение: GLMOCR_BACKEND=cuda|ollama ./setup-nvidia.sh
# Индекс колёс CUDA: TORCH_CUDA_INDEX (по умолчанию cu128, нужен драйвер >= 570).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY_BIN="${PY_BIN:-python3}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Нужен Python 3.10–3.12. На Ubuntu: sudo apt install python3 python3-venv"
  exit 1
fi

# shellcheck source=scripts/nvidia-backend.sh
source "$ROOT/scripts/nvidia-backend.sh"
echo "Карта: ${GPU_NAME:-не найдена}, compute $COMPUTE_CAP, VRAM ${VRAM_MIB} МиБ -> бэкенд: $GLMOCR_BACKEND"

if [[ ! -d .venv-sdk ]]; then
  "$PY_BIN" -m venv .venv-sdk
fi
.venv-sdk/bin/python -m pip install --upgrade pip setuptools wheel
.venv-sdk/bin/pip install -r requirements/sdk.txt
.venv-sdk/bin/pip install "glmocr[layout] @ file://${ROOT}/GLM-OCR"

if [[ "$GLMOCR_BACKEND" == cuda ]]; then
  # CUDA-сборка PyTorch: PP-DocLayoutV3 считается на той же карте.
  .venv-sdk/bin/pip install --upgrade --force-reinstall torch torchvision --index-url "$TORCH_CUDA_INDEX"
  .venv-sdk/bin/python -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())" || true

  # vLLM — отдельное окружение: у него свои закреплённые версии torch/transformers.
  if [[ ! -d .venv-vllm ]]; then
    "$PY_BIN" -m venv .venv-vllm
  fi
  .venv-vllm/bin/python -m pip install --upgrade pip setuptools wheel
  # Версии из GLM-OCR/README.md: vllm>=0.19.0, transformers>=5.3.0.
  .venv-vllm/bin/pip install -U "vllm>=0.19.0"
  .venv-vllm/bin/pip install -U "transformers>=5.3.0"
  echo "vLLM готов: $ROOT/.venv-vllm"
else
  # CPU-сборка PyTorch. CUDA-колёса свежего torch часто собраны без Pascal (sm_61),
  # а детектору видеокарта всё равно не нужна: её занимает Ollama.
  .venv-sdk/bin/pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo "$GLMOCR_BACKEND" > "$ROOT/config/nvidia-backend"
echo "SDK готов: $ROOT/.venv-sdk (бэкенд $GLMOCR_BACKEND)"
echo "Дальше: ./start-glm-ocr-nvidia.sh"
