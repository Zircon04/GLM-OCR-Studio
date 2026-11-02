#!/bin/bash
# Распознавание через уже запущенный сервер (start-glm-ocr-nvidia.sh):
# vLLM на CUDA для RTX или Ollama для Pascal. НЕ ПРОВЕРЯЛОСЬ НА РЕАЛЬНОЙ КАРТЕ.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
INPUT="${1:-}"
PARENT="${2:-$ROOT/results}"
if [[ -z "$INPUT" || ! -e "$INPUT" ]]; then
  echo "Использование: ./run-ocr-nvidia.sh /путь/к/файлу [/папка/результатов]"
  exit 1
fi
if [[ ! -d "$PARENT" ]]; then
  echo "Папки нет: $PARENT"
  exit 1
fi

if [[ ! -x "$ROOT/.venv-sdk/bin/glmocr" ]]; then
  echo "Нет .venv-sdk. Один раз выполните: ./setup-nvidia.sh"
  exit 1
fi

if [[ -z "${GLMOCR_BACKEND:-}" && -f "$ROOT/config/nvidia-backend" ]]; then
  GLMOCR_BACKEND="$(tr -cd 'a-z' < "$ROOT/config/nvidia-backend")"
fi
# shellcheck source=scripts/nvidia-backend.sh
source "$ROOT/scripts/nvidia-backend.sh"
if [[ "$GLMOCR_BACKEND" == cuda ]]; then
  HEALTH="http://127.0.0.1:${GLMOCR_VLLM_PORT:-8080}/health"
  CONFIG_NAME="nvidia-rtx"
else
  HEALTH="http://127.0.0.1:11434/api/tags"
  CONFIG_NAME="nvidia-pascal"
fi
if ! curl --silent --fail --max-time 3 "$HEALTH" >/dev/null 2>&1; then
  echo "Сервер модели ($GLMOCR_BACKEND) не отвечает. Сначала: ./start-glm-ocr-nvidia.sh"
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="$(basename "${INPUT%.*}")"
OUTPUT="$PARENT/${BASE}-${STAMP}"
mkdir -p "$OUTPUT"

export GLMOCR_CONFIG="$ROOT/config/${CONFIG_NAME}.yaml"
export GLMOCR_HEALTH_URL="$HEALTH"
export GLMOCR_PROFILE_FILE="$ROOT/config/${CONFIG_NAME}.env"
export GLMOCR_ALLOW_DOWNLOAD=1
export PYTHONUNBUFFERED=1

echo "Бэкенд: $GLMOCR_BACKEND, конфиг: $GLMOCR_CONFIG"
echo "Результат будет в $OUTPUT"
"$ROOT/.venv-sdk/bin/python" "$ROOT/manager/job_runner.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --enhance off \
  --exports
echo "Готово: $OUTPUT"
