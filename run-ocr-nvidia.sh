#!/bin/bash
# Распознавание через уже запущенный Ollama (start-glm-ocr-nvidia.sh).
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

if ! curl --silent --fail --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama не отвечает. Сначала: ./start-glm-ocr-nvidia.sh"
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="$(basename "${INPUT%.*}")"
OUTPUT="$PARENT/${BASE}-${STAMP}"
mkdir -p "$OUTPUT"

export GLMOCR_CONFIG="$ROOT/config/nvidia-pascal.yaml"
export GLMOCR_HEALTH_URL="http://127.0.0.1:11434/api/tags"
export GLMOCR_PROFILE_FILE="$ROOT/config/nvidia-pascal.env"
export GLMOCR_ALLOW_DOWNLOAD=1
export PYTHONUNBUFFERED=1

echo "Результат будет в $OUTPUT"
"$ROOT/.venv-sdk/bin/python" "$ROOT/manager/job_runner.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --enhance off \
  --exports
echo "Готово: $OUTPUT"
