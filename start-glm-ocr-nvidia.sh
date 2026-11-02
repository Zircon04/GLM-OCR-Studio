#!/bin/bash
# Запуск сервера модели GLM-OCR на NVIDIA (Linux / WSL2). НЕ ПРОВЕРЯЛОСЬ НА РЕАЛЬНОЙ КАРТЕ.
#
#   RTX (compute >= 7.5): vllm serve zai-org/GLM-OCR на CUDA, порт 8080, MTP-спекуляция.
#   Pascal/GTX 10xx:      Ollama (scripts/start-ollama-nvidia.sh), порт 11434.
# Переопределение: GLMOCR_BACKEND=cuda|ollama. Остановка: ./stop-glm-ocr-nvidia.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi не найден. Поставьте драйвер NVIDIA (в WSL2 — драйвер Windows с поддержкой WSL)."
  exit 1
fi
if [[ -z "${GLMOCR_BACKEND:-}" && -f "$ROOT/config/nvidia-backend" ]]; then
  GLMOCR_BACKEND="$(tr -cd 'a-z' < "$ROOT/config/nvidia-backend")"
fi
# shellcheck source=scripts/nvidia-backend.sh
source "$ROOT/scripts/nvidia-backend.sh"
echo "Карта: $GPU_NAME, compute $COMPUTE_CAP, всего ${VRAM_MIB} МиБ, свободно ${FREE_MIB} МиБ -> $GLMOCR_BACKEND"

if [[ "$GLMOCR_BACKEND" == ollama ]]; then
  exec "$ROOT/scripts/start-ollama-nvidia.sh"
fi

VLLM="$ROOT/.venv-vllm/bin/vllm"
PORT="${GLMOCR_VLLM_PORT:-8080}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
LOG_FILE="$ROOT/logs/vllm-rtx.log"
PID_FILE="$ROOT/logs/vllm-rtx.pid"
if [[ ! -x "$VLLM" ]]; then
  echo "Нет .venv-vllm. Один раз выполните: ./setup-nvidia.sh"
  exit 1
fi

if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "vLLM уже отвечает на $PORT."
  exit 0
fi

# Доля VRAM для vLLM (веса 0.9B ~2 ГБ + KV-кэш). Остаток — детектору вёрстки
# (~1–1.5 ГБ на CUDA) и рабочему столу. Переопределение: GLMOCR_GPU_UTIL=0.xx
if [[ -n "${GLMOCR_GPU_UTIL:-}" ]]; then
  UTIL="$GLMOCR_GPU_UTIL"
elif (( VRAM_MIB <= 8192 )); then
  UTIL=0.60
elif (( VRAM_MIB <= 12288 )); then
  UTIL=0.70
elif (( VRAM_MIB <= 16384 )); then
  UTIL=0.75
else
  UTIL=0.80
fi
MAX_LEN="${GLMOCR_MAX_MODEL_LEN:-16384}"
if (( VRAM_MIB <= 8192 )) && [[ -z "${GLMOCR_MAX_MODEL_LEN:-}" ]]; then
  MAX_LEN=8192
fi
MAX_SEQS="${GLMOCR_MAX_NUM_SEQS:-16}"

ARGS=(serve zai-org/GLM-OCR
  --served-model-name glm-ocr
  --host 127.0.0.1 --port "$PORT"
  --gpu-memory-utilization "$UTIL"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --dtype auto)
# MTP-спекуляция из GLM-OCR/README.md. Отключить: GLMOCR_VLLM_MTP=0
if [[ "${GLMOCR_VLLM_MTP:-1}" != 0 ]]; then
  ARGS+=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
fi

echo "Запускаю vLLM: util=$UTIL, max-model-len=$MAX_LEN, max-num-seqs=$MAX_SEQS. Журнал: $LOG_FILE"
nohup "$VLLM" "${ARGS[@]}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

# Первый старт скачивает веса (~2 ГБ) и компилирует CUDA-графы: до 10 минут.
for _ in $(seq 1 600); do
  if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vLLM завершился при старте. Последние строки журнала:"
    tail -n 30 "$LOG_FILE"
    echo "Если не хватило памяти: GLMOCR_GPU_UTIL=0.5 GLMOCR_MAX_MODEL_LEN=8192 ./start-glm-ocr-nvidia.sh"
    echo "Запасной путь: GLMOCR_BACKEND=ollama ./start-glm-ocr-nvidia.sh"
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "vLLM не ответил за 10 минут. Журнал: $LOG_FILE"
  exit 1
fi

cat <<EOT

Сервер готов: http://127.0.0.1:${PORT}  (vLLM, модель glm-ocr на CUDA).
Дальше: ./run-ocr-nvidia.sh /путь/к/скану.pdf
Остановить: ./stop-glm-ocr-nvidia.sh
EOT
