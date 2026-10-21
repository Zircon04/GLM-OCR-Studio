#!/bin/bash
# GLM-OCR на NVIDIA без RTX. GTX 1060 6 ГБ — основной случай.
#
# vLLM/SGLang требуют compute capability 7.5+ (GeForce RTX 20 и новее).
# 1060 — это Pascal 6.1, официальный vLLM на ней не запускается.
# Здесь модель крутится через Ollama (llama.cpp): веса q8_0 около 1.6 ГБ,
# вместе с проектором и рабочим столом помещаются в 6 ГБ.
# Скорость ниже, чем на RTX или Apple Silicon: тензорных ядер нет.
# Скрипты windows/*.ps1 — тот же Ollama-путь, не «только RTX». На Windows они не проверялись.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/config"
MODEL_ALIAS="glm-ocr-pascal"
MODELFILE="$ROOT/config/ollama-pascal.Modelfile"
LOG_FILE="$ROOT/logs/ollama-pascal.log"
HEALTH_URL="http://127.0.0.1:11434/api/tags"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi не найден. Поставьте драйвер NVIDIA."
  echo "Без видеокарты Ollama всё равно может считать на CPU, но это будет медленно:"
  echo "  ollama pull glm-ocr:q8_0 && ollama serve"
  exit 1
fi

IFS=',' read -r GPU_NAME VRAM_MIB COMPUTE_CAP < <(
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -1
)
GPU_NAME="$(echo "$GPU_NAME" | sed 's/^ *//;s/ *$//')"
VRAM_MIB="$(echo "$VRAM_MIB" | tr -cd '0-9')"
COMPUTE_CAP="$(echo "$COMPUTE_CAP" | sed 's/^ *//;s/ *$//')"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -cd '0-9')"
echo "Карта: $GPU_NAME, compute $COMPUTE_CAP, всего ${VRAM_MIB} МиБ, свободно ${FREE_MIB} МиБ"

MAJOR="${COMPUTE_CAP%%.*}"
MINOR="${COMPUTE_CAP#*.}"
MINOR="${MINOR%%.*}"
# 7.5 и выше — Turing/RTX. Ниже vLLM ставить бессмысленно.
if (( MAJOR < 7 || (MAJOR == 7 && MINOR < 5) )); then
  echo "Compute $COMPUTE_CAP ниже 7.5: vLLM и SGLang эту карту не поддерживают. Берём Ollama."
else
  echo "Карта подходит и для vLLM, но этот сценарий всё равно поднимает Ollama:"
  echo "на одной карте так проще удержать память. Для vLLM смотрите README GLM-OCR."
fi

if (( VRAM_MIB <= 6144 || FREE_MIB < 4500 )); then
  TAG="glm-ocr:q8_0"
  echo "Мало видеопамяти — квант q8_0 (около 1.6 ГБ). F16 сюда лучше не класть."
else
  TAG="glm-ocr:latest"
  echo "Памяти хватает на F16 (glm-ocr:latest, около 2.7 ГБ с проектором)."
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama не установлена."
  echo "Linux:  curl -fsSL https://ollama.com/install.sh | sh"
  echo "Windows: https://ollama.com/download  и затем этот скрипт из WSL."
  exit 1
fi

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
if (( VRAM_MIB <= 8192 )); then
  # Меньше места под KV-кэш. На 0.9B это не качество текста, а запас по памяти.
  export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
fi

if ! curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Запускаю ollama serve…"
  nohup ollama serve >>"$LOG_FILE" 2>&1 &
  echo $! >"$ROOT/ollama-pascal.pid"
  for _ in $(seq 1 30); do
    if curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
else
  echo "Ollama уже слушает 11434. Параллельность этого процесса могла быть задана раньше."
  echo "Если распознавание упрётся в память, перезапустите serve с OLLAMA_NUM_PARALLEL=1."
fi

if ! curl --silent --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Ollama не поднялась. Журнал: $LOG_FILE"
  exit 1
fi

echo "Скачиваю $TAG …"
if ! ollama pull "$TAG"; then
  if [[ "$TAG" != "glm-ocr:latest" ]]; then
    echo "Тег $TAG недоступен, беру glm-ocr:latest. На 6 ГБ закройте лишние программы: F16 тяжелее."
    TAG="glm-ocr:latest"
    ollama pull "$TAG"
  else
    exit 1
  fi
fi

cat >"$MODELFILE" <<EOF
FROM $TAG
PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER num_predict 4096
EOF
ollama create "$MODEL_ALIAS" -f "$MODELFILE"

echo "Прогрев модели…"
if ! curl --silent --fail --max-time 180 "http://127.0.0.1:11434/api/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ALIAS\",\"prompt\":\"Text Recognition:\",\"stream\":false,\"options\":{\"num_predict\":8}}" \
  >/dev/null; then
  echo "Модель не ответила. Частая причина на 1060 — рабочий стол уже занял часть 6 ГБ."
  echo "Закройте браузер и игры и запустите скрипт снова. Журнал: $LOG_FILE"
  exit 1
fi

cat <<EOF

Сервер готов: http://127.0.0.1:11434  (модель $MODEL_ALIAS из $TAG).
Дальше, из корня репозитория:

  ./run-ocr-nvidia.sh /путь/к/скану.pdf

Ожидание по скорости: одна страница из нескольких блоков — десятки секунд,
не доли секунды. Это нормально для Pascal без тензорных ядер.
Первый запуск ещё скачает детектор вёрстки PP-DocLayoutV3 на CPU.
EOF
