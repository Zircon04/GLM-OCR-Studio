#!/bin/zsh
set -u

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_BIN="$WORK_DIR/.venv-mlx/bin/mlx_vlm.server"
MODEL="mlx-community/GLM-OCR-bf16"
PID_FILE="$WORK_DIR/glm-ocr-server.pid"
LOG_FILE="$WORK_DIR/logs/mlx-vlm.log"
PROFILE_FILE="$WORK_DIR/config/performance-profile.env"

server_seqs=8
profile_name="balanced"
if [[ -f "$PROFILE_FILE" ]]; then
  saved_profile="$(sed -n 's/^PROFILE=//p' "$PROFILE_FILE" | head -1)"
  saved_seqs="$(sed -n 's/^SERVER_SEQS=//p' "$PROFILE_FILE" | head -1)"
  if [[ "$saved_profile" == (auto|low|balanced|high|max|extreme|fast) ]]; then
    profile_name="$saved_profile"
  fi
  if [[ "$saved_seqs" == <1-32> ]]; then
    server_seqs="$saved_seqs"
  fi
fi

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/results"

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o command= | grep -Fq "$SERVER_BIN"; then
    echo "Локальный сервер GLM-OCR уже запущен (PID $pid)."
    echo "Адрес: http://127.0.0.1:8080"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Порт 8080 занят другим процессом. Он не будет остановлен."
  echo "Освободите порт 8080 либо остановите известный вам процесс и запустите сценарий снова."
  exit 1
fi

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Не найдено окружение MLX: $SERVER_BIN"
  echo "Переустановите GLM-OCR-local или проверьте папку установки."
  exit 1
fi

echo "Запускаю локальный GLM-OCR с ускорением Apple Metal…"
echo "Профиль нагрузки: $profile_name, параллельных запросов MLX: $server_seqs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nohup "$SERVER_BIN" --host 127.0.0.1 --port 8080 --model "$MODEL" --trust-remote-code --max-num-seqs "$server_seqs" --log-level INFO >> "$LOG_FILE" 2>&1 < /dev/null &!
pid=$!
echo "$pid" > "$PID_FILE"

for i in {1..45}; do
  if curl --silent --fail --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "Сервер готов: http://127.0.0.1:8080"
    echo "Журнал: $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Сервер завершился с ошибкой. Последние строки журнала:"
    tail -n 30 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 2
done

echo "Сервер ещё загружается. Проверьте журнал: $LOG_FILE"
echo "После первой загрузки модели это может занять несколько минут."
