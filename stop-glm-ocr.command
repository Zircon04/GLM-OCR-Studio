#!/bin/zsh
set -u

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_BIN="$WORK_DIR/.venv-mlx/bin/mlx_vlm.server"
PID_FILE="$WORK_DIR/glm-ocr-server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Файл PID не найден: сервер, запущенный этим набором сценариев, не обнаружен."
  exit 0
fi

pid="$(<"$PID_FILE")"
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Сервер уже остановлен."
  exit 0
fi

if ! ps -p "$pid" -o command= | grep -Fq "$SERVER_BIN"; then
  echo "PID $pid не принадлежит серверу GLM-OCR данного набора. Ничего не остановлено."
  exit 1
fi

kill "$pid"
# mlx-vlm завершает уже принятый запрос перед остановкой. Тяжёлый фрагмент
# может вычисляться несколько минут, поэтому ждём его корректного завершения.
for i in {1..90}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Локальный сервер GLM-OCR остановлен."
    exit 0
  fi
  sleep 2
done

echo "Сервер получил запрос на остановку и ещё завершает работу."
exit 1
