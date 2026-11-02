#!/bin/bash
# Остановка сервера, поднятого start-glm-ocr-nvidia.sh (vLLM или Ollama). НЕ ПРОВЕРЯЛОСЬ НА КАРТЕ.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
stopped=0
for pid_file in "$ROOT/logs/vllm-rtx.pid" "$ROOT/logs/ollama-pascal.pid"; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    echo "Остановлен процесс $pid ($(basename "$pid_file" .pid))"
    stopped=1
  fi
  rm -f "$pid_file"
done
(( stopped )) || echo "Запущенных этим проектом серверов не найдено."
