#!/bin/zsh
set -u

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_PYTHON="$WORK_DIR/.venv-sdk/bin/python"
JOB_RUNNER="$WORK_DIR/manager/job_runner.py"
RESULTS_DIR="$WORK_DIR/results"
PROFILE_FILE="$WORK_DIR/config/performance-profile.env"

echo "Локальный GLM-OCR (изображения, PDF или папка)"
echo "Файлы обрабатываются только на этом Mac через localhost/MLX."
printf "Введите путь к изображению, PDF или папке: "
IFS= read -r input_path

input_path="${input_path/#\~/$HOME}"
if [[ -z "$input_path" || ! -e "$input_path" ]]; then
  echo "Путь не существует. Ничего не обработано."
  exit 1
fi

server_port=8080
if [[ -f "$PROFILE_FILE" ]]; then
  saved_port="$(sed -n 's/^SERVER_PORT=//p' "$PROFILE_FILE" | head -1)"
  if [[ "$saved_port" =~ ^[0-9]+$ ]]; then
    server_port="$saved_port"
  fi
fi

if ! curl --silent --fail --max-time 2 http://127.0.0.1:$server_port/health >/dev/null 2>&1; then
  echo "Локальный сервер не отвечает на порту $server_port."
  echo "Сначала запустите: $WORK_DIR/start-glm-ocr.command"
  exit 1
fi

if [[ ! -x "$SDK_PYTHON" || ! -f "$JOB_RUNNER" ]]; then
  echo "Не найдены локальный SDK или менеджер заданий."
  exit 1
fi

base_name="${input_path:t}"
safe_name="${base_name:r}"
timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="$RESULTS_DIR/${safe_name}-${timestamp}"
mkdir -p "$output_dir"

echo "Распознавание запущено. Результат: $output_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$SDK_PYTHON" "$JOB_RUNNER" --input "$input_path" --output "$output_dir" --enhance hybrid --exports
exit_code=$?
if [[ $exit_code -eq 0 ]]; then
  echo "Готово. Markdown, JSON, HTML, DOCX, searchable PDF и ZIP сохранены в: $output_dir"
  echo "Если выполнение прервётся, снова запустите тот же путь через приложение: оно продолжит с сохранённого блока."
else
  echo "Распознавание завершилось с ошибкой (код $exit_code). Проверьте журнал сервера: $WORK_DIR/logs/mlx-vlm.log"
fi
exit $exit_code
