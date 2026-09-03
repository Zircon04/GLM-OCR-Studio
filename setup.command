#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "      GLM-OCR Studio / Local (Apple Silicon Setup)"
echo "=========================================================="

# Check architecture
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Внимание: Данное приложение оптимизировано под Apple Silicon (M1/M2/M3/M4, arm64)."
  echo "Текущая архитектура: $ARCH"
fi

# Find Python 3.12 or 3.11/3.10
PY_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$ver" == "3.12" || "$ver" == "3.11" || "$ver" == "3.10" ]]; then
      PY_BIN="$(command -v "$candidate")"
      break
    fi
  fi
done

if [[ -z "$PY_BIN" ]]; then
  echo "Python 3.12/3.11 не найден."
  if command -v brew >/dev/null 2>&1; then
    echo "Устанавливаю python@3.12 через Homebrew..."
    brew install python@3.12
    PY_BIN="$(brew --prefix python@3.12)/bin/python3.12"
  else
    echo "Пожалуйста, установите Python 3.12 (через brew install python@3.12 или python.org) и запустите скрипт снова."
    exit 1
  fi
fi

echo "Используется Python: $PY_BIN ($("$PY_BIN" --version))"

# Create virtual environments if missing
echo "==> Проверка окружений Python..."

if [[ ! -d ".venv-mlx" ]]; then
  echo "Создание .venv-mlx..."
  "$PY_BIN" -m venv .venv-mlx
  ./.venv-mlx/bin/pip install --upgrade pip setuptools wheel
  ./.venv-mlx/bin/pip install -r requirements/mlx.txt
fi

if [[ ! -d ".venv-enhance" ]]; then
  echo "Создание .venv-enhance..."
  "$PY_BIN" -m venv .venv-enhance
  ./.venv-enhance/bin/pip install --upgrade pip setuptools wheel
  ./.venv-enhance/bin/pip install -r requirements/enhance.txt
fi

if [[ ! -d ".venv-sdk" ]]; then
  echo "Создание .venv-sdk..."
  "$PY_BIN" -m venv .venv-sdk
  ./.venv-sdk/bin/pip install --upgrade pip setuptools wheel
  ./.venv-sdk/bin/pip install -r requirements/sdk.txt
fi

# Compile Swift GUI App
echo "==> Сборка приложения macOS (GLM OCR Local.app)..."
if command -v swiftc >/dev/null 2>&1; then
  ./scripts/build_app.sh
else
  echo "Команда swiftc не найдена (не установлены Xcode Command Line Tools)."
  echo "Для сборки GUI выполните в терминале: xcode-select --install"
fi

mkdir -p logs results tmp

echo ""
echo "=========================================================="
echo "  Установка успешно завершена!"
echo "  Для запуска графического интерфейса откройте: GLM OCR Local.app"
echo "  Или запустите в терминале: ./start-glm-ocr.command"
echo "=========================================================="
