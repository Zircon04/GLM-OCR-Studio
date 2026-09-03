#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

VERSION="2.5.1"
RELEASE_NAME="GLM-OCR-Studio-v${VERSION}-macOS-arm64"
DIST_DIR="$ROOT_DIR/dist"
ARCHIVE_PATH="$DIST_DIR/${RELEASE_NAME}.zip"

echo "==> Сборка приложения перед архивацией..."
./scripts/build_app.sh

echo "==> Создание каталога релиза..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

STAGE_DIR="$DIST_DIR/$RELEASE_NAME"
mkdir -p "$STAGE_DIR"

echo "==> Копирование файлов в архив..."
cp -R "GLM OCR Local.app" "$STAGE_DIR/"
cp -R config "$STAGE_DIR/"
cp -R enhancement "$STAGE_DIR/"
cp -R GLM-OCR "$STAGE_DIR/"
cp -R gui "$STAGE_DIR/"
cp -R manager "$STAGE_DIR/"
cp -R requirements "$STAGE_DIR/"
cp -R scripts "$STAGE_DIR/"
cp setup.command "$STAGE_DIR/"
cp start-glm-ocr.command "$STAGE_DIR/"
cp stop-glm-ocr.command "$STAGE_DIR/"
cp run-ocr.command "$STAGE_DIR/"
cp README.md "$STAGE_DIR/"
cp LICENSE "$STAGE_DIR/"
cp .gitmodules "$STAGE_DIR/"
cp .gitignore "$STAGE_DIR/"

# Clean up unwanted artifacts from staging
find "$STAGE_DIR" -name ".DS_Store" -delete
find "$STAGE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$STAGE_DIR" -name "*.pyc" -delete
find "$STAGE_DIR" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean up debug binaries in gui inside stage
setopt +o nomatch 2>/dev/null || true
rm -f "$STAGE_DIR"/gui/GLMOCRLocal.debug* "$STAGE_DIR/gui/GLMOCRLocal.fixed" "$STAGE_DIR/gui/GLMOCRLocal.new" 2>/dev/null || true
setopt -o nomatch 2>/dev/null || true

echo "==> Упаковка в zip: $ARCHIVE_PATH..."
cd "$DIST_DIR"
export COPYFILE_DISABLE=1
zip -r -q -y "${RELEASE_NAME}.zip" "$RELEASE_NAME" -x "*.DS_Store"
rm -rf "$RELEASE_NAME"

SIZE=$(du -sh "$ARCHIVE_PATH" | awk '{print $1}')
echo "=========================================================="
echo "  Архив релиза успешно создан!"
echo "  Путь: $ARCHIVE_PATH ($SIZE)"
echo "=========================================================="
