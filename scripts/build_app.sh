#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="GLM OCR Local.app"
echo "Сборка $APP_NAME..."

# Ensure build target arm64
swiftc -O -parse-as-library -target arm64-apple-macos14.0 gui/main.swift -o gui/GLMOCRLocal

mkdir -p "$APP_NAME/Contents/MacOS"
mkdir -p "$APP_NAME/Contents/Resources"

cp gui/Info.plist "$APP_NAME/Contents/"
cp gui/AppIcon.icns "$APP_NAME/Contents/Resources/"
cp gui/GLMOCRLocal "$APP_NAME/Contents/MacOS/"
chmod +x "$APP_NAME/Contents/MacOS/GLMOCRLocal"

# Sign ad-hoc
codesign --force --deep --sign - "$APP_NAME"

echo "Успешно собрано: $ROOT_DIR/$APP_NAME"
