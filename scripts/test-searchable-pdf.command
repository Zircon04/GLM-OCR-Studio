#!/bin/bash
# Быстрая проверка searchable PDF. Двойной щелчок в Finder или:
#   ./scripts/test-searchable-pdf.command
cd "$(dirname "$0")/.." || exit 1
if [[ -x ".venv-sdk/bin/python" ]]; then
  PY=".venv-sdk/bin/python"
else
  PY="python3"
fi
exec "$PY" scripts/test_searchable_pdf.py "$@"
