# GLM-OCR Studio — окружение SDK для Windows.
#
# НЕ ПРОВЕРЯЛОСЬ НА WINDOWS. Скрипт написан по той же схеме, что Linux-версия,
# и на реальной Windows-машине не запускался.
#
# Это не сборка «только для RTX».
# vLLM на Windows штатно не ставится (нужен WSL и карта с compute capability 7.5+,
# то есть RTX 20 и новее). GTX 1060 и другие карты без RTX идут через Ollama.
# На RTX этот же скрипт тоже подходит: vLLM он не включает.
#
# Запуск из корня репозитория, PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\windows\setup-sdk.ps1

$ErrorActionPreference = "Stop"
Write-Host "НЕ ПРОВЕРЯЛОСЬ НА WINDOWS. Если скрипт упадёт, это ожидаемо до первой проверки."

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Get-Command py -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Host "Нужен Python 3.10-3.12 с лаунчером py. https://www.python.org/downloads/"
    exit 1
}

if (-not (Test-Path "$Root\.venv-sdk\Scripts\python.exe")) {
    & py -3.12 -m venv "$Root\.venv-sdk"
    if (-not (Test-Path "$Root\.venv-sdk\Scripts\python.exe")) {
        & py -3 -m venv "$Root\.venv-sdk"
    }
}

$VenvPython = "$Root\.venv-sdk\Scripts\python.exe"
$VenvPip = "$Root\.venv-sdk\Scripts\pip.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPip install -r "$Root\requirements\sdk.txt"
& $VenvPip install -e "$Root\GLM-OCR[layout]"
# Детектор вёрстки на CPU. CUDA-колёса свежего torch часто без Pascal (GTX 10xx).
& $VenvPip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

Write-Host "SDK готов: $Root\.venv-sdk"
Write-Host "Дальше: .\windows\start-glm-ocr.ps1"
