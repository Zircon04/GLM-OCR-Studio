# GLM-OCR Studio — окружение SDK для Windows.
#
# НЕ ПРОВЕРЯЛОСЬ НА WINDOWS. Скрипт написан по той же схеме, что Linux-версия,
# и на реальной Windows-машине не запускался.
#
# RTX (compute >= 7.5) автоматически уходит в WSL2 + vLLM на CUDA (windows\rtx-wsl.ps1).
# GTX 1060 и другие карты без RTX идут через Ollama (этот скрипт ниже).
#
# Запуск из корня репозитория, PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\windows\setup-sdk.ps1

$ErrorActionPreference = "Stop"

# RTX (compute >= 7.5): нативный CUDA-путь через WSL2 (windows\rtx-wsl.ps1).
# Ollama-путь для RTX: $env:GLMOCR_BACKEND = "ollama"
function Test-RtxCuda {
    if ($env:GLMOCR_BACKEND -eq "ollama") { return $false }
    if ($env:GLMOCR_BACKEND -eq "cuda") { return $true }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return $false }
    $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | Select-Object -First 1)
    $v = 0.0
    if (-not [double]::TryParse(($cap -as [string]).Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { return $false }
    return ($v -ge 7.5)
}
if (Test-RtxCuda) {
    & "$PSScriptRoot\rtx-wsl.ps1" setup @args
    exit $LASTEXITCODE
}
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
