# Распознавание на Windows через уже запущенную Ollama.
#
# НЕ ПРОВЕРЯЛОСЬ НА WINDOWS.
#   .\windows\run-ocr.ps1 C:\путь\к\файлу.pdf

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
    & "$PSScriptRoot\rtx-wsl.ps1" run @args
    exit $LASTEXITCODE
}
Write-Host "НЕ ПРОВЕРЯЛОСЬ НА WINDOWS."

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$InputPath = $args[0]
$Parent = if ($args.Count -ge 2 -and $args[1]) { $args[1] } else { Join-Path $Root "results" }
if (-not $InputPath -or -not (Test-Path -LiteralPath $InputPath)) {
    Write-Host "Использование: .\windows\run-ocr.ps1 C:\путь\к\файлу [C:\папка\результатов]"
    exit 1
}
if (-not (Test-Path -LiteralPath $Parent -PathType Container)) {
    Write-Host "Папки нет: $Parent"
    exit 1
}

$Python = "$Root\.venv-sdk\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Нет .venv-sdk. Один раз: .\windows\setup-sdk.ps1"
    exit 1
}

try {
    $probe = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3
    if ($probe.StatusCode -ne 200) { throw "status $($probe.StatusCode)" }
} catch {
    Write-Host "Ollama не отвечает. Сначала: .\windows\start-glm-ocr.ps1"
    exit 1
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Base = [IO.Path]::GetFileNameWithoutExtension($InputPath)
$Output = Join-Path $Parent "$Base-$Stamp"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$env:GLMOCR_CONFIG = "$Root\config\nvidia-pascal.yaml"
$env:GLMOCR_HEALTH_URL = "http://127.0.0.1:11434/api/tags"
$env:GLMOCR_PROFILE_FILE = "$Root\config\nvidia-pascal.env"
$env:GLMOCR_ALLOW_DOWNLOAD = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Результат будет в $Output"
& $Python "$Root\manager\job_runner.py" --input $InputPath --output $Output --enhance off --exports
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Готово: $Output"
