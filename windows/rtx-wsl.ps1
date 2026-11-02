# GLM-OCR на Windows + RTX: нативный CUDA-путь через WSL2.
#
# НЕ ПРОВЕРЯЛОСЬ НА WINDOWS И НА РЕАЛЬНОЙ КАРТЕ.
#
# vLLM официально не поддерживает обычный Windows, поэтому модель считается
# в Linux внутри WSL2 на той же видеокарте (CUDA проброшен драйвером Windows).
# Скрипт вызывает Linux-скрипты репозитория внутри WSL:
#   setup -> ./setup-nvidia.sh, start -> ./start-glm-ocr-nvidia.sh,
#   run   -> ./run-ocr-nvidia.sh, stop -> ./stop-glm-ocr-nvidia.sh
#
# Один раз: wsl --install -d Ubuntu (затем перезагрузка), свежий драйвер NVIDIA
# для Windows. Внутри Ubuntu: sudo apt install python3 python3-venv curl
#
#   .\windows\rtx-wsl.ps1 setup
#   .\windows\rtx-wsl.ps1 start
#   .\windows\rtx-wsl.ps1 run C:\путь\к\скану.pdf [C:\папка\результатов]
#   .\windows\rtx-wsl.ps1 stop
# Дистрибутив: $env:GLMOCR_WSL_DISTRO (по умолчанию — дистрибутив WSL по умолчанию).

param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("setup", "start", "run", "stop")][string]$Action,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
Write-Host "НЕ ПРОВЕРЯЛОСЬ НА WINDOWS. RTX-путь: vLLM на CUDA внутри WSL2."

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "WSL не найден. Установите: wsl --install -d Ubuntu  (PowerShell от администратора), затем перезагрузка."
    exit 1
}
$Root = Split-Path -Parent $PSScriptRoot
$WslBase = @()
if ($env:GLMOCR_WSL_DISTRO) { $WslBase = @("-d", $env:GLMOCR_WSL_DISTRO) }

function Convert-ToWslPath([string]$WinPath) {
    $full = [IO.Path]::GetFullPath($WinPath)
    $out = & wsl.exe @WslBase -- wslpath -a -u "$full"
    if ($LASTEXITCODE -ne 0 -or -not $out) { throw "wslpath не смог преобразовать путь: $full" }
    return ($out | Select-Object -First 1).Trim()
}
function Quote-Bash([string]$s) { return "'" + ($s -replace "'", "'\''") + "'" }

& wsl.exe @WslBase -- nvidia-smi -L
if ($LASTEXITCODE -ne 0) {
    Write-Host "Внутри WSL не видна видеокарта. Обновите драйвер NVIDIA для Windows (с поддержкой WSL) и выполните: wsl --update"
    exit 1
}

# Репозиторий на диске Windows (/mnt/c/...) работает, но venv и модели там медленнее.
# Для скорости можно склонировать проект прямо в WSL и задать $env:GLMOCR_WSL_ROOT.
$RepoWsl = if ($env:GLMOCR_WSL_ROOT) { $env:GLMOCR_WSL_ROOT } else { Convert-ToWslPath $Root }

switch ($Action) {
    "setup" { $cmd = "./setup-nvidia.sh" }
    "start" { $cmd = "./start-glm-ocr-nvidia.sh" }
    "stop"  { $cmd = "./stop-glm-ocr-nvidia.sh" }
    "run" {
        if (-not $Rest -or $Rest.Count -lt 1 -or -not (Test-Path -LiteralPath $Rest[0])) {
            Write-Host "Использование: .\windows\rtx-wsl.ps1 run C:\путь\к\файлу [C:\папка\результатов]"
            exit 1
        }
        $cmd = "./run-ocr-nvidia.sh " + (Quote-Bash (Convert-ToWslPath $Rest[0]))
        if ($Rest.Count -ge 2 -and $Rest[1]) { $cmd += " " + (Quote-Bash (Convert-ToWslPath $Rest[1])) }
    }
}
$backend = if ($env:GLMOCR_BACKEND) { $env:GLMOCR_BACKEND } else { "cuda" }
$script = "cd " + (Quote-Bash $RepoWsl) + " && GLMOCR_BACKEND=" + $backend + " " + $cmd
& wsl.exe @WslBase -- bash -lc $script
exit $LASTEXITCODE
