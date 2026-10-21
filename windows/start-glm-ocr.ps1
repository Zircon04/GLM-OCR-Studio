# GLM-OCR на Windows через Ollama. GTX 1060 6 ГБ и RTX — один и тот же путь.
#
# НЕ ПРОВЕРЯЛОСЬ НА WINDOWS.
#
# RTX здесь не требуется. Ограничение compute capability 7.5+ есть только у vLLM,
# а vLLM в обычный Windows не ставится. Этот скрипт поднимает Ollama:
# на 6 ГБ квант q8_0, на картах от 8 ГБ — F16.
# Скорость на Pascal ниже, чем на RTX: тензорных ядер нет.
#
# Один раз поставьте Ollama: https://ollama.com/download
# Затем из корня репозитория:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\windows\start-glm-ocr.ps1

$ErrorActionPreference = "Stop"
Write-Host "НЕ ПРОВЕРЯЛОСЬ НА WINDOWS."

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path "$Root\logs", "$Root\config" | Out-Null

$ModelAlias = "glm-ocr-pascal"
$Modelfile = "$Root\config\ollama-pascal.Modelfile"
$HealthUrl = "http://127.0.0.1:11434/api/tags"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi не найден. Поставьте драйвер NVIDIA."
    Write-Host "Без карты Ollama может считать на CPU, это будет медленно: ollama pull glm-ocr:q8_0"
    exit 1
}

$Line = (& nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | Select-Object -First 1)
$Parts = $Line -split ","
$GpuName = $Parts[0].Trim()
$VramMib = [int]($Parts[1].Trim() -replace "[^0-9]", "")
$ComputeCap = $Parts[2].Trim()
$FreeLine = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
$FreeMib = [int](($FreeLine | Select-Object -First 1).ToString().Trim() -replace "[^0-9]", "")
Write-Host "Карта: $GpuName, compute $ComputeCap, всего $VramMib МиБ, свободно $FreeMib МиБ"

$Cap = [version](($ComputeCap -split "\s+")[0] + ".0")
if ($Cap -lt [version]"7.5.0") {
    Write-Host "Compute $ComputeCap ниже 7.5: vLLM эту карту не поддерживает. Берём Ollama, в том числе на GTX 1060."
} else {
    Write-Host "Карта подошла бы и для vLLM, но на Windows vLLM штатно не ставится. Этот скрипт всё равно использует Ollama."
}

if ($VramMib -le 6144 -or $FreeMib -lt 4500) {
    $Tag = "glm-ocr:q8_0"
    Write-Host "Мало видеопамяти — квант q8_0. На 1060 закройте браузер и игры."
} else {
    $Tag = "glm-ocr:latest"
    Write-Host "Памяти хватает на F16 (glm-ocr:latest)."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama не установлена: https://ollama.com/download"
    exit 1
}

# Служба Ollama читает переменные пользователя только после перезапуска.
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "User")
if ($VramMib -le 8192) {
    [Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
}
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "30m"
if ($VramMib -le 8192) { $env:OLLAMA_KV_CACHE_TYPE = "q8_0" }

function Test-Ollama {
    try {
        $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Ollama)) {
    Write-Host "Ollama не отвечает. Запустите приложение Ollama из меню Пуск и повторите."
    Write-Host "Если оно уже было запущено до установки переменных, закройте его через трей и откройте снова."
    exit 1
}

Write-Host "Скачиваю $Tag …"
& ollama pull $Tag
if ($LASTEXITCODE -ne 0 -and $Tag -ne "glm-ocr:latest") {
    Write-Host "Тег $Tag недоступен, беру glm-ocr:latest. На 6 ГБ это впритык."
    $Tag = "glm-ocr:latest"
    & ollama pull $Tag
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$ModelText = @"
FROM $Tag
PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER num_predict 4096
"@
# Без BOM: Ollama читает Modelfile как обычный текст.
[System.IO.File]::WriteAllText($Modelfile, $ModelText.Trim() + "`n")

& ollama create $ModelAlias -f $Modelfile
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Прогрев модели…"
$Body = @{
    model  = $ModelAlias
    prompt = "Text Recognition:"
    stream = $false
    options = @{ num_predict = 8 }
} | ConvertTo-Json
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -Body $Body -ContentType "application/json" -TimeoutSec 180 | Out-Null
} catch {
    Write-Host "Модель не ответила. На 1060 часто кончается 6 ГБ, если рабочий стол уже занял память."
    Write-Host $_.Exception.Message
    exit 1
}

Write-Host ""
Write-Host "Сервер готов: http://127.0.0.1:11434  (модель $ModelAlias из $Tag)."
Write-Host "Дальше: .\windows\run-ocr.ps1 C:\путь\к\скану.pdf"
Write-Host "Страница на Pascal занимает десятки секунд. Это нормально."
Write-Host "Повтор: этот сценарий на Windows не проверялся."
