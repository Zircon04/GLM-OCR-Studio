# GLM-OCR Studio (macOS Metal / MLX)

> **Полностью автономная (offline) рабочая станция оптического распознавания текста (OCR) и анализа документов для Mac на базе Apple Silicon (M1/M2/M3/M4).** Текущая версия исходников: **1.2.0**.

В 1.2.0: поисковый PDF с текстом по координатам блоков, выбор папки результата, склейка переносов в Markdown, очистка временных файлов и интерфейс Liquid Glass (на macOS 26 — системное стекло). Для RTX — нативный CUDA-путь через vLLM (на Windows — через WSL2), для GTX 1060 и других карт без RTX — Ollama. Скрипты Windows в релиз входят, но на Windows не проверялись. Готовый zip для Mac по-прежнему собирается на самом Mac: `./scripts/package_release.sh 1.2.0`.

[English Description Below](#english-overview)

---

## Возможности

- **100% Offline и приватность**: Все вычисления (распознавание текста, таблиц, формул и изображений) происходят локально на вашем Mac через фреймворк Apple MLX с аппаратным ускорением Metal GPU. Никакие файлы или запросы не отправляются в облако.
- **Основано на GLM-OCR**: Современная мультимодальная модель `mlx-community/GLM-OCR-bf16`, оптимизированная для сверхточного распознавания сложных макетов, рукописного текста, математических формул и таблиц.
- **Апскейлинг и улучшение качества (Super-Resolution)**: Встроенный пайплайн на базе `RealESRGAN-MLX` (Metal GPU), автоматически восстанавливающий резкость размытых и низкокачественных сканов/фотографий перед или во время распознавания.
- **Нативный интерфейс macOS (Swift / AppKit)**:
  - Очередь заданий с возможностью паузы, возобновления и защиты от сбоев (чекпоинты по страницам).
  - Сравнение «До / После» в реальном времени с интерактивным слайдером.
  - Встроенный просмотрщик Markdown, структуры документа и превью страниц.
  - Локальная библиотека документов с SQLite FTS5 (полнотекстовый поиск по всем ранее обработанным документам).
  - Индикатор использования оперативной памяти и переключатель профилей производительности (Eco, Balanced, High, Extreme).
  - Встроенный менеджер очистки кэша и временных файлов.
- **Экспорт в любые форматы**:
  - Markdown (`.md`)
  - Чистый HTML (`.html`)
  - Документы Microsoft Word (`.docx`)
  - JSON со структурными блоками и координатами
  - Searchable PDF (оригинальный PDF с невидимым текстовым слоем для поиска)
  - Очищенный цифровой PDF (`_clean.pdf`)
  - ZIP-архив со всеми результатами

---

## Архитектура проекта

```
GLM-OCR-local / GLM-OCR-Studio
├── GLM OCR Local.app             # Собранное нативное приложение macOS (Swift)
├── setup.command                 # Автоматический скрипт настройки окружений (One-Click)
├── start-glm-ocr.command         # Скрипт запуска MLX-сервера
├── stop-glm-ocr.command          # Скрипт остановки MLX-сервера
├── run-ocr.command               # Сценарий командной строки для одиночного распознавания
├── config/
│   ├── local-mlx.yaml            # Конфигурация локального OCR-пайплайна
│   └── performance-profile.env   # Профиль нагрузки и выделения памяти
├── gui/
│   ├── main.swift                # Исходный код Swift GUI (AppKit)
│   ├── Info.plist                # Метаданные приложения
│   └── AppIcon.icns              # Иконка приложения
├── manager/
│   ├── job_runner.py             # Менеджер очередей, парсинга страниц и запуска задач
│   ├── export_results.py         # Модуль экспорта (MD, HTML, DOCX, Searchable PDF, ZIP)
│   ├── reconstruct_pdf.py        # Сборка чистого цифрового PDF
│   ├── library_index.py          # SQLite FTS5 индексация и поиск
│   └── cleanup_manager.py        # Менеджер очистки дискового пространства
├── enhancement/
│   ├── enhance_results.py        # Обертка апскейлера изображений
│   └── realesrgan-mlx/           # Модуль RealESRGAN на Apple MLX
├── GLM-OCR/                      # Базовая библиотека GLM-OCR (SDK)
├── requirements/                 # Зависимости для изолированных venv
└── scripts/
    ├── build_app.sh              # Сборка .app бандла из main.swift
    └── package_release.sh        # Создание zip-архива для GitHub Releases
```

---

## Требования

- **macOS**: 14.0 (Sonoma) или новее
- **Процессор**: Apple Silicon (M1, M2, M3, M4 или Pro/Max/Ultra)
- **Оперативная память**: 16 ГБ RAM или больше (рекомендуется для комфортной работы `GLM-OCR-bf16`)
- **Python**: 3.10 – 3.12 (рекомендуется 3.12 из Homebrew: `brew install python@3.12`)

---

## Быстрый старт

### 1. Установка и сборка
Клонируйте репозиторий с субмодулями:
```bash
git clone --recurse-submodules git@github.com:Zircon04/GLM-OCR-Studio.git
cd GLM-OCR-Studio
```

Запустите скрипт автоматической настройки:
```bash
./setup.command
```
Скрипт автоматически:
1. Проверит наличие Python 3.12.
2. Создаст три изолированных окружения (`.venv-mlx`, `.venv-enhance`, `.venv-sdk`).
3. Скомпилирует `GLM OCR Local.app`.

### 2. Запуск приложения
Дважды кликните по **`GLM OCR Local.app`** или выполните:
```bash
open "GLM OCR Local.app"
```

При первом старте интерфейс загрузит модель `mlx-community/GLM-OCR-bf16` из Hugging Face в локальный кэш (около 4.5 ГБ, загружается один раз).

---

## NVIDIA (Linux) и Windows

Приложение Swift остаётся только для Mac на Apple Silicon. На NVIDIA работает тот же пайплайн из командной строки. Бэкенд выбирается автоматически по compute capability карты (`nvidia-smi`):

| Карта | Compute | Бэкенд | Где считается модель |
|---|---|---|---|
| RTX 20/30/40/50 и новее | ≥ 7.5 | **CUDA (vLLM)** — по умолчанию | Напрямую на CUDA и тензорных ядрах, как MLX на Mac. Детектор вёрстки PP-DocLayoutV3 тоже на CUDA |
| GTX 10xx (1060 и др.), Pascal и старше | < 7.5 | **Ollama** | llama.cpp, квант `glm-ocr:q8_0` на 6 ГБ, F16 от 8 ГБ. Детектор вёрстки на CPU |

Переопределить выбор: `GLMOCR_BACKEND=cuda` или `GLMOCR_BACKEND=ollama` (например, RTX через Ollama). Апскейл Real-ESRGAN на NVIDIA выключен в обоих режимах: он собран под MLX.

> **Не проверялось на реальных картах NVIDIA.** Скрипты написаны по документации GLM-OCR и vLLM, но ни на RTX, ни на GTX, ни на Windows не запускались.

### Linux (или WSL2)

```bash
./setup-nvidia.sh            # RTX: .venv-sdk с CUDA-torch + .venv-vllm (vllm>=0.19); Pascal: CPU-torch
./start-glm-ocr-nvidia.sh    # RTX: vllm serve zai-org/GLM-OCR :8080; Pascal: Ollama :11434
./run-ocr-nvidia.sh /путь/к/скану.pdf [/папка/результатов]
./stop-glm-ocr-nvidia.sh
```

RTX-путь: `vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8080` с MTP-спекуляцией (`--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'`, как в README GLM-OCR). Доля видеопамяти подбирается по объёму карты (0.60 на 8 ГБ … 0.80 на 24 ГБ+), остаток — детектору вёрстки и рабочему столу. Конфиг SDK — `config/nvidia-rtx.yaml`, профиль — `config/nvidia-rtx.env`. Журнал и PID — `logs/vllm-rtx.*`.

Настройки через переменные: `GLMOCR_GPU_UTIL` (доля VRAM), `GLMOCR_MAX_MODEL_LEN` (по умолчанию 16384, на 8 ГБ — 8192), `GLMOCR_MAX_NUM_SEQS` (16), `GLMOCR_VLLM_MTP=0` (выключить спекуляцию), `GLMOCR_VLLM_PORT`, `TORCH_CUDA_INDEX` (колёса PyTorch, по умолчанию cu128 — нужен драйвер ≥ 570), `GLMOCR_LAYOUT_DEVICE=cpu` (детектор вёрстки на CPU, если не хватает VRAM).

Если vLLM не стартует по памяти: `GLMOCR_GPU_UTIL=0.5 GLMOCR_MAX_MODEL_LEN=8192 ./start-glm-ocr-nvidia.sh`. Запасной путь: `GLMOCR_BACKEND=ollama`.

### Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows\setup-sdk.ps1
.\windows\start-glm-ocr.ps1
.\windows\run-ocr.ps1 C:\путь\к\скану.pdf
```

- **RTX (compute ≥ 7.5):** vLLM в обычный Windows не ставится, поэтому эти скрипты сами передают работу в **WSL2** (`windows\rtx-wsl.ps1`), где модель считается на CUDA той же видеокарты. Один раз: `wsl --install -d Ubuntu`, свежий драйвер NVIDIA для Windows, в Ubuntu `sudo apt install python3 python3-venv curl`. Остановка: `.\windows\rtx-wsl.ps1 stop`. Дистрибутив — `$env:GLMOCR_WSL_DISTRO`.
- **GTX 1060 и другие карты без RTX:** прежний путь через [Ollama для Windows](https://ollama.com/download) и Python 3.10–3.12.
- RTX через Ollama без WSL: `$env:GLMOCR_BACKEND = "ollama"`.

**Сборка для Windows не проверялась.**

---

<a name="english-overview"></a>
## English Overview

**GLM-OCR Studio** is a 100% offline, native macOS document understanding & OCR desktop workstation built specifically for Apple Silicon (M1/M2/M3/M4) using Apple's MLX framework and Metal acceleration.

### Features
- **Zero Cloud Leakage**: All inference runs on-device using Metal GPU acceleration via `mlx-vlm`.
- **Integrated Super-Resolution**: Built-in `RealESRGAN-MLX` pipeline to upscale and restore degraded documents.
- **Native Swift GUI**: Complete UI featuring batch queue management, live split comparison slider, document viewer, and real-time logs.
- **Full-Text Searchable Library**: Indexed with SQLite FTS5 for quick retrieval of historical OCR results.
- **Universal Multi-Format Export**: Generates Markdown, Word DOCX, clean HTML, Searchable PDF with hidden text layer, and structured JSON.

---

## Лицензия

Распространяется под лицензией [Apache 2.0](LICENSE).
