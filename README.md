# GLM-OCR Studio (macOS Metal / MLX)

> **Полностью автономная (offline) рабочая станция оптического распознавания текста (OCR) и анализа документов для Mac на базе Apple Silicon (M1/M2/M3/M4).** Текущая версия исходников: **1.2.0**.

В 1.2.0: поисковый PDF с текстом по координатам блоков, выбор папки результата, склейка переносов в Markdown, очистка временных файлов и интерфейс Liquid Glass (на macOS 26 — системное стекло). Для GTX 1060 и других карт без RTX добавлен запуск через Ollama. Скрипты Windows в релиз входят, но на Windows не проверялись. Готовый zip для Mac по-прежнему собирается на самом Mac: `./scripts/package_release.sh 1.2.0`.

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

Приложение Swift остаётся только для Mac на Apple Silicon. На NVIDIA тот же пайплайн ходит в Ollama, а не в MLX.

RTX для этого не нужен. vLLM и SGLang требуют compute capability 7.5+ (GeForce RTX 20 и новее) и на GTX 1060 не запускаются. GTX 1060 6 ГБ идёт через Ollama, квант `glm-ocr:q8_0`. На картах от 8 ГБ скрипт берёт F16. Детектор вёрстки сидит на CPU, на видеокарте один запрос за раз. Апскейл RealESRGAN в этом режиме выключен: он собран под MLX.

Linux, включая карту без RTX:

```bash
./setup-nvidia.sh
./start-glm-ocr-nvidia.sh
./run-ocr-nvidia.sh /путь/к/скану.pdf
```

Windows, те же карты, и 1060, и RTX:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows\setup-sdk.ps1
.\windows\start-glm-ocr.ps1
.\windows\run-ocr.ps1 C:\путь\к\скану.pdf
```

Нужны драйвер NVIDIA, [Ollama для Windows](https://ollama.com/download) и Python 3.10–3.12. vLLM этот сценарий не ставит даже на RTX: в обычный Windows он не ставится, только в WSL.

**Сборка для Windows не проверялась.** Скрипты в `windows/` написаны по схеме Linux-версии и на реальной Windows-машине не запускались.

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
