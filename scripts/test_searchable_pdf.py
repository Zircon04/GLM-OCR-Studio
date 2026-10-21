#!/usr/bin/env python3
"""Быстрая проверка searchable PDF без модели GLM-OCR.

Скрипт за несколько секунд:
1. Рисует 3 страницы (заголовок, русский текст, ссылка, таблица, формула, рисунок, сноска).
2. Превращает их в скан — в PDF больше нет текстового слоя.
3. Пишет JSON того же вида, что сохраняет GLM-OCR (bbox_2d 0..1000, метки, markdown).
4. Собирает searchable PDF и проверяет, что каждый фрагмент лежит в своём прямоугольнике,
   ссылки кликаются, оглавление на месте, картинка страницы не пропала и сжатие уменьшает растр.

Полный прогон модели здесь сознательно не запускается: он требует Apple MLX и несколько минут.
Координаты в JSON — это контракт OCR, который проверяется.

Запуск из корня репозитория:
    python3 scripts/test_searchable_pdf.py
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "manager"))

from export_results import make_searchable_pdf  # noqa: E402
from searchable_pdf import coordinate_frame, find_font, iter_json_pages, to_normalized  # noqa: E402

W, H = 595.27, 841.89
TITLE = "Поисковый PDF: проверка слоя"
LINK_LABEL = "GLM-OCR"
FORMULA = r"$$E = mc^2$$" + "\n" + r"$$\int_0^1 x^2\,dx = \frac{1}{3}$$"
TABLE = """<table>
<tr><th>Позиция</th><th>Кол-во</th><th>Цена</th></tr>
<tr><td>Яблоки</td><td>3</td><td>120</td></tr>
<tr><td>Груши</td><td>2</td><td>90</td></tr>
</table>"""


def norm(rect: pymupdf.Rect) -> list[float]:
    return [
        round(rect.x0 / W * 1000, 2),
        round(rect.y0 / H * 1000, 2),
        round(rect.x1 / W * 1000, 2),
        round(rect.y1 / H * 1000, 2),
    ]


def layout() -> list[list[dict]]:
    """Единый источник координат: им же рисуется скан и JSON «как от OCR»."""
    title = pymupdf.Rect(48, 46, 548, 92)
    body = pymupdf.Rect(48, 120, 548, 210)
    cross = pymupdf.Rect(48, 230, 548, 280)
    section = pymupdf.Rect(48, 46, 548, 90)
    table = pymupdf.Rect(48, 120, 548, 280)
    formula = pymupdf.Rect(48, 320, 548, 430)
    appendix = pymupdf.Rect(48, 46, 548, 90)
    figure = pymupdf.Rect(48, 120, 300, 300)
    caption = pymupdf.Rect(48, 314, 420, 360)
    note = pymupdf.Rect(48, 760, 548, 812)
    return [
        [
            {"index": 0, "label": "doc_title", "content": TITLE, "bbox_2d": norm(title), "_rect": title},
            {
                "index": 1,
                "label": "text",
                "content": (
                    "Локальная модель [" + LINK_LABEL + "](https://example.com/glm-ocr) "
                    "читает русский текст, формулы и таблицы."
                ),
                "bbox_2d": norm(body),
                "_rect": body,
            },
            {
                "index": 2,
                "label": "text",
                "content": "Подробности приведены на стр. 3 и в сноске[^1].",
                "bbox_2d": norm(cross),
                "_rect": cross,
            },
            {"index": 3, "label": "text", "content": "битый блок", "bbox_2d": "нет"},
        ],
        [
            {"index": 0, "label": "paragraph_title", "content": "Таблица и формула", "bbox_2d": norm(section), "_rect": section},
            {"index": 1, "label": "table", "content": TABLE, "bbox_2d": norm(table), "_rect": table},
            {"index": 2, "label": "formula", "content": FORMULA, "bbox_2d": norm(formula), "_rect": formula},
        ],
        [
            {"index": 0, "label": "paragraph_title", "content": "Приложение", "bbox_2d": norm(appendix), "_rect": appendix},
            {
                "index": 1,
                "label": "image",
                "content": "![Метка фигуры](imgs/mark.png)",
                "image_path": "imgs/mark.png",
                "bbox_2d": norm(figure),
                "_rect": figure,
            },
            {
                "index": 2,
                "label": "text",
                "content": "Простая фигура для проверки слоя.",
                "bbox_2d": norm(caption),
                "_rect": caption,
            },
            {
                "index": 3,
                "label": "vision_footnote",
                "content": "[^1]: Обработка выполняется на Mac локально.",
                "bbox_2d": norm(note),
                "_rect": note,
            },
        ],
    ]


def draw_mark(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (240, 180), (245, 248, 252))
    pixels = image.load()
    for y in range(40, 140):
        for x in range(30, 110):
            pixels[x, y] = (30, 90, 180)
    for y in range(50, 130):
        for x in range(130, 210):
            if (x - 170) ** 2 + (y - 90) ** 2 <= 40 ** 2:
                pixels[x, y] = (210, 90, 40)
    image.save(path, format="PNG")


def build_scan(directory: Path) -> tuple[Path, Path, Path, list[list[dict]]]:
    pages = layout()
    font = str(find_font())
    vector = pymupdf.open()
    mark = directory / "result" / "imgs" / "mark.png"
    draw_mark(mark)
    for blocks in pages:
        page = vector.new_page(width=W, height=H)
        for block in blocks:
            rect = block.get("_rect")
            if rect is None:
                continue
            label = block["label"]
            if label == "table":
                shape = page.new_shape()
                shape.draw_rect(rect)
                cols, rows = 3, 3
                for index in range(1, cols):
                    x = rect.x0 + rect.width * index / cols
                    shape.draw_line(pymupdf.Point(x, rect.y0), pymupdf.Point(x, rect.y1))
                for index in range(1, rows):
                    y = rect.y0 + rect.height * index / rows
                    shape.draw_line(pymupdf.Point(rect.x0, y), pymupdf.Point(rect.x1, y))
                shape.finish(color=(0.2, 0.2, 0.2), width=0.6)
                shape.commit()
                cells = ["Позиция", "Кол-во", "Цена", "Яблоки", "3", "120", "Груши", "2", "90"]
                for index, cell in enumerate(cells):
                    cell_rect = pymupdf.Rect(
                        rect.x0 + (index % cols) * rect.width / cols + 4,
                        rect.y0 + (index // cols) * rect.height / rows + 8,
                        rect.x0 + (index % cols + 1) * rect.width / cols - 4,
                        rect.y0 + (index // cols + 1) * rect.height / rows - 4,
                    )
                    page.insert_textbox(cell_rect, cell, fontfile=font, fontname="body", fontsize=12)
            elif label == "image":
                page.insert_image(rect, filename=str(mark), keep_proportion=True)
            elif label == "formula":
                page.insert_textbox(rect, "E = mc²\n∫ x² dx = 1/3", fontfile=font, fontname="body", fontsize=16, align=1)
            elif label == "vision_footnote":
                page.insert_textbox(rect, "1. Обработка выполняется на Mac локально.", fontfile=font, fontname="body", fontsize=10)
            else:
                shown = block["content"]
                shown = shown.replace(f"[{LINK_LABEL}](https://example.com/glm-ocr)", LINK_LABEL)
                shown = shown.replace("[^1]", "¹")
                size = 18 if label in {"doc_title", "paragraph_title"} else 12
                page.insert_textbox(rect, shown, fontfile=font, fontname="body", fontsize=size)
    vector.set_toc([[1, "Закладка исходного PDF", 1], [2, "Раздел из исходного PDF", 2]])
    vector_path = directory / "vector.pdf"
    vector.save(vector_path)
    scan = pymupdf.open()
    for page in vector:
        pixmap = page.get_pixmap(dpi=144, alpha=False)
        new = scan.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pixmap)
    scan.set_toc(vector.get_toc())
    scan_path = directory / "scan.pdf"
    scan.save(scan_path, deflate=True)
    vector.close()
    scan.close()

    public = []
    for blocks in pages:
        public.append([{key: value for key, value in block.items() if key != "_rect"} for block in blocks])
    result = directory / "result"
    result.mkdir(parents=True, exist_ok=True)
    json_path = result / "sample.json"
    json_path.write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
    (result / "sample.md").write_text("# проверка\n", encoding="utf-8")
    return scan_path, json_path, result, pages


def assert_inside(page: pymupdf.Page, needle: str, rect: pymupdf.Rect, pad: float = 16) -> pymupdf.Rect:
    hits = page.search_for(needle)
    if not hits:
        raise AssertionError(f"«{needle}» не найден на странице. Текст: {page.get_text()[:500]!r}")
    area = pymupdf.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    for hit in hits:
        center = pymupdf.Point((hit.x0 + hit.x1) / 2, (hit.y0 + hit.y1) / 2)
        if area.contains(center):
            return hit
    raise AssertionError(f"«{needle}» есть, но не в своём bbox {rect}: {hits}")


def first_page_image(path: Path) -> tuple[int, int, str]:
    doc = pymupdf.open(path)
    try:
        info = doc.extract_image(doc[0].get_images()[0][0])
        return int(info["width"]), int(info["height"]), str(info["ext"])
    finally:
        doc.close()


def check_positioning(path: Path, pages: list[list[dict]]) -> None:
    doc = pymupdf.open(path)
    try:
        if doc.page_count != 3:
            raise AssertionError(f"ожидались 3 страницы, получено {doc.page_count}")
        if "битый блок" in doc[0].get_text():
            raise AssertionError("блок без bbox попал в текстовый слой")
        assert_inside(doc[0], "Поисковый", pages[0][0]["_rect"])
        assert_inside(doc[0], "русский", pages[0][1]["_rect"])
        assert_inside(doc[0], LINK_LABEL, pages[0][1]["_rect"])
        assert_inside(doc[0], "стр.", pages[0][2]["_rect"])
        assert_inside(doc[1], "Таблица", pages[1][0]["_rect"])
        apples = assert_inside(doc[1], "Яблоки", pages[1][1]["_rect"])
        price = assert_inside(doc[1], "120", pages[1][1]["_rect"])
        pears = assert_inside(doc[1], "Груши", pages[1][1]["_rect"])
        header = assert_inside(doc[1], "Позиция", pages[1][1]["_rect"])
        table = pages[1][1]["_rect"]
        col = table.width / 3
        if not (table.x0 <= (apples.x0 + apples.x1) / 2 <= table.x0 + col + 8):
            raise AssertionError(f"«Яблоки» не в первом столбце: {apples}")
        if not (table.x0 + 2 * col - 8 <= (price.x0 + price.x1) / 2 <= table.x1 + 8):
            raise AssertionError(f"«120» не в третьем столбце: {price}")
        if pears.y0 <= header.y0:
            raise AssertionError("строки таблицы не разъехались по вертикали")
        assert_inside(doc[1], "mc", pages[1][2]["_rect"])
        formula_text = doc[1].get_textbox(pages[1][2]["_rect"])
        if "²" not in formula_text and "dx" not in formula_text and "(1)/(3)" not in formula_text:
            raise AssertionError(f"формула не стала читаемым текстом: {formula_text!r}")
        assert_inside(doc[2], "Приложение", pages[2][0]["_rect"])
        assert_inside(doc[2], "Метка", pages[2][1]["_rect"])
        assert_inside(doc[2], "фигура", pages[2][2]["_rect"])
        assert_inside(doc[2], "локально", pages[2][3]["_rect"])
        if doc[2].search_for("локально")[0].y0 < 600:
            raise AssertionError("сноска оказалась в верхней части страницы")
        for index, page in enumerate(doc):
            if not page.get_images():
                raise AssertionError(f"на странице {index + 1} пропало исходное изображение")
        links = doc[0].get_links()
        if not any(str(item.get("uri") or "").startswith("https://example.com/glm-ocr") for item in links):
            raise AssertionError(f"нет внешней ссылки: {links}")
        gotos = [item for item in links if item.get("kind") == pymupdf.LINK_GOTO and item.get("page") == 2]
        if not gotos:
            raise AssertionError(f"нет перехода на стр. 3: {links}")
        if not any(item.get("to", pymupdf.Point(0, 0)).y > 700 for item in gotos):
            raise AssertionError(f"сноска не ведёт вниз стр. 3: {gotos}")
        toc = doc.get_toc()
        if not any(item[1] == "Закладка исходного PDF" and item[2] == 1 for item in toc):
            raise AssertionError(f"оглавление исходного PDF потеряно: {toc}")
        if len(doc[2].get_images()) < 2:
            raise AssertionError("кроп фигуры не добавлен поверх скана")
    finally:
        doc.close()


def check_ocr_outline(scan: Path, json_path: Path, output: Path) -> None:
    bare = pymupdf.open(scan)
    bare.set_toc([])
    stripped = output.with_name("scan-no-toc.pdf")
    bare.save(stripped)
    bare.close()
    make_searchable_pdf(stripped, json_path, output, jpeg_quality=0, overlay_crops=False)
    doc = pymupdf.open(output)
    try:
        titles = [item[1] for item in doc.get_toc()]
        if TITLE not in titles or "Таблица и формула" not in titles or "Приложение" not in titles:
            raise AssertionError(f"оглавление не собрано из заголовков OCR: {titles}")
    finally:
        doc.close()


def check_pixel_bbox(directory: Path) -> None:
    image = Image.new("RGB", (2000, 2500), (250, 250, 248))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=500)
    page.insert_image(page.rect, stream=buffer.getvalue())
    source = directory / "pixel-scan.pdf"
    doc.save(source)
    doc.close()
    bbox = [200, 250, 1600, 700]
    payload = [[{"label": "text", "content": "ПиксельныйЯкорь", "bbox_2d": bbox}]]
    json_path = directory / "pixel.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    output = directory / "pixel-searchable.pdf"
    make_searchable_pdf(source, json_path, output, jpeg_quality=0)
    expected = pymupdf.Rect(200 / 2000 * 400, 250 / 2500 * 500, 1600 / 2000 * 400, 700 / 2500 * 500)
    doc = pymupdf.open(output)
    try:
        assert_inside(doc[0], "ПиксельныйЯкорь", expected, pad=8)
    finally:
        doc.close()
    frame = coordinate_frame(payload[0], (2000, 2500))
    if frame[0] != "pixel":
        raise AssertionError(f"пиксельный bbox распознан как {frame}")
    normalized = to_normalized([float(v) for v in bbox], frame[1])
    if abs(normalized[0] - 100) > 0.1 or abs(normalized[2] - 800) > 0.1:
        raise AssertionError(normalized)


def check_noisy_compression(directory: Path) -> None:
    """Плотная картинка: видно, что жмётся растр, а текст остаётся."""
    random.seed(1)
    image = Image.new("RGB", (1800, 2500))
    pixels = image.load()
    for y in range(0, 2500, 2):
        for x in range(0, 1800, 2):
            pixels[x, y] = (random.randrange(256), random.randrange(256), random.randrange(256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    page.insert_image(page.rect, stream=buffer.getvalue())
    source = directory / "noisy.pdf"
    doc.save(source, deflate=False)
    doc.close()
    json_path = directory / "noisy.json"
    json_path.write_text(
        json.dumps([[{"label": "text", "content": "Сжатие не трогает текст", "bbox_2d": [80, 80, 700, 160]}]]),
        encoding="utf-8",
    )
    output = directory / "noisy-searchable.pdf"
    make_searchable_pdf(source, json_path, output, downsample_dpi=72, jpeg_quality=40)
    if output.stat().st_size >= source.stat().st_size / 4:
        raise AssertionError(f"шумная страница почти не сжалась: {output.stat().st_size} против {source.stat().st_size}")
    doc = pymupdf.open(output)
    try:
        if "Сжатие" not in doc[0].get_text():
            raise AssertionError("после сжатия пропал текстовый слой")
        info = doc.extract_image(doc[0].get_images()[0][0])
        if info["ext"] != "jpeg" or info["width"] > 800:
            raise AssertionError(f"downsample не сработал: {info['width']}x{info['height']} {info['ext']}")
    finally:
        doc.close()


def check_stream(json_path: Path) -> None:
    pages = list(iter_json_pages(json_path))
    if len(pages) != 3 or not isinstance(pages[0], list):
        raise AssertionError("потоковый разбор JSON сломан")


def check_cli(scan: Path, result: Path, directory: Path) -> None:
    exports = directory / "cli-out"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "manager" / "export_results.py"),
            "--result",
            str(result),
            "--source",
            str(scan),
            "--output-root",
            str(exports),
            "--formats",
            "searchable_pdf",
            "--jpeg-quality",
            "40",
            "--downsample-dpi",
            "72",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr[-2000:] or completed.stdout[-2000:])
    if "searchable_pdf_progress" not in completed.stdout:
        raise AssertionError("CLI не сообщил прогресс по страницам")
    produced = exports / "exports" / f"{result.name}_searchable.pdf"
    if not produced.is_file():
        raise AssertionError("CLI не создал searchable PDF")
    doc = pymupdf.open(produced)
    try:
        if "русский" not in doc[0].get_text():
            raise AssertionError("CLI-файл без текстового слоя")
    finally:
        doc.close()


def check_grayscale(scan: Path, json_path: Path, output: Path) -> None:
    make_searchable_pdf(scan, json_path, output, jpeg_quality=40, force_grayscale=True, downsample_dpi=72)
    doc = pymupdf.open(output)
    try:
        info = doc.extract_image(doc[0].get_images()[0][0])
        mode = Image.open(io.BytesIO(info["image"])).mode
        if mode != "L":
            raise AssertionError(f"ожидался серый JPEG, получен {mode}")
    finally:
        doc.close()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="glmocr-searchable-") as raw:
        directory = Path(raw)
        print("1/5 рисую тестовый скан и JSON в формате GLM-OCR")
        scan, json_path, result, pages = build_scan(directory)
        check_stream(json_path)
        plain = pymupdf.open(scan)
        if plain[0].get_text().strip():
            plain.close()
            raise AssertionError("скан всё ещё содержит текстовый слой — проверка позиции была бы нечестной")
        plain.close()

        print("2/5 собираю searchable PDF со сжатием и кропом")
        compressed = directory / "searchable.pdf"
        untouched = directory / "searchable-raw.pdf"
        make_searchable_pdf(
            scan,
            json_path,
            compressed,
            downsample_dpi=72,
            jpeg_quality=40,
            overlay_crops=True,
            result_dir=result,
        )
        make_searchable_pdf(scan, json_path, untouched, jpeg_quality=0, overlay_crops=False)
        wide, _high, raw_ext = first_page_image(untouched)
        narrow, _low, small_ext = first_page_image(compressed)
        if narrow > wide * 0.7 or small_ext != "jpeg":
            raise AssertionError(f"картинка страницы не ужата: {wide}px {raw_ext} → {narrow}px {small_ext}")
        check_noisy_compression(directory)

        print("3/5 проверяю позиции, ссылки, таблицу, формулу и оглавление")
        check_positioning(compressed, pages)
        check_ocr_outline(scan, json_path, directory / "from-headings.pdf")

        print("4/5 проверяю пиксельные bbox, серый режим и CLI")
        check_pixel_bbox(directory)
        check_grayscale(scan, json_path, directory / "gray.pdf")
        check_cli(scan, result, directory)

        print("5/5 готово")
        print(
            f"OK  страница {wide}px {raw_ext} → {narrow}px {small_ext}, "
            f"файлы {compressed.stat().st_size // 1024} КБ и {untouched.stat().st_size // 1024} КБ"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise
