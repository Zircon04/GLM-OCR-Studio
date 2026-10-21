#!/usr/bin/env python3
"""Searchable PDF: исходный скан + невидимый текстовый слой по bbox.

Страницы и JSON обрабатываются по одной, без загрузки всех картинок
документа в память. Сжатие касается только растровых изображений страниц.
"""

from __future__ import annotations

import html
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import pymupdf

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow есть в requirements/sdk.txt
    Image = None  # type: ignore

# Координаты GLM-OCR нормализованы в 0..1000 (см. reconstruct_pdf.block_rect).
# Значения заметно выше этого диапазона считаем пикселями скана.
_NORM_LIMIT = 1100.0

_FONT_CANDIDATES = (
    # macOS: кириллица, математические символы.
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    # Linux / CI.
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)

_HEADING_LEVELS = {
    "doc_title": 1,
    "title": 1,
    "paragraph_title": 2,
    "section_title": 2,
}
_FORMULA_LABELS = {
    "formula",
    "display_formula",
    "inline_formula",
    "equation",
    "isolate_formula",
}
_IMAGE_LABELS = {"image", "chart", "figure"}
_FOOTNOTE_LABELS = {"footnote", "vision_footnote", "footer_note"}

_MD_LINK = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)|\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HTML_A = re.compile(
    r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_RAW_URL = re.compile(r"(?:https?://|mailto:)[^\s<>\[\]()\"']+", re.IGNORECASE)
_FN_REF = re.compile(r"\[\^([^\]]+)\]")
_FN_DEF = re.compile(r"^\[\^([^\]]+)\]\s*:?\s*")
_PAGE_REF = re.compile(
    r"(?:стр(?:\.|\u0430ниц[\u0430еуы])?|pages?|p\.|\u0441\.)\s*(\d{1,4})",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_PIPE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def find_font() -> Path:
    """Шрифт с кириллицей: на Mac — Arial Unicode, иначе DejaVu/Liberation."""
    for raw in _FONT_CANDIDATES:
        path = Path(raw)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Не найден шрифт с кириллицей (Arial Unicode или DejaVu Sans). "
        "На macOS он лежит в /System/Library/Fonts/Supplemental/."
    )


def iter_json_pages(path: Path) -> Iterator[Any]:
    """Отдаёт страницы OCR-JSON по одной.

    Формат GLM-OCR — массив страниц ``[[block, ...], ...]``. Файл целиком
    в память не читается: декодируется только текущая страница.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        eof = False

        def pull(min_chars: int = 1) -> None:
            nonlocal buffer, eof
            while not eof and len(buffer) < min_chars:
                chunk = handle.read(1 << 16)
                if not chunk:
                    eof = True
                    break
                buffer += chunk

        pull(1)
        buffer = buffer.lstrip()
        if not buffer:
            return
        if buffer[0] != "[":
            # Редкий обёрнутый объект, не массив страниц. Такие файлы небольшие.
            payload = json.loads(buffer + handle.read())
            yield from _coerce_document(payload)
            return
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                pull(1)
                buffer = buffer.lstrip()
            if not buffer or buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            while True:
                try:
                    page, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof or len(buffer) > 48 * 1024 * 1024:
                        raise ValueError(f"Некорректный JSON результата OCR: {path}") from None
                    before = len(buffer)
                    pull(before + (1 << 16))
                    if len(buffer) == before:
                        raise ValueError(f"Некорректный JSON результата OCR: {path}") from None
                    continue
                buffer = buffer[index:]
                yield page
                break


def _coerce_document(payload: Any) -> Iterator[Any]:
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        for key in ("pages", "json_result", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from value
                return
    raise ValueError("JSON результата не содержит список страниц")


def normalize_blocks(page: Any) -> list[dict[str, Any]]:
    if isinstance(page, dict):
        for key in ("blocks", "regions", "elements", "layout"):
            if isinstance(page.get(key), list):
                page = page[key]
                break
        else:
            page = [page] if ("content" in page or "bbox_2d" in page) else []
    if not isinstance(page, list):
        return []
    return [block for block in page if isinstance(block, dict)]


def parse_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if box[2] < box[0]:
        box[0], box[2] = box[2], box[0]
    if box[3] < box[1]:
        box[1], box[3] = box[3], box[1]
    if box[2] - box[0] < 0.5 or box[3] - box[1] < 0.5:
        return None
    return box


def coordinate_frame(blocks: list[dict[str, Any]], image_px: tuple[int, int] | None) -> tuple[str, tuple[float, float]]:
    """Возвращает ('norm'|'pixel', (width, height)) для перевода bbox в 0..1000."""
    boxes = [box for box in (parse_bbox(block.get("bbox_2d")) for block in blocks) if box]
    if not boxes:
        return "norm", (1000.0, 1000.0)
    peak = max(max(box) for box in boxes)
    if peak <= _NORM_LIMIT:
        return "norm", (1000.0, 1000.0)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    if image_px and image_px[0] >= max_x * 0.8 and image_px[1] >= max_y * 0.8:
        return "pixel", (float(image_px[0]), float(image_px[1]))
    return "pixel", (max_x, max_y)


def to_normalized(box: list[float], frame: tuple[float, float]) -> list[float]:
    width, height = frame
    if width <= 0 or height <= 0:
        return box
    # Уже нормализованные координаты передают кадр 1000×1000.
    if abs(width - 1000) < 0.1 and abs(height - 1000) < 0.1 and max(box) <= _NORM_LIMIT:
        return box
    return [
        box[0] / width * 1000.0,
        box[1] / height * 1000.0,
        box[2] / width * 1000.0,
        box[3] / height * 1000.0,
    ]


def label_of(block: dict[str, Any]) -> str:
    return str(block.get("label") or block.get("native_label") or "text").strip().lower()


def footnote_id(block: dict[str, Any], raw: str) -> str | None:
    explicit = block.get("footnote_id") or block.get("fn")
    match = _FN_DEF.match(raw.strip())
    if match:
        return match.group(1).strip()
    if label_of(block) in _FOOTNOTE_LABELS or explicit:
        if explicit:
            return str(explicit).strip()
        plain = re.match(r"^(\d{1,3})[.)]\s+", raw.strip())
        if plain:
            return plain.group(1)
    return None


def heading_level(block: dict[str, Any], raw: str) -> int | None:
    match = re.match(r"^(#{1,6})\s+\S", raw.strip())
    if match:
        return len(match.group(1))
    return _HEADING_LEVELS.get(label_of(block))


def _superscript(value: str) -> str:
    if value.isdigit():
        return value.translate(_SUP)
    return value


def strip_markup(raw: str) -> tuple[str, list[dict[str, Any]], str | None]:
    """Убирает markdown/HTML и собирает ссылки, не теряя текст якоря.

    Возвращает (текст для очистки формул, список ссылок, id сноски-определения).
    """
    defined = footnote_id({"label": ""}, raw)
    # footnote_id без label не увидит «1.» — определение [^id] всё равно поймает.
    text = raw.strip()
    if defined and _FN_DEF.match(text):
        text = _FN_DEF.sub("", text, count=1)

    links: list[dict[str, Any]] = []

    def markdown_link(match: re.Match[str]) -> str:
        # Картинки сюда не кладём: их обрабатывает ветка image.
        if match.group(1) is not None:
            return match.group(0)
        label = (match.group(3) or "").strip()
        target = (match.group(4) or "").strip()
        links.append({"anchor": label, "target": target})
        return label

    text = _MD_LINK.sub(markdown_link, text)

    def html_link(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        label = _TAG.sub("", match.group(2))
        label = html.unescape(re.sub(r"\s+", " ", label)).strip()
        links.append({"anchor": label or target, "target": target})
        return label or target

    text = _HTML_A.sub(html_link, text)

    def footnote_ref(match: re.Match[str]) -> str:
        fid = match.group(1).strip()
        mark = _superscript(fid)
        links.append({"anchor": mark, "target": f"#fn:{fid}"})
        return mark

    text = _FN_REF.sub(footnote_ref, text)

    def raw_url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;)")
        if any(item.get("target") == url for item in links):
            return url
        links.append({"anchor": url, "target": url})
        return url

    text = _RAW_URL.sub(raw_url, text)
    return text, links, defined


def _links_for(raw: str) -> tuple[str, bool, list[dict[str, Any]], str | None]:
    stripped, links, defined = strip_markup(raw)
    stripped = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", stripped)
    from reconstruct_pdf import clean_markdown_text

    text, heading = clean_markdown_text(stripped)
    text = html.unescape(_TAG.sub(" ", text)).replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    occupied = {str(item.get("anchor") or "") for item in links}
    for match in _PAGE_REF.finditer(text):
        anchor = match.group(0)
        if anchor in occupied:
            continue
        links.append({"anchor": anchor, "target": f"#page:{match.group(1)}"})
    # Якоря ссылок тоже прогоняем через очистку, чтобы найти их в итоговом тексте.
    cleaned_links = []
    for item in links:
        anchor = str(item.get("anchor") or "")
        if anchor and anchor not in text:
            anchor_clean, _ = clean_markdown_text(anchor)
            anchor_clean = html.unescape(_TAG.sub(" ", anchor_clean)).strip()
            if anchor_clean:
                anchor = anchor_clean
        cleaned_links.append({**item, "anchor": anchor})
    return text, heading, cleaned_links, defined


def parse_table(raw: str) -> list[list[tuple[str, list[dict[str, Any]]]]] | None:
    """HTML- или markdown-таблица → строки ячеек (текст, ссылки)."""

    def cells_from(pieces: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
        parsed: list[tuple[str, list[dict[str, Any]]]] = []
        for piece in pieces:
            text, _heading, links, _defined = _links_for(piece)
            parsed.append((re.sub(r"\s+", " ", text).strip(), links))
        return parsed

    if "<table" in raw.lower() or "<tr" in raw.lower():
        rows = []
        for row_html in _TR.findall(raw):
            row = cells_from(_CELL.findall(row_html))
            if any(text for text, _links in row):
                rows.append(row)
        return rows or None
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2 or not all("|" in line for line in lines[:2]):
        return None
    rows = []
    for line in lines:
        if _PIPE_RULE.match(line):
            continue
        row = cells_from([cell.strip() for cell in line.strip().strip("|").split("|")])
        if any(text for text, _links in row):
            rows.append(row)
    return rows or None


def _line_metrics(font: pymupdf.Font, size: float) -> tuple[float, float]:
    ascent = max(0.55, float(font.ascender)) * size
    descent = min(-0.15, float(font.descender)) * size
    return ascent, ascent - descent


def _break_token(token: str, font: pymupdf.Font, size: float, width: float) -> list[str]:
    if font.text_length(token, fontsize=size) <= width:
        return [token]
    parts: list[str] = []
    buf = ""
    for char in token:
        trial = buf + char
        if buf and font.text_length(trial, fontsize=size) > width:
            parts.append(buf)
            buf = char
        else:
            buf = trial
    if buf:
        parts.append(buf)
    return parts or [token]


def wrap_text(text: str, font: pymupdf.Font, size: float, width: float) -> list[str]:
    lines: list[str] = []
    paragraphs = re.split(r"\n+", text.strip())
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        current = ""
        for word in words:
            for piece in _break_token(word, font, size, width):
                trial = piece if not current else f"{current} {piece}"
                if not current or font.text_length(trial, fontsize=size) <= width + 0.2:
                    current = trial
                else:
                    lines.append(current)
                    current = piece
        if current:
            lines.append(current)
    return lines


def layout_text(text: str, font: pymupdf.Font, rect: pymupdf.Rect) -> tuple[float, list[str]]:
    """Подбирает кегль, чтобы строки легли внутрь прямоугольника блока."""
    text = text.strip()
    if not text or rect.width < 2 or rect.height < 2:
        return 0.0, []
    min_size = 3.0
    max_size = min(26.0, max(min_size, rect.height * 0.92))
    best_size = min_size
    best_lines = wrap_text(text, font, min_size, rect.width)

    def fits(size: float, lines: list[str]) -> bool:
        if not lines:
            return False
        _ascent, step = _line_metrics(font, size)
        if step * len(lines) > rect.height + 0.6:
            return False
        return all(font.text_length(line, fontsize=size) <= rect.width + 0.8 for line in lines)

    low, high = min_size, max_size
    for _ in range(14):
        mid = (low + high) / 2
        lines = wrap_text(text, font, mid, rect.width)
        if fits(mid, lines):
            best_size, best_lines = mid, lines
            low = mid
        else:
            high = mid
    return best_size, best_lines


def write_invisible(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    lines: list[str],
    font: pymupdf.Font,
    size: float,
    align: str = "left",
) -> None:
    """Кладёт текст строго в rect.

    Режим отрисовки 3 (ни заливка, ни обводка) — стандартный невидимый слой OCR.
    Графическая opacity=0 через ExtGState на macOS Preview часто выпадает из
    поиска и выделения, поэтому здесь используется именно render mode 3:
    визуально текст отсутствует, но ищется и копируется.
    """
    if not lines or size <= 0:
        return
    ascent, step = _line_metrics(font, size)
    writer = pymupdf.TextWriter(page.rect)
    y = rect.y0 + ascent
    wrote = False
    for line in lines:
        if y + (step - ascent) > rect.y1 + 0.8:
            break
        if line:
            safe = _font_safe(font, line)
            width = font.text_length(safe, fontsize=size)
            if align == "center":
                x = rect.x0 + max(0.0, (rect.width - width) / 2)
            else:
                x = rect.x0
            writer.append(pymupdf.Point(x, y), safe, font=font, fontsize=size)
            wrote = True
        y += step
    if wrote:
        writer.write_text(page, render_mode=3, overlay=True)


def _font_safe(font: pymupdf.Font, text: str) -> str:
    """Символы вне шрифта заменяются, чтобы слой не падал на редкой формуле."""
    if not hasattr(font, "has_glyph"):
        return text
    chars = []
    for char in text:
        if char.isspace() or font.has_glyph(ord(char)):
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars)


def _span_rect(
    rect: pymupdf.Rect,
    lines: list[str],
    font: pymupdf.Font,
    size: float,
    needle: str,
    start_at: list[int],
) -> pymupdf.Rect | None:
    if not needle or not lines or size <= 0:
        return None
    _ascent, step = _line_metrics(font, size)
    for index, line in enumerate(lines):
        begin = start_at[index] if index < len(start_at) else 0
        found = line.find(needle, begin)
        if found < 0:
            continue
        if index < len(start_at):
            start_at[index] = found + max(1, len(needle))
        prefix = font.text_length(line[:found], fontsize=size)
        width = font.text_length(needle, fontsize=size)
        y0 = rect.y0 + index * step
        box = pymupdf.Rect(rect.x0 + prefix, y0, rect.x0 + prefix + max(width, 4), y0 + step)
        clipped = box & rect
        return clipped if not clipped.is_empty else rect
    return None


def _safe_uri(target: str) -> str | None:
    value = target.strip()
    if len(value) > 2000:
        return None
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "mailto:")):
        return value
    return None


def _image_alt(raw: str) -> str:
    match = re.search(r"!\[([^\]]*)\]\([^)]+\)", raw)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _resolve_crop(result_dir: Path | None, block: dict[str, Any]) -> Path | None:
    if result_dir is None:
        return None
    raw = str(block.get("image_path") or "").strip()
    if not raw:
        match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", str(block.get("content") or ""))
        raw = match.group(1).strip() if match else ""
    if not raw or raw.startswith(("http://", "https://", "data:")):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = result_dir / path
    try:
        resolved = path.resolve()
        root = result_dir.resolve()
    except OSError:
        return None
    if resolved != root and root not in resolved.parents:
        return None
    enhanced = result_dir / "imgs_enhanced" / path.name
    if enhanced.is_file():
        return enhanced
    return path if path.is_file() else None


def _overlay_crop(
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    block: dict[str, Any],
    overlay_crops: bool,
    result_dir: Path | None,
) -> None:
    """Опционально кладёт вырезанную моделью фигуру поверх скана. Текст пишется следом."""
    if not overlay_crops:
        return
    crop = _resolve_crop(result_dir, block)
    if crop is None:
        return
    try:
        page.insert_image(rect, filename=str(crop), keep_proportion=True, overlay=True)
    except Exception as exc:
        _warn(f"кроп {crop.name} не вставлен: {exc}")


def _largest_image(page: pymupdf.Page) -> tuple[int, tuple[int, int]] | None:
    best_xref = 0
    best_area = 0
    best_size: tuple[int, int] | None = None
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []
    for info in infos:
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        xref = int(info.get("xref") or 0)
        if xref and width * height > best_area:
            best_area = width * height
            best_xref = xref
            best_size = (width, height)
    if best_size:
        return best_xref, best_size
    return None


def _displayed_dpi(pixel_size: tuple[int, int], rects: list[pymupdf.Rect]) -> float:
    if not rects:
        return 0.0
    rect = max(rects, key=lambda item: abs(item.width * item.height))
    if rect.width <= 1 or rect.height <= 1:
        return 0.0
    dpi_x = pixel_size[0] / (rect.width / 72.0)
    dpi_y = pixel_size[1] / (rect.height / 72.0)
    return max(dpi_x, dpi_y)


def _open_raster(doc: pymupdf.Document, xref: int) -> Any:
    info = doc.extract_image(xref)
    blob = info.get("image") if info else None
    if blob and Image is not None:
        try:
            with Image.open(io.BytesIO(blob)) as raw:
                return raw.convert("RGBA") if raw.mode in {"P", "RGBA", "LA"} else raw.copy()
        except Exception:
            pass
    pixmap = pymupdf.Pixmap(doc, xref)
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pixmap, 0)
    if Image is None:
        return pixmap
    png = pixmap.tobytes("png")
    pixmap = None
    with Image.open(io.BytesIO(png)) as raw:
        return raw.copy()


def _flatten(image: Any, grayscale: bool) -> Any:
    if Image is None or not isinstance(image, Image.Image):
        return image
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=image.getchannel("A"))
        image.close()
        image = background
    elif image.mode == "CMYK":
        converted = image.convert("RGB")
        image.close()
        image = converted
    elif image.mode not in {"RGB", "L"}:
        converted = image.convert("RGB")
        image.close()
        image = converted
    if grayscale and image.mode != "L":
        converted = image.convert("L")
        image.close()
        image = converted
    return image


def _recompress_xref(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    xref: int,
    *,
    downsample_dpi: int | None,
    quality: int,
    grayscale: bool,
) -> None:
    """Пережимает один xref. Текстовый слой здесь ещё не создан и не затрагивается."""
    if Image is None:
        raise RuntimeError("Для сжатия страниц нужен Pillow")
    image = _open_raster(doc, xref)
    if not isinstance(image, Image.Image):
        return
    try:
        image = _flatten(image, grayscale)
        width, height = image.size
        if downsample_dpi:
            try:
                rects = list(page.get_image_rects(xref))
            except Exception:
                rects = []
            dpi = _displayed_dpi((width, height), rects)
            if dpi > downsample_dpi * 1.05:
                scale = downsample_dpi / dpi
                resized = image.resize(
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    Image.Resampling.LANCZOS,
                )
                image.close()
                image = resized
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=False)
    finally:
        close = getattr(image, "close", None)
        if close:
            close()
    page.replace_image(xref, stream=buffer.getvalue())


def recompress_page(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    seen: set[int],
    *,
    downsample_dpi: int | None,
    quality: int,
    grayscale: bool,
) -> None:
    try:
        images = page.get_images(full=True)
    except Exception as exc:
        _warn(f"не удалось прочитать изображения страницы: {exc}")
        return
    for item in images:
        xref = int(item[0])
        if xref in seen:
            continue
        seen.add(xref)
        try:
            _recompress_xref(
                doc,
                page,
                xref,
                downsample_dpi=downsample_dpi,
                quality=quality,
                grayscale=grayscale,
            )
        except Exception as exc:
            _warn(f"изображение {xref} оставлено без пережатия: {exc}")


def _warn(message: str) -> None:
    # Не используем слово ERROR: job_runner поднимает такие строки в журнал GUI.
    print(f"searchable-pdf: {message}", file=sys.stderr)


def _index_pages(json_path: Path) -> tuple[dict[str, tuple[int, list[float]]], dict[str, tuple[int, list[float]]], list[list[Any]]]:
    """Лёгкий проход по JSON: сноски, якоря заголовков и оглавление.

    Сами тексты страниц не копятся — только точки перехода.
    """
    footnotes: dict[str, tuple[int, list[float]]] = {}
    headings: dict[str, tuple[int, list[float]]] = {}
    outline: list[list[Any]] = []
    last_level = 0
    for page_index, raw_page in enumerate(iter_json_pages(json_path)):
        for block in normalize_blocks(raw_page):
            raw = str(block.get("content") or "")
            box = parse_bbox(block.get("bbox_2d")) or [0.0, 0.0, 1000.0, 40.0]
            fid = footnote_id(block, raw)
            if fid and fid not in footnotes:
                footnotes[fid] = (page_index, box)
            level = heading_level(block, raw)
            if level is None:
                continue
            title, _heading, _links, _defined = _links_for(raw.strip())
            title = title.splitlines()[0].strip()[:180] if title else ""
            if not title:
                continue
            if last_level == 0:
                level = 1
            elif level > last_level + 1:
                level = last_level + 1
            last_level = level
            outline.append([level, title, page_index + 1])
            for key in _heading_keys(title):
                headings.setdefault(key, (page_index, box))
    return footnotes, headings, outline


def _heading_keys(title: str) -> list[str]:
    slug = re.sub(r"\s+", "-", title.strip().lower())
    slug = re.sub(r"[^\w\-]+", "", slug, flags=re.UNICODE).strip("-")
    keys = {title.strip().lower(), slug, slug.replace("-", "")}
    return [key for key in keys if key]


def _resolve_action(
    target: str,
    *,
    footnotes: dict[str, tuple[int, list[float]]],
    headings: dict[str, tuple[int, list[float]]],
    page_count: int,
) -> tuple[str, Any] | None:
    value = target.strip()
    if value.startswith("#fn:"):
        found = footnotes.get(value[4:])
        return ("goto", found) if found else None
    if value.startswith("#page:"):
        try:
            number = int(value[6:])
        except ValueError:
            return None
        if 1 <= number <= page_count:
            return ("goto", (number - 1, [0.0, 0.0, 1000.0, 30.0]))
        return None
    if value.startswith("#"):
        token = value[1:].strip()
        page_match = re.fullmatch(r"(?:page|p)[=-](\d+)", token, flags=re.IGNORECASE)
        if page_match:
            number = int(page_match.group(1))
            if 1 <= number <= page_count:
                return ("goto", (number - 1, [0.0, 0.0, 1000.0, 30.0]))
            return None
        for key in _heading_keys(token.replace("-", " ")) + _heading_keys(token):
            if key in headings:
                return ("goto", headings[key])
        return None
    uri = _safe_uri(value)
    if uri:
        return ("uri", uri)
    return None


def _page_point(page: pymupdf.Page, box: list[float]) -> pymupdf.Point:
    from reconstruct_pdf import block_rect

    rect = block_rect(page, box)
    return pymupdf.Point(rect.x0, rect.y0)


def _place_block(
    page: pymupdf.Page,
    block: dict[str, Any],
    frame: tuple[float, float],
    font: pymupdf.Font,
    *,
    overlay_crops: bool,
    result_dir: Path | None,
) -> list[tuple[pymupdf.Rect, dict[str, Any]]]:
    """Рисует один блок и возвращает прямоугольники будущих ссылок."""
    from reconstruct_pdf import block_rect

    box = parse_bbox(block.get("bbox_2d"))
    if not box:
        return []
    rect = block_rect(page, to_normalized(box, frame))
    inner = pymupdf.Rect(rect.x0 + 0.4, rect.y0 + 0.3, rect.x1 - 0.4, rect.y1 - 0.3)
    if inner.width < 3 or inner.height < 3:
        inner = rect
    raw = str(block.get("content") or "")
    label = label_of(block)
    text, heading, links, _defined = _links_for(raw)
    explicit = str(block.get("url") or block.get("href") or block.get("link") or block.get("uri") or "").strip()
    if explicit:
        links.append({"anchor": "", "target": explicit, "whole": True})
    if isinstance(block.get("target_page"), int):
        links.append({"anchor": "", "target": f"#page:{int(block['target_page'])}", "whole": True})

    pending: list[tuple[pymupdf.Rect, dict[str, Any]]] = []
    if label in _IMAGE_LABELS or (not text and re.search(r"!\[[^\]]*\]\([^)]+\)", raw)):
        _overlay_crop(page, rect, block, overlay_crops, result_dir)
        alt = text or _image_alt(raw)
        if alt:
            size, lines = layout_text(alt, font, inner)
            write_invisible(page, inner, lines, font, size, align="left")
            pending.extend(_link_boxes(inner, lines, font, size, links))
        elif explicit:
            pending.append((rect, {"target": explicit, "whole": True, "anchor": ""}))
        return pending

    rows = parse_table(raw) if (label == "table" or "<table" in raw.lower() or raw.count("|") >= 2) else None
    if label == "table" or rows:
        if label == "table":
            _overlay_crop(page, rect, block, overlay_crops, result_dir)
        if rows:
            cols = max(len(row) for row in rows)
            row_h = rect.height / max(1, len(rows))
            col_w = rect.width / max(1, cols)
            for row_index, row in enumerate(rows):
                for col_index, (cell, cell_links) in enumerate(row):
                    if not cell:
                        continue
                    cell_rect = pymupdf.Rect(
                        rect.x0 + col_index * col_w + 1,
                        rect.y0 + row_index * row_h + 0.6,
                        rect.x0 + (col_index + 1) * col_w - 1,
                        rect.y0 + (row_index + 1) * row_h - 0.4,
                    )
                    if cell_rect.width < 2 or cell_rect.height < 2:
                        continue
                    size, lines = layout_text(cell, font, cell_rect)
                    write_invisible(page, cell_rect, lines, font, size, align="left")
                    pending.extend(_link_boxes(cell_rect, lines, font, size, cell_links))
            return pending
        flat = re.sub(r"\s+", " ", text).strip()
        size, lines = layout_text(flat, font, inner)
        write_invisible(page, inner, lines, font, size)
        return _link_boxes(inner, lines, font, size, links)

    if not text:
        return pending
    align = "center" if label in _FORMULA_LABELS or label in {"doc_title", "title"} or heading and label == "doc_title" else "left"
    if label in _FORMULA_LABELS:
        align = "center"
    size, lines = layout_text(text, font, inner)
    write_invisible(page, inner, lines, font, size, align=align)
    pending.extend(_link_boxes(inner, lines, font, size, links))
    return pending


def _link_boxes(
    rect: pymupdf.Rect,
    lines: list[str],
    font: pymupdf.Font,
    size: float,
    links: list[dict[str, Any]],
) -> list[tuple[pymupdf.Rect, dict[str, Any]]]:
    boxes: list[tuple[pymupdf.Rect, dict[str, Any]]] = []
    cursor = [0] * max(1, len(lines))
    for link in links:
        anchor = str(link.get("anchor") or "")
        if link.get("whole") or not anchor:
            boxes.append((rect, link))
            continue
        found = _span_rect(rect, lines, font, size, anchor, cursor)
        boxes.append((found or rect, link))
    return boxes


def _apply_links(
    doc: pymupdf.Document,
    pending: list[tuple[int, pymupdf.Rect, dict[str, Any]]],
    footnotes: dict[str, tuple[int, list[float]]],
    headings: dict[str, tuple[int, list[float]]],
    frames: dict[int, tuple[float, float]],
) -> None:
    seen: set[tuple[Any, ...]] = set()
    for page_index, rect, link in pending:
        action = _resolve_action(
            str(link.get("target") or ""),
            footnotes=footnotes,
            headings=headings,
            page_count=doc.page_count,
        )
        if action is None:
            continue
        kind, payload = action
        page = doc[page_index]
        area = rect & page.rect
        if area.is_empty or area.width < 1 or area.height < 1:
            continue
        if kind == "uri":
            key = (page_index, round(area.x0, 1), round(area.y0, 1), payload)
            if key in seen:
                continue
            seen.add(key)
            try:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": area, "uri": payload})
            except Exception as exc:
                _warn(f"ссылка {payload} не добавлена: {exc}")
            continue
        target_page, target_box = payload
        if not (0 <= int(target_page) < doc.page_count):
            continue
        key = (page_index, round(area.x0, 1), round(area.y0, 1), int(target_page))
        if key in seen:
            continue
        seen.add(key)
        try:
            frame = frames.get(int(target_page), (1000.0, 1000.0))
            destination = _page_point(doc[int(target_page)], to_normalized(list(target_box), frame))
            page.insert_link({
                "kind": pymupdf.LINK_GOTO,
                "from": area,
                "page": int(target_page),
                "to": destination,
            })
        except Exception as exc:
            _warn(f"внутренняя ссылка на стр. {int(target_page) + 1} не добавлена: {exc}")


def make_searchable_pdf(
    source: Path,
    json_path: Path,
    output: Path,
    downsample_dpi: int | None = None,
    jpeg_quality: int | None = 80,
    force_grayscale: bool = False,
    overlay_crops: bool = False,
    result_dir: Path | None = None,
    progress: Any = None,
) -> None:
    """Добавляет невидимый текстовый слой поверх исходного PDF.

    Параметры сжатия относятся только к картинкам страниц:
    ``downsample_dpi=None`` не меняет число пикселей, ``jpeg_quality<=0``
    отключает пережатие в JPEG, ``force_grayscale`` переводит картинки в серый.
    Существующий вызов ``make_searchable_pdf(source, json_path, output)`` сохранён.
    """
    source = Path(source)
    json_path = Path(json_path)
    output = Path(output)
    if source.suffix.lower() != ".pdf":
        return
    if not source.is_file():
        raise FileNotFoundError(f"Исходный PDF не найден: {source}")
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON результата OCR не найден: {json_path}")

    quality = None if jpeg_quality is None or jpeg_quality <= 0 else max(1, min(95, int(jpeg_quality)))
    dpi = int(downsample_dpi) if downsample_dpi and int(downsample_dpi) > 0 else None
    grayscale = bool(force_grayscale)
    compress = quality is not None or dpi is not None or grayscale
    jpeg_out = quality or 85

    output.parent.mkdir(parents=True, exist_ok=True)
    font = pymupdf.Font(fontfile=str(find_font()))
    footnotes, headings, ocr_outline = _index_pages(json_path)

    doc = pymupdf.open(source)
    part = output.with_name(output.name + ".part")
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PermissionError("PDF защищён паролем, текстовый слой не добавлен")
        try:
            source_toc = doc.get_toc(simple=True) or []
        except Exception:
            source_toc = []
        seen_images: set[int] = set()
        # Оригинальный размер картинки нужен для пиксельных bbox и не должен
        # измениться, если xref сначала пережали на предыдущей странице.
        original_px: dict[int, tuple[int, int]] = {}
        page_frames: dict[int, tuple[float, float]] = {}
        pending_links: list[tuple[int, pymupdf.Rect, dict[str, Any]]] = []
        pages = iter_json_pages(json_path)
        total = doc.page_count
        try:
            for page_index in range(total):
                page = doc[page_index]
                image_px = _remember_image_size(page, original_px)
                if compress:
                    recompress_page(
                        doc,
                        page,
                        seen_images,
                        downsample_dpi=dpi,
                        quality=jpeg_out,
                        grayscale=grayscale,
                    )
                try:
                    raw_page = next(pages)
                except StopIteration:
                    raw_page = None
                if raw_page is not None:
                    blocks = normalize_blocks(raw_page)
                    _space, frame = coordinate_frame(blocks, image_px)
                    page_frames[page_index] = frame
                    ordered = sorted(
                        blocks,
                        key=lambda block: (
                            (parse_bbox(block.get("bbox_2d")) or [0, 0, 0, 0])[1],
                            (parse_bbox(block.get("bbox_2d")) or [0, 0, 0, 0])[0],
                        ),
                    )
                    for block in ordered:
                        try:
                            for rect, link in _place_block(
                                page,
                                block,
                                frame,
                                font,
                                overlay_crops=overlay_crops,
                                result_dir=Path(result_dir) if result_dir else None,
                            ):
                                pending_links.append((page_index, rect, link))
                        except Exception as exc:
                            _warn(f"блок на стр. {page_index + 1} пропущен: {exc}")
                page = None
                if progress is not None:
                    progress(page_index + 1, total)
        finally:
            pages.close()

        _apply_links(doc, pending_links, footnotes, headings, page_frames)
        if not source_toc and ocr_outline:
            try:
                doc.set_toc(ocr_outline)
            except Exception as exc:
                _warn(f"оглавление OCR не записано: {exc}")
        # garbage/deflate/clean сжимают структуру и потоки, но не перерисовывают текст.
        doc.save(part, garbage=3, deflate=True, clean=True)
    except Exception:
        if part.exists():
            part.unlink(missing_ok=True)
        raise
    finally:
        doc.close()
    part.replace(output)


def _remember_image_size(page: pymupdf.Page, cache: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
    found = _largest_image(page)
    if not found:
        return None
    xref, size = found
    cache.setdefault(xref, size)
    return cache[xref]


__all__ = [
    "find_font",
    "iter_json_pages",
    "make_searchable_pdf",
    "parse_bbox",
    "to_normalized",
    "coordinate_frame",
]
