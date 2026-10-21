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
    r"(?:стр(?:\.|\u0430ниц[\u0430\u0435\u0443\u044b])?|pages?|p\.|\u0441\.)\s*(\d{1,4})",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_PIPE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
