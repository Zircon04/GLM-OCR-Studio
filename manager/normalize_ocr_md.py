#!/usr/bin/env python3
"""Склеивает случайные переносы OCR-Markdown для Obsidian.

Формулы ($$, \\[ \\]), таблицы, код, заголовки и картинки не склеиваются.
Перенос слова через дефис в конце строки снимается (рас-\\nпознавание → распознавание).
Настоящий дефис в «из-за», «кто-то», «Санкт-Петербург» сохраняется.
Повторный запуск ничего не меняет.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Левая или правая часть составных слов, которые нельзя склеивать без дефиса.
_KEEP_LEFT = {
    "из", "во", "кое", "по", "пол", "экс", "вице", "мини", "нью", "лос",
    "санкт", "ростов", "йошкар", "усть", "соль",
}
_KEEP_RIGHT = {
    "то", "либо", "нибудь", "таки", "за", "под", "над", "мой", "твоему",
    "моему", "нашему", "видимому", "прежнему", "русски", "английски",
    "немецки", "французски", "латыни", "разному", "новому", "старому",
}

_HEADING = re.compile(r"^#{1,6}\s")
_LIST = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_HR = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_IMAGE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")


def _is_table_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    if stripped.count("|") >= 2 and (stripped.endswith("|") or "---" in stripped):
        return True
    return False


def _is_hard(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _HEADING.match(stripped) or _HR.match(stripped) or _IMAGE.match(stripped):
        return True
    if stripped.startswith(("<!--", "```", "<", ">", "$$", "\\[", "\\]")):
        return True
    if _is_table_line(stripped):
        return True
    return False


def _keep_hyphen(left: str, right: str) -> bool:
    if not right or not right[0].isalpha():
        return True
    if right[0].isupper():
        return True
    word = left[:-1].split()[-1] if left[:-1].strip() else ""
    if word.casefold() in _KEEP_LEFT or right.split()[0].casefold() in _KEEP_RIGHT:
        return True
    return False


def _join(left: str, right: str) -> str:
    right = right.strip()
    if left.endswith("-") and right[:1].isalpha() and not _keep_hyphen(left, right):
        return left[:-1] + right
    if left.endswith("-") and right[:1].isalpha():
        return left + right
    return left + " " + right


def _take_until(lines: list[str], start: int, opener: str, closer: str) -> tuple[str, int]:
    block = [lines[start].rstrip()]
    index = start + 1
    while index < len(lines):
        block.append(lines[index].rstrip())
        if lines[index].strip() == closer or lines[index].strip().startswith(closer):
            index += 1
            break
        index += 1
    return "\n".join(block), index


def _take_html_table(lines: list[str], start: int) -> tuple[str, int]:
    block = [lines[start].rstrip()]
    index = start + 1
    while index < len(lines):
        block.append(lines[index].rstrip())
        if "</table>" in lines[index].casefold():
            index += 1
            break
        index += 1
    return "\n".join(block), index


def _take_table(lines: list[str], start: int) -> tuple[str, int]:
    block = [lines[start].rstrip()]
    index = start + 1
    while index < len(lines) and _is_table_line(lines[index]):
        block.append(lines[index].rstrip())
        index += 1
    return "\n".join(block), index


def _take_fence(lines: list[str], start: int) -> tuple[str, int]:
    block = [lines[start].rstrip()]
    index = start + 1
    while index < len(lines):
        block.append(lines[index].rstrip())
        if lines[index].strip().startswith("```"):
            index += 1
            break
        index += 1
    return "\n".join(block), index


def normalize_text(text: str) -> str:
    """Возвращает Markdown без случайных переносов внутри абзацев."""
    ended = text.endswith("\n")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    pieces: list[str] = []
    buffer = ""
    index = 0

    def flush() -> None:
        nonlocal buffer
        if buffer:
            pieces.append(buffer)
            buffer = ""

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush()
            block, index = _take_fence(lines, index)
            pieces.append(block)
            continue
        if stripped.startswith("<table"):
            flush()
            block, index = _take_html_table(lines, index)
            pieces.append(block)
            continue
        if stripped == "$$" or (stripped.startswith("$$") and not stripped.endswith("$$")):
            flush()
            block, index = _take_until(lines, index, "$$", "$$")
            pieces.append(block)
            continue
        if stripped == "\\[" or stripped.startswith("\\["):
            flush()
            block, index = _take_until(lines, index, "\\[", "\\]")
            pieces.append(block)
            continue
        if _is_table_line(stripped):
            flush()
            block, index = _take_table(lines, index)
            pieces.append(block)
            continue
        if stripped == "":
            flush()
            pieces.append("")
            index += 1
            continue
        if _is_hard(stripped) or _LIST.match(stripped):
            flush()
            if _LIST.match(stripped):
                buffer = raw.rstrip()
            else:
                pieces.append(raw.rstrip())
            index += 1
            continue
        if buffer:
            buffer = _join(buffer, stripped)
        else:
            buffer = raw.rstrip()
        index += 1
    flush()

    collapsed: list[str] = []
    for piece in pieces:
        if piece == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(piece)
    result = "\n".join(collapsed).strip("\n")
    if ended or result:
        result += "\n"
    return result


def normalize_path(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    cleaned = normalize_text(original)
    if cleaned == original:
        return False
    path.write_text(cleaned, encoding="utf-8")
    return True


def iter_markdown(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".md" else []
    found: list[Path] = []
    for item in path.rglob("*.md"):
        if not item.is_file():
            continue
        if ".job" in item.parts or "imgs" in item.parts or "imgs_enhanced" in item.parts:
            continue
        found.append(item)
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Склеить переносы строк в OCR-Markdown")
    parser.add_argument("path", type=Path, help="Файл .md или папка с ними")
    args = parser.parse_args(argv)
    target = args.path.expanduser()
    if not target.exists():
        print(f"Не найдено: {target}", file=sys.stderr)
        return 2
    files = iter_markdown(target)
    if not files:
        print("Markdown-файлов нет")
        return 0
    changed = 0
    for file in files:
        if normalize_path(file):
            changed += 1
            print(file)
    print(f"Обновлено {changed} из {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
