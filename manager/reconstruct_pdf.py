#!/usr/bin/env python3
"""Build a clean digital PDF from GLM-OCR blocks, entirely locally."""

from __future__ import annotations

import argparse
import html
import io
import json
import re
from pathlib import Path
from typing import Any

import pymupdf

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT_FILE = FONT_DIR / "Arial Unicode.ttf"
MATH_FONT_FILE = FONT_DIR / "STIXGeneral.otf"

BASE_CSS = """
@font-face { font-family: OCRSans; src: url('Arial Unicode.ttf'); }
@font-face { font-family: OCRMath; src: url('STIXGeneral.otf'); }
* { box-sizing: border-box; }
body, div, p { margin: 0; padding: 0; font-family: OCRSans, sans-serif; color: #111; }
table { width: 100%; border-collapse: collapse; font-family: OCRSans, sans-serif; }
td, th { border: 0.55pt solid #555; padding: 2.5pt; vertical-align: top; }
th { font-weight: bold; background: #f2f2f2; }
"""

SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
SUBSCRIPT = str.maketrans("0123456789+-=()aehi jklmnoprstuvx".replace(" ", ""),
                          "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
COMMANDS = {
    r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\approx": "≈", r"\infty": "∞",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\sqrt": "√", r"\lg": "log",
    r"\log": "log", r"\ln": "ln", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\delta": "δ", r"\Delta": "Δ", r"\lambda": "λ", r"\mu": "μ", r"\pi": "π",
    r"\rho": "ρ", r"\sigma": "σ", r"\omega": "ω", r"\Omega": "Ω", r"\varphi": "φ",
}


def emit(done: int, total: int) -> None:
    print(f"CLEAN_PDF_PROGRESS {done} {total}", flush=True)


def readable_math(value: str) -> str:
    """Turn small inline LaTeX fragments into clean, readable Unicode text."""
    value = value.strip().strip("$")
    for _ in range(4):
        changed = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
        if changed == value:
            break
        value = changed
    for command, replacement in COMMANDS.items():
        value = value.replace(command, replacement)

    def script(match: re.Match[str]) -> str:
        marker, body = match.group(1), match.group(2)
        table = SUPERSCRIPT if marker == "^" else SUBSCRIPT
        translated = body.translate(table)
        return translated if len(translated) == len(body) and not any(ch in translated for ch in "{}") else marker + "(" + body + ")"

    value = re.sub(r"([_^])\{([^{}]+)\}", script, value)
    value = re.sub(r"([_^])([0-9A-Za-z])", script, value)
    value = value.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    value = re.sub(r"\\(?:mathrm|mathbf|text|operatorname)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    return value.replace("{", "").replace("}", "").strip()


def readable_inline_formulas(value: str) -> str:
    value = re.sub(r"\$\$?(.+?)\$\$?", lambda match: readable_math(match.group(1)), value, flags=re.DOTALL)
    # OCR can occasionally lose only one of the two dollar delimiters.
    value = value.replace("$", "")
    value = re.sub(r"([_^])\{([^{}]+)\}", lambda match: readable_math(match.group(0)), value)
    return value


def clean_markdown_text(value: str) -> tuple[str, bool]:
    value = value.strip()
    heading = bool(re.match(r"^#{1,6}\s+", value))
    value = re.sub(r"^#{1,6}\s+", "", value)
    value = re.sub(r"^[-*+]\s+", "• ", value, flags=re.MULTILINE)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*", r"\1", value)
    value = value.replace("`", "")
    value = readable_inline_formulas(value)
    return value, heading


def block_rect(page: pymupdf.Page, bbox: list[float]) -> pymupdf.Rect:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    width, height = page.rect.width, page.rect.height
    rect = pymupdf.Rect(x0 / 1000 * width, y0 / 1000 * height,
                        x1 / 1000 * width, y1 / 1000 * height)
    # Avoid zero-sized or edge-clipped boxes.
    rect.x0 = max(0, min(rect.x0, width - 2)); rect.x1 = max(rect.x0 + 2, min(rect.x1, width))
    rect.y0 = max(0, min(rect.y0, height - 2)); rect.y1 = max(rect.y0 + 2, min(rect.y1, height))
    return rect


def font_size_for(text: str, rect: pymupdf.Rect, heading: bool = False) -> float:
    lines = max(1, text.count("\n") + 1)
    longest = max((len(line) for line in text.splitlines()), default=1)
    by_height = rect.height / max(1.25, lines * 1.18)
    by_width = rect.width / max(4.0, longest * 0.52)
    upper = 20.0 if heading else 13.0
    return max(5.5, min(upper, by_height, by_width * (1.12 if heading else 1.0)))


def formula_png(content: str) -> bytes | None:
    formula = content.strip()
    formula = re.sub(r"^\$\$?|\$\$?$", "", formula).strip()
    formula = re.sub(r"\\begin\{(?:aligned|align\*?|equation\*?|gathered)\}", r"\\begin{matrix}", formula)
    formula = re.sub(r"\\end\{(?:aligned|align\*?|equation\*?|gathered)\}", r"\\end{matrix}", formula)
    formula = formula.replace("\\lg", "\\log")
    if not formula:
        return None
    try:
        from matplotlib.mathtext import math_to_image

        output = io.BytesIO()
        math_to_image(f"${formula}$", output, dpi=260, format="png", color="#111111")
        return output.getvalue()
    except Exception:
        return None


def enhanced_map(result: Path) -> dict[str, Path]:
    manifest = result / "image-enhancement-manifest.json"
    mapping: dict[str, Path] = {}
    if manifest.exists():
        try:
            for item in json.loads(manifest.read_text(encoding="utf-8")):
                original = Path(str(item.get("original", "")))
                enhanced = Path(str(item.get("enhanced", "")))
                if enhanced.exists():
                    mapping[str(original.resolve())] = enhanced
                    mapping[original.name] = enhanced
        except Exception:
            pass
    return mapping


def resolve_image(result: Path, block: dict[str, Any], enhanced: dict[str, Path]) -> Path | None:
    raw = str(block.get("image_path") or "").strip()
    if not raw:
        match = re.search(r"!\[[^]]*\]\(([^)]+)\)", str(block.get("content") or ""))
        raw = match.group(1) if match else ""
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = result / path
    if str(path.resolve()) in enhanced:
        return enhanced[str(path.resolve())]
    if path.name in enhanced:
        return enhanced[path.name]
    sibling = result / "imgs_enhanced" / path.name
    if sibling.exists():
        return sibling
    return path if path.exists() else None


def page_sizes(source: Path, count: int) -> list[tuple[float, float]]:
    if source.suffix.lower() == ".pdf" and source.exists():
        original = pymupdf.open(source)
        sizes = [(page.rect.width, page.rect.height) for page in original]
        original.close()
        if sizes:
            return [sizes[min(index, len(sizes) - 1)] for index in range(count)]
    return [(595.28, 841.89)] * count


def build(source: Path, json_path: Path, result: Path, output: Path) -> None:
    pages = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list):
        raise ValueError("JSON результата не содержит список страниц")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    archive = pymupdf.Archive(str(FONT_DIR))
    image_mapping = enhanced_map(result)
    sizes = page_sizes(source, len(pages))
    source_doc = pymupdf.open(source) if source.suffix.lower() == ".pdf" and source.exists() else None

    for page_index, blocks in enumerate(pages):
        width, height = sizes[page_index]
        page = document.new_page(width=width, height=height)
        if not isinstance(blocks, list):
            emit(page_index + 1, len(pages)); continue
        ordered = sorted((b for b in blocks if isinstance(b, dict)),
                         key=lambda b: ((b.get("bbox_2d") or [0, 0])[1], (b.get("bbox_2d") or [0])[0]))
        for block in ordered:
            bbox = block.get("bbox_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            rect = block_rect(page, bbox)
            label = str(block.get("label") or block.get("native_label") or "text").lower()
            content = str(block.get("content") or "").strip()
            if label in {"image", "chart"}:
                image_path = resolve_image(result, block, image_mapping)
                if image_path:
                    try: page.insert_image(rect, filename=str(image_path), keep_proportion=True, overlay=True)
                    except Exception: pass
                continue
            if label in {"formula", "display_formula", "inline_formula", "equation", "isolate_formula"}:
                rendered = formula_png(content)
                placed = False
                if rendered:
                    try:
                        page.insert_image(rect, stream=rendered, keep_proportion=True, overlay=True)
                        placed = True
                    except Exception:
                        placed = False
                if not placed and source_doc is not None and page_index < source_doc.page_count:
                    clip = block_rect(source_doc[page_index], bbox)
                    if clip.width > 2 and clip.height > 2:
                        try:
                            crop = source_doc[page_index].get_pixmap(clip=clip, dpi=144, alpha=False).tobytes("png")
                            page.insert_image(rect, stream=crop, keep_proportion=True, overlay=True)
                            placed = True
                        except Exception:
                            placed = False
                if not placed:
                    fallback = html.escape(re.sub(r"\$+", "", content)).replace("\n", "<br>")
                    page.insert_htmlbox(rect, f"<div style='font-family:OCRMath;font-size:10pt;text-align:center'>{fallback}</div>",
                                        css=BASE_CSS, archive=archive, scale_low=0.35, overlay=True)
                # Add selectable LaTeX as an invisible accessibility/search layer.
                hidden = html.escape(re.sub(r"\s+", " ", content))
                page.insert_htmlbox(rect, f"<div style='font-size:3pt'>{hidden}</div>", css=BASE_CSS,
                                    archive=archive, opacity=0, scale_low=0.1, overlay=True)
                continue
            if label == "table" or content.lower().startswith("<table"):
                table_content = readable_inline_formulas(content)
                table_html = table_content if "<table" in table_content.lower() else f"<div>{html.escape(table_content)}</div>"
                size = max(5.0, min(9.5, rect.height / 12))
                page.insert_htmlbox(rect, f"<div style='font-size:{size:.2f}pt'>{table_html}</div>",
                                    css=BASE_CSS, archive=archive, scale_low=0.25, overlay=True)
                continue
            if not content:
                continue
            text, heading = clean_markdown_text(content)
            size = font_size_for(text, rect, heading)
            weight = "bold" if heading or label in {"doc_title", "paragraph_title", "title"} else "normal"
            alignment = "center" if label in {"doc_title", "title"} else "left"
            body = html.escape(text).replace("\n", "<br>")
            page.insert_htmlbox(rect, f"<div style='font-size:{size:.2f}pt;font-weight:{weight};text-align:{alignment};line-height:1.16'>{body}</div>",
                                css=BASE_CSS, archive=archive, scale_low=0.30, overlay=True)
        emit(page_index + 1, len(pages))

    if source_doc is not None:
        source_doc.close()
    document.set_metadata({"title": result.name, "author": "GLM OCR Local",
                           "subject": "Цифровая реконструкция OCR с локальной обработкой MLX/Metal"})
    try:
        document.subset_fonts()
    except Exception:
        pass
    document.save(output, garbage=4, deflate=True, clean=True)
    document.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(Path(args.source), Path(args.json), Path(args.result), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
