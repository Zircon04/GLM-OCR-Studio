"""Markdown processing utilities for image region resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from glmocr.utils.image_utils import crop_image_region, pdf_to_images_pil
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_image_regions(
    json_result: list,
    markdown_result: str,
    source: str,
    image_prefix: str = "cropped",
) -> Tuple[list, str, Dict[str, Any]]:
    """Crop image regions from the original file, resolve markdown and JSON paths.

    For results where image regions only have bbox references (e.g. MaaS),
    this function loads the original file, crops each image region, and
    produces the ``image_files`` dict that ``PipelineResult.save()`` persists
    to disk.

    Args:
        json_result: List-of-pages recognition results (list of lists of
            region dicts).
        markdown_result: Markdown text potentially containing
            ``![](page=N,bbox=[...])`` placeholders.
        source: Path to the original image or PDF file.
        image_prefix: Filename prefix for cropped images.

    Returns:
        (updated_json_result, updated_markdown_result, image_files)
    """
    has_images = any(
        r.get("label") == "image"
        for page in json_result
        if isinstance(page, list)
        for r in page
        if isinstance(r, dict)
    )
    if not has_images:
        return json_result, markdown_result, {}

    path = Path(source)
    loaded_images: list = []
    try:
        if path.suffix.lower() == ".pdf" and path.is_file():
            loaded_images = pdf_to_images_pil(
                str(path),
                dpi=200,
                max_width_or_height=3500,
            )
        elif path.is_file():
            img = Image.open(str(path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            loaded_images.append(img)
    except Exception as e:
        logger.warning("Cannot load source %s for image cropping: %s", source, e)
        return json_result, markdown_result, {}

    if not loaded_images:
        return json_result, markdown_result, {}

    image_files: Dict[str, Any] = {}
    image_counter = 0
    updated_json: List[list] = []

    for page_idx, page in enumerate(json_result):
        if not isinstance(page, list):
            updated_json.append(page)
            continue
        page_copy = []
        for region in page:
            if (
                not isinstance(region, dict)
                or region.get("label") != "image"
                or page_idx >= len(loaded_images)
            ):
                page_copy.append(region)
                continue

            bbox = region.get("bbox_2d")
            region_copy = dict(region)
            if bbox:
                try:
                    cropped = crop_image_region(loaded_images[page_idx], bbox)
                    filename = f"{image_prefix}_page{page_idx}_idx{image_counter}.jpg"
                    rel_path = f"imgs/{filename}"
                    image_files[filename] = cropped
                    region_copy["image_path"] = rel_path

                    old_tag = f"![](page={page_idx},bbox={bbox})"
                    new_tag = f"![Image {page_idx}-{image_counter}]({rel_path})"
                    markdown_result = markdown_result.replace(old_tag, new_tag, 1)
                    image_counter += 1
                except Exception as e:
                    logger.warning(
                        "Failed to crop image (page=%d, bbox=%s): %s",
                        page_idx,
                        bbox,
                        e,
                    )
            page_copy.append(region_copy)
        updated_json.append(page_copy)

    return updated_json, markdown_result, image_files
