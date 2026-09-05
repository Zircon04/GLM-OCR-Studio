"""Shared helpers for the pipeline package."""

from __future__ import annotations

from typing import Any, Dict, List, Union

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def extract_image_sources(request_data: Dict[str, Any]) -> List[Union[str, bytes]]:
    """Extract image sources (URLs or raw bytes) from a request payload."""
    sources: List[Union[str, bytes]] = []
    for msg in request_data.get("messages", []):
        if msg.get("role") == "user":
            contents = msg.get("content", [])
            if isinstance(contents, list):
                for content in contents:
                    if content.get("type") == "image_url":
                        sources.append(content["image_url"]["url"])
                    elif content.get("type") == "image_bytes":
                        sources.append(content["data"])
    return sources


def make_original_inputs(sources: List[Union[str, bytes]]) -> List[str]:
    """Return display-friendly names for each input source."""
    results: List[str] = []
    for i, src in enumerate(sources):
        if isinstance(src, bytes):
            results.append(f"document_{i}")
        elif src.startswith("file://"):
            results.append(src[7:])
        else:
            results.append(src)
    return results


def extract_ocr_content(response: Dict[str, Any]) -> str:
    """Pull the content string out of an OpenAI-style OCR response."""
    return response.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── Queue message "identifier" field values ──────────────────────────
# Every queue message is a dict with an "identifier" key.
IDENTIFIER_IMAGE = "image"
IDENTIFIER_UNIT_DONE = "unit_done"  # t1 → t2: all pages for one input unit are queued
IDENTIFIER_REGION = "region"
IDENTIFIER_DONE = "done"
