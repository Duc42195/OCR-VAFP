"""Chandra OCR - Layout-aware vision language model for document processing."""

from chandra.core.model import InferenceManager, BatchInputItem, BatchOutputItem
from chandra.core.input import load_file, load_image, load_pdf_images
from chandra.core.output import (
    parse_markdown,
    parse_html,
    parse_layout,
    parse_chunks,
    draw_layout,
    generate_qr,
    generate_viet_qr,
)

__all__ = [
    "InferenceManager",
    "BatchInputItem", 
    "BatchOutputItem",
    "load_file",
    "load_image",
    "load_pdf_images",
    "parse_markdown",
    "parse_html",
    "parse_layout",
    "parse_chunks",
    "draw_layout",
    "generate_qr",
    "generate_viet_qr",
]
