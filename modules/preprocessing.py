"""Image validation, loading, and resizing helpers."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG"}


def load_image(file: bytes | BinaryIO) -> Image.Image:
    """Load an uploaded image as RGB and reject unsupported or oversized input."""
    if hasattr(file, "read"):
        raw = file.read()
    else:
        raw = file

    if not isinstance(raw, bytes):
        raise TypeError("Image input must be bytes or a binary file.")
    if not raw:
        raise ValueError("The uploaded file is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("The image must be smaller than 10 MB.")

    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError("Only JPEG and PNG images are supported.")
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def resize_for_analysis(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Return a copy whose longest side is at most ``max_size`` pixels."""
    if max_size < 1:
        raise ValueError("max_size must be positive.")
    if image.width <= max_size and image.height <= max_size:
        return image.copy()

    scale = max_size / max(image.width, image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)
