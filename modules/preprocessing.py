"""Image loading, validation, resizing, GrabCut background removal, and
garment type inference from mask geometry.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from modules.segmentation import WearType

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
ANALYSIS_MAX_SIDE = 1024


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------

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
        raise ValueError("The image must be smaller than 15 MB.")

    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format '{image.format}'. Use JPEG, PNG, or WebP.")
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def resize_for_analysis(image: Image.Image, max_side: int = ANALYSIS_MAX_SIDE) -> Image.Image:
    """Return a copy whose longest side is at most ``max_side`` pixels."""
    if max_side < 1:
        raise ValueError("max_side must be positive.")
    if image.width <= max_side and image.height <= max_side:
        return image.copy()
    scale = max_side / max(image.width, image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# GrabCut background removal
# ---------------------------------------------------------------------------

def grabcut_foreground(
    image: Image.Image,
    iterations: int = 5,
    margin_fraction: float = 0.05,
) -> Image.Image:
    """Remove the background using OpenCV GrabCut and return an RGBA image.

    The initial rectangle is inset by ``margin_fraction`` on each side, which
    is a good heuristic for garment images where the subject occupies most of
    the frame.

    Args:
        image: RGB PIL image.
        iterations: Number of GrabCut iterations (more = slower but cleaner).
        margin_fraction: Fractional margin from each edge for the seed rect.

    Returns:
        RGBA PIL image with background pixels made transparent.
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    margin_x = max(1, round(w * margin_fraction))
    margin_y = max(1, round(h * margin_fraction))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask_gc = np.zeros((h, w), np.uint8)

    cv2.grabCut(img_array, mask_gc, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Pixels marked as probable/definite foreground
    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    result = image.convert("RGBA")
    result.putalpha(Image.fromarray(fg_mask))
    return result


def grabcut_mask_from_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    iterations: int = 5,
) -> Image.Image:
    """GrabCut using a provided bounding box (x, y, w, h) as the seed rectangle.

    Returns an L-mode mask (255 = foreground, 0 = background).
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mask_gc = np.zeros((h, w), np.uint8)

    bx, by, bw, bh = bbox
    bx = max(0, bx)
    by = max(0, by)
    bw = min(bw, w - bx)
    bh = min(bh, h - by)

    if bw < 2 or bh < 2:
        # Fallback: white mask over whole image
        return Image.new("L", image.size, 255)

    cv2.grabCut(img_array, mask_gc, (bx, by, bw, bh), bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return Image.fromarray(fg_mask)


def crop_to_bbox(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop image to bounding box given as (x, y, w, h)."""
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


# ---------------------------------------------------------------------------
# Garment type inference from mask geometry
# ---------------------------------------------------------------------------

def classify_wear_type_from_mask(mask: Image.Image) -> "WearType":
    """Infer whether an uploaded garment is TOP, BOTTOM, or FULL using the
    GrabCut foreground mask bounding-box geometry.

    Heuristic rules (applied in order):
    1. Foreground height-coverage > 68 % of image height  → FULL / dress
    2. Centroid Y in top 42 % of image                    → TOP
    3. Centroid Y in bottom 52 % of image                 → BOTTOM
    4. Aspect ratio (fg_w / fg_h) > 1.6 (wide, short)    → TOP
    5. Aspect ratio < 0.6 (tall, narrow)                  → BOTTOM
    6. Otherwise                                          → UNKNOWN

    These thresholds handle the most common garment-photo compositions where
    the garment is centred and fills most of the frame.

    Args:
        mask: L-mode or RGBA PIL image (the GrabCut result).

    Returns:
        One of WearType.TOP, BOTTOM, FULL, or UNKNOWN.
    """
    # Import here to avoid circular imports at module load time
    from modules.segmentation import WearType

    arr = np.array(mask.convert("L"))
    h_img, w_img = arr.shape

    # Remove isolated GrabCut noise and fill small holes before measuring the
    # silhouette. Geometry from a fragmented mask is not meaningful.
    binary = np.where(arr > 0, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if component_count > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Bounding box of the cleaned, largest foreground component.
    rows_fg = np.any(binary > 0, axis=1)
    cols_fg = np.any(binary > 0, axis=0)

    if not np.any(rows_fg):
        return WearType.UNKNOWN

    top_row    = int(np.argmax(rows_fg))
    bottom_row = h_img - 1 - int(np.argmax(rows_fg[::-1]))
    left_col   = int(np.argmax(cols_fg))
    right_col  = w_img - 1 - int(np.argmax(cols_fg[::-1]))

    fg_h = bottom_row - top_row + 1
    fg_w = right_col - left_col + 1
    centroid_y = (top_row + bottom_row) / 2.0

    height_coverage = fg_h / h_img
    centroid_frac   = centroid_y / h_img
    aspect_ratio    = fg_w / max(fg_h, 1)

    if height_coverage > 0.68:
        return WearType.FULL
    if centroid_frac < 0.42:
        return WearType.TOP
    if centroid_frac > 0.52:
        return WearType.BOTTOM
    if aspect_ratio > 1.6:
        return WearType.TOP
    if aspect_ratio < 0.6:
        return WearType.BOTTOM
    return WearType.UNKNOWN
