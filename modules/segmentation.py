"""DeepFashion2 annotation parsing, garment classification, and mask generation.

DeepFashion2 category_id → garment type mapping:
  1  short_sleeve_top      → TOP
  2  long_sleeve_top       → TOP
  3  short_sleeve_outwear  → TOP
  4  long_sleeve_outwear   → TOP
  5  vest                  → TOP
  6  sling                 → TOP
  7  shorts                → BOTTOM
  8  trousers              → BOTTOM
  9  skirt                 → BOTTOM
  10 short_sleeve_dress    → FULL
  11 long_sleeve_dress     → FULL
  12 vest_dress            → FULL
  13 sling_dress           → FULL
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET_ROOT = Path(r"E:\DeepFashion2\deepfashion2_original_images")
DEFAULT_IMAGE_LIMIT = 2_000


class WearType(str, Enum):
    TOP = "Top wear"
    BOTTOM = "Bottom wear"
    FULL = "Full / dress"
    UNKNOWN = "Unknown"


# Maps category_id → (human-readable name, WearType)
CATEGORY_MAP: dict[int, tuple[str, WearType]] = {
    1: ("Short-sleeve top", WearType.TOP),
    2: ("Long-sleeve top", WearType.TOP),
    3: ("Short-sleeve outwear", WearType.TOP),
    4: ("Long-sleeve outwear", WearType.TOP),
    5: ("Vest", WearType.TOP),
    6: ("Sling / cami top", WearType.TOP),
    7: ("Shorts", WearType.BOTTOM),
    8: ("Trousers", WearType.BOTTOM),
    9: ("Skirt", WearType.BOTTOM),
    10: ("Short-sleeve dress", WearType.FULL),
    11: ("Long-sleeve dress", WearType.FULL),
    12: ("Vest dress", WearType.FULL),
    13: ("Sling dress", WearType.FULL),
}

# DeepFashion2 style compatibility pairs (used for retrieval filtering)
TOP_CATEGORIES = {k for k, (_, wt) in CATEGORY_MAP.items() if wt == WearType.TOP}
BOTTOM_CATEGORIES = {k for k, (_, wt) in CATEGORY_MAP.items() if wt == WearType.BOTTOM}
FULL_CATEGORIES = {k for k, (_, wt) in CATEGORY_MAP.items() if wt == WearType.FULL}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GarmentItem:
    """One annotated garment from a DeepFashion2 annotation file."""
    item_key: str
    category_id: int
    category_name: str
    wear_type: WearType
    bbox: tuple[int, int, int, int]          # x, y, w, h (pixel coords)
    segmentation: list[list[float]]          # list of polygon point-lists [x0,y0,x1,y1,...]
    style: int                               # DF2 style attribute
    occlusion: int                           # 0-2 (none, partial, heavy)


@dataclass
class ParsedAnnotation:
    """All garment items from one DF2 JSON annotation file."""
    source: str
    items: list[GarmentItem]

    @property
    def has_top(self) -> bool:
        return any(i.wear_type == WearType.TOP for i in self.items)

    @property
    def has_bottom(self) -> bool:
        return any(i.wear_type == WearType.BOTTOM for i in self.items)

    @property
    def has_full(self) -> bool:
        return any(i.wear_type == WearType.FULL for i in self.items)

    def items_of_type(self, wear_type: WearType) -> list[GarmentItem]:
        return [i for i in self.items if i.wear_type == wear_type]

    @property
    def overall_wear_type(self) -> WearType:
        """Infer the dominant wear context for the whole image."""
        if self.has_full:
            return WearType.FULL
        if self.has_top and self.has_bottom:
            return WearType.FULL        # outfit with both pieces
        if self.has_top:
            return WearType.TOP
        if self.has_bottom:
            return WearType.BOTTOM
        return WearType.UNKNOWN


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def training_image_directory(dataset_root: Path) -> Path:
    return dataset_root / "train" / "image"


def annotation_directory(dataset_root: Path) -> Path:
    return dataset_root / "train" / "annos"


def list_training_images(
    dataset_root: Path, limit: int | None = DEFAULT_IMAGE_LIMIT
) -> list[Path]:
    """Return matched training JPEG paths (image + annotation), optionally capped."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None.")

    image_dir = training_image_directory(dataset_root)
    anno_dir = annotation_directory(dataset_root)
    if not image_dir.is_dir() or not anno_dir.is_dir():
        raise FileNotFoundError(
            f"Expected train\\image and train\\annos under {dataset_root}."
        )

    image_paths = sorted(
        p for p in image_dir.glob("*.jpg")
        if (anno_dir / f"{p.stem}.json").is_file()
    )
    return image_paths if limit is None else image_paths[:limit]


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_bbox(raw: list) -> tuple[int, int, int, int]:
    """Convert DF2 bbox [x1, y1, x2, y2] → (x, y, w, h)."""
    x1, y1, x2, y2 = (int(round(v)) for v in raw[:4])
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def load_annotation(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read annotation: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def parse_annotation(path: Path) -> ParsedAnnotation:
    """Parse a DeepFashion2 JSON annotation into a structured ``ParsedAnnotation``."""
    raw = load_annotation(path)
    items: list[GarmentItem] = []

    for key, value in raw.items():
        if not key.startswith("item") or not isinstance(value, dict):
            continue

        cat_id = int(value.get("category_id", 0))
        if cat_id not in CATEGORY_MAP:
            continue

        cat_name, wear_type = CATEGORY_MAP[cat_id]
        bbox_raw = value.get("bounding_box", [])
        seg_raw = value.get("segmentation", [])
        if not bbox_raw or len(bbox_raw) < 4:
            continue

        items.append(
            GarmentItem(
                item_key=key,
                category_id=cat_id,
                category_name=cat_name,
                wear_type=wear_type,
                bbox=_parse_bbox(bbox_raw),
                segmentation=[s for s in seg_raw if isinstance(s, list) and len(s) >= 6],
                style=int(value.get("style", 0)),
                occlusion=int(value.get("occlusion", 0)),
            )
        )

    return ParsedAnnotation(source=path.name, items=items)


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def polygon_mask_for_item(image_size: tuple[int, int], item: GarmentItem) -> Image.Image:
    """Rasterize one garment item's segmentation polygons into an L-mode mask."""
    mask = Image.new("L", image_size, 0)
    if not item.segmentation:
        # Fall back to bounding box
        x, y, w, h = item.bbox
        from PIL import ImageDraw as _ID
        _ID.Draw(mask).rectangle([x, y, x + w, y + h], fill=255)
        return mask

    draw = ImageDraw.Draw(mask)
    for polygon in item.segmentation:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        draw.polygon(points, fill=255)
    return mask


def polygon_mask_for_types(
    image_size: tuple[int, int],
    annotation: ParsedAnnotation,
    wear_types: set[WearType],
) -> Image.Image:
    """Union mask for all items matching the given wear types."""
    mask = Image.new("L", image_size, 0)
    for item in annotation.items:
        if item.wear_type in wear_types:
            item_mask = polygon_mask_for_item(image_size, item)
            mask = Image.fromarray(
                np.maximum(np.array(mask), np.array(item_mask))
            )
    return mask


def isolate_garment(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Return RGBA image with background transparent outside ``mask``."""
    if image.size != mask.size:
        raise ValueError("Image and mask must be the same size.")
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def mask_coverage(mask: Image.Image) -> float:
    """Fraction of pixels covered by the mask."""
    arr = np.asarray(mask.convert("L"))
    return float(np.count_nonzero(arr) / arr.size)


# ---------------------------------------------------------------------------
# Training sample loader
# ---------------------------------------------------------------------------

def load_training_sample(
    dataset_root: Path, image_path: Path
) -> tuple[Image.Image, ParsedAnnotation]:
    """Load a DF2 training image and its parsed annotation."""
    if image_path.parent != training_image_directory(dataset_root):
        raise ValueError("Image is not inside the configured train\\image folder.")
    image = Image.open(image_path).convert("RGB")
    anno_path = annotation_directory(dataset_root) / f"{image_path.stem}.json"
    annotation = parse_annotation(anno_path)
    return image, annotation
