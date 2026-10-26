"""DeepFashion2 image retrieval by color similarity.

Pipeline:
1. Index: scan dataset annotations → extract HSV histogram per image per wear-type.
   Cache index to a .npz file for fast subsequent loads.
2. Query: given an input palette and desired WearType, find nearest neighbors
   by Bhattacharyya histogram distance.
3. Return ranked list of (image_path, score) tuples for display.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from modules.color_analysis import (
    ColorHistogram,
    compute_hsv_histogram,
    histogram_distance,
    palette_histogram_similarity,
    DominantColor,
)
from modules.segmentation import (
    WearType,
    CATEGORY_MAP,
    TOP_CATEGORIES,
    BOTTOM_CATEGORIES,
    FULL_CATEGORIES,
    parse_annotation,
    annotation_directory,
    polygon_mask_for_item,
)

CACHE_DIR = Path.home() / ".chromatic_sync_cache"
MAX_INDEX_IMAGES = 2_000
TOP_K = 8  # number of results to return


# ---------------------------------------------------------------------------
# Index record
# ---------------------------------------------------------------------------

class _IndexRecord:
    __slots__ = ("image_path", "wear_type", "category_id", "hist")

    def __init__(self, image_path: str, wear_type: str, category_id: int, hist: np.ndarray):
        self.image_path = image_path
        self.wear_type = wear_type
        self.category_id = category_id
        self.hist = hist


# ---------------------------------------------------------------------------
# Build / load index
# ---------------------------------------------------------------------------

def _cache_key(dataset_root: Path, limit: int) -> str:
    key = f"{dataset_root}:{limit}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(dataset_root: Path, limit: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"index_{_cache_key(dataset_root, limit)}.pkl"


# Thumbnail size for fast indexing (color histogram needs only coarse resolution)
_INDEX_THUMB_SIZE = (128, 128)


def _build_index(
    dataset_root: Path,
    image_paths: list[Path],
    progress_callback=None,
) -> list[_IndexRecord]:
    """Build a histogram index for all images.

    Images are downsampled to 128×128 thumbnails before histogram extraction —
    this is ~10× faster than full resolution with negligible impact on color
    retrieval accuracy.

    Slow only on first run; subsequent loads are instant from pickle cache.
    """
    anno_dir = annotation_directory(dataset_root)
    records: list[_IndexRecord] = []

    for i, img_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(i, len(image_paths), img_path.name)

        anno_path = anno_dir / f"{img_path.stem}.json"
        if not anno_path.exists():
            continue
        try:
            annotation = parse_annotation(anno_path)
            # Open at full resolution for mask rasterisation, then downscale
            image_full = Image.open(img_path).convert("RGB")
            image_small = image_full.resize(_INDEX_THUMB_SIZE, Image.Resampling.BILINEAR)
        except Exception:
            continue

        orig_w, orig_h = image_full.size
        tw, th = _INDEX_THUMB_SIZE
        scale_x = tw / orig_w
        scale_y = th / orig_h

        for item in annotation.items:
            try:
                # Rasterise mask at full size, then downscale to thumbnail
                mask_full = polygon_mask_for_item(image_full.size, item)
                mask_small = mask_full.resize(_INDEX_THUMB_SIZE, Image.Resampling.NEAREST)
                hist = compute_hsv_histogram(image_small, mask_small)
                records.append(
                    _IndexRecord(
                        image_path=str(img_path),
                        wear_type=item.wear_type.value,
                        category_id=item.category_id,
                        hist=hist.bins,
                    )
                )
            except Exception:
                continue

    return records


def load_or_build_index(
    dataset_root: Path,
    image_paths: list[Path],
    force_rebuild: bool = False,
    progress_callback=None,
) -> list[_IndexRecord]:
    """Return the histogram index, building and caching it if necessary."""
    cache = _cache_path(dataset_root, len(image_paths))
    if not force_rebuild and cache.exists():
        try:
            with cache.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # corrupt cache — rebuild

    records = _build_index(dataset_root, image_paths, progress_callback)
    with cache.open("wb") as f:
        pickle.dump(records, f)
    return records


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def _query_wear_types(query_type: WearType) -> set[str]:
    """What types of garments should we retrieve to complement the query?"""
    if query_type == WearType.TOP:
        return {WearType.BOTTOM.value}
    if query_type == WearType.BOTTOM:
        return {WearType.TOP.value}
    # FULL / UNKNOWN → find complementary full outfits and both types
    return {WearType.TOP.value, WearType.BOTTOM.value, WearType.FULL.value}


def retrieve_compatible(
    palette: list[DominantColor],
    query_wear_type: WearType,
    index: list[_IndexRecord],
    top_k: int = TOP_K,
) -> list[tuple[Path, float, str, int]]:
    """Find dataset images whose color histogram is most compatible with ``palette``.

    Compatibility is defined as *low Bhattacharyya distance* between the query
    palette's hue histogram and the indexed image's hue histogram.  We return
    images of the *complementary* wear type (top → bottom, etc.).

    Returns:
        List of (image_path, distance_score, wear_type_str, category_id) sorted
        ascending by distance (most compatible first).
    """
    query_hist_arr = palette_histogram_similarity(palette)
    target_types = _query_wear_types(query_wear_type)

    candidates: list[tuple[float, _IndexRecord]] = []
    for record in index:
        if record.wear_type not in target_types:
            continue
        rec_hist = ColorHistogram(bins=record.hist, saturation_mean=0, value_mean=0)
        query_hist = ColorHistogram(bins=query_hist_arr, saturation_mean=0, value_mean=0)
        dist = histogram_distance(query_hist, rec_hist)
        candidates.append((dist, record))

    candidates.sort(key=lambda x: x[0])
    results = []
    seen_paths: set[str] = set()
    for dist, rec in candidates:
        if rec.image_path in seen_paths:
            continue
        seen_paths.add(rec.image_path)
        results.append((Path(rec.image_path), dist, rec.wear_type, rec.category_id))
        if len(results) >= top_k:
            break

    return results
