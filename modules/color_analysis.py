"""Color analysis: K-Means clustering, HSV/LAB conversion, color histograms.

This module is the core image processing engine.
Techniques used:
  - OpenCV HSV histogram for color distribution
  - K-Means clustering in RGB space for dominant color extraction
  - CIELAB color space for perceptually uniform color comparison
  - Nearest named-color matching via Euclidean distance
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Named color palette
# ---------------------------------------------------------------------------

NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (10, 10, 10),
    "white": (245, 245, 245),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "red": (210, 30, 30),
    "crimson": (185, 15, 50),
    "orange": (240, 130, 20),
    "amber": (255, 190, 0),
    "yellow": (235, 210, 30),
    "lime": (150, 220, 40),
    "green": (40, 150, 70),
    "teal": (0, 128, 128),
    "cyan": (0, 180, 200),
    "sky blue": (100, 180, 230),
    "blue": (35, 95, 190),
    "navy": (20, 45, 100),
    "indigo": (60, 30, 140),
    "purple": (125, 55, 170),
    "violet": (145, 50, 200),
    "magenta": (200, 0, 180),
    "pink": (225, 105, 155),
    "rose": (240, 80, 120),
    "brown": (130, 75, 35),
    "tan": (185, 140, 90),
    "beige": (210, 185, 140),
    "cream": (240, 230, 200),
    "khaki": (165, 155, 80),
    "olive": (100, 105, 30),
    "maroon": (110, 20, 30),
    "burgundy": (100, 0, 40),
    "lavender": (185, 160, 220),
    "mauve": (170, 115, 145),
    "coral": (240, 115, 90),
    "salmon": (245, 145, 125),
    "mint": (150, 220, 190),
    "turquoise": (50, 200, 185),
    "gold": (215, 180, 30),
    "charcoal": (55, 55, 60),
    "denim": (75, 115, 160),
    "rust": (185, 65, 25),
    "camel": (195, 155, 85),
}

MAX_CLUSTER_PIXELS = 30_000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DominantColor:
    """A dominant cluster color extracted from a garment region."""
    rgb: tuple[int, int, int]
    hex_code: str
    name: str
    proportion: float

    @property
    def lab(self) -> tuple[float, float, float]:
        return rgb_to_lab(self.rgb)

    @property
    def hsv(self) -> tuple[float, float, float]:
        r, g, b = (c / 255 for c in self.rgb)
        return colorsys.rgb_to_hsv(r, g, b)


@dataclass(frozen=True)
class ColorHistogram:
    """HSV hue histogram (36 bins, 10° each) for a garment region."""
    bins: np.ndarray        # shape (36,) normalized
    saturation_mean: float
    value_mean: float


# ---------------------------------------------------------------------------
# Color-space conversions
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert sRGB to CIELAB (D65 reference white)."""
    vals = []
    for channel in rgb:
        v = channel / 255.0
        vals.append(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92)
    r, g, b = vals
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722)
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    x, y, z = f(x), f(y), f(z)
    return (116 * y - 16, 500 * (x - y), 200 * (y - z))


def delta_e_76(a: DominantColor, b: DominantColor) -> float:
    """CIELAB Delta E 1976 perceptual distance."""
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a.lab, b.lab)))


def is_neutral(color: DominantColor) -> bool:
    """True if the color carries no meaningful hue signal.

    Threshold notes:
    - sat < 0.12 : genuinely achromatic (white, light gray, near-white)
    - val < 0.10 : very dark / near-black
    - Named neutrals override the sat/val check only for the exact achromatic
      family (white/gray/black/silver/charcoal/cream/beige).
      Brown, navy, camel ARE chromatic enough for color-wheel suggestions.
    """
    _, sat, val = color.hsv
    achromatic_names = {"black", "white", "gray", "silver", "charcoal", "cream", "beige"}
    return (
        color.name.lower() in achromatic_names
        or sat < 0.12
        or val < 0.10
    )


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    best_name = "unknown"
    best_dist = float("inf")
    for name, ref in NAMED_COLORS.items():
        dist = sum((c - r) ** 2 for c, r in zip(rgb, ref))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


# ---------------------------------------------------------------------------
# Masked pixel extraction
# ---------------------------------------------------------------------------

def _masked_pixels(image: Image.Image, mask: Image.Image) -> np.ndarray:
    if image.size != mask.size:
        raise ValueError("Image and mask must have the same dimensions.")
    pixels = np.asarray(image.convert("RGB"))
    selected = np.asarray(mask.convert("L")) > 0
    values = pixels[selected]
    if values.size == 0:
        raise ValueError("The garment mask contains no foreground pixels.")
    return values.astype(np.float32)


# ---------------------------------------------------------------------------
# K-Means dominant color extraction
# ---------------------------------------------------------------------------

def extract_dominant_colors(
    image: Image.Image,
    mask: Image.Image,
    cluster_count: int = 5,
    max_pixels: int = MAX_CLUSTER_PIXELS,
) -> list[DominantColor]:
    """Extract dominant RGB colors using K-Means clustering on masked pixels.

    Args:
        image: RGB source image.
        mask: L-mode mask (255 = foreground, 0 = background).
        cluster_count: Number of K-Means clusters (dominant colors).
        max_pixels: Cap on pixels sampled for clustering speed.

    Returns:
        List of ``DominantColor`` sorted by pixel proportion (most dominant first).
    """
    if cluster_count < 1:
        raise ValueError("cluster_count must be >= 1.")
    pixels = _masked_pixels(image, mask)

    if len(pixels) > max_pixels:
        indices = np.random.RandomState(42).choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[indices]

    unique = np.unique(pixels, axis=0)
    k = min(cluster_count, len(unique))

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(pixels)
    centers = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(int)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)

    results: list[DominantColor] = []
    for idx in order:
        rgb = tuple(int(v) for v in centers[idx])
        results.append(
            DominantColor(
                rgb=rgb,
                hex_code="#{:02X}{:02X}{:02X}".format(*rgb),
                name=_nearest_color_name(rgb),
                proportion=float(counts[idx] / len(labels)),
            )
        )
    return results


# ---------------------------------------------------------------------------
# HSV color histogram (for fast retrieval indexing)
# ---------------------------------------------------------------------------

def compute_hsv_histogram(
    image: Image.Image,
    mask: Image.Image | None = None,
    hue_bins: int = 36,
) -> ColorHistogram:
    """Compute a normalized HSV hue histogram for the (optionally masked) image.

    The hue is divided into ``hue_bins`` equal-width bins (default 36 × 10°).
    Pixels with saturation < 0.15 are excluded as neutrals (they have unreliable hue).

    Args:
        image: RGB PIL image.
        mask: Optional L-mode mask. If None, uses the full image.
        hue_bins: Number of hue histogram bins.

    Returns:
        ``ColorHistogram`` with normalized bin array, saturation mean, value mean.
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Normalize OpenCV HSV to 0–1
    img_hsv[:, :, 0] /= 180.0
    img_hsv[:, :, 1] /= 255.0
    img_hsv[:, :, 2] /= 255.0

    if mask is not None:
        fg = np.asarray(mask.convert("L")) > 0
        pixels = img_hsv[fg]
    else:
        pixels = img_hsv.reshape(-1, 3)

    # Exclude near-neutral pixels (low saturation) from hue histogram
    chromatic = pixels[pixels[:, 1] > 0.15]
    if len(chromatic) == 0:
        hist = np.zeros(hue_bins, dtype=np.float32)
    else:
        hues = chromatic[:, 0]
        hist, _ = np.histogram(hues, bins=hue_bins, range=(0.0, 1.0))
        total = hist.sum()
        hist = hist.astype(np.float32) / total if total > 0 else hist.astype(np.float32)

    sat_mean = float(pixels[:, 1].mean()) if len(pixels) > 0 else 0.0
    val_mean = float(pixels[:, 2].mean()) if len(pixels) > 0 else 0.0
    return ColorHistogram(bins=hist, saturation_mean=sat_mean, value_mean=val_mean)


def histogram_distance(a: ColorHistogram, b: ColorHistogram) -> float:
    """Bhattacharyya distance between two normalized hue histograms (lower = more similar)."""
    # Bhattacharyya coefficient
    bc = float(np.sum(np.sqrt(a.bins * b.bins)))
    bc = max(min(bc, 1.0), 1e-10)
    return -math.log(bc)


def palette_histogram_similarity(palette: list[DominantColor]) -> np.ndarray:
    """Convert a dominant-color palette to a 36-bin hue histogram for retrieval."""
    hist = np.zeros(36, dtype=np.float32)
    for color in palette:
        hue, sat, _ = color.hsv
        if sat < 0.15:
            continue
        bin_idx = int(hue * 36) % 36
        hist[bin_idx] += color.proportion
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
