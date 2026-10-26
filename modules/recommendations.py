"""Color swatch recommendations derived from the input palette.

Strategy:
- Find the most chromatic (highest saturation × proportion) color in the palette as the base.
- If that base is still neutral (achromatic), branch to neutral-aware suggestions
  that vary based on the color's perceived warmth (from the Lab a* channel).
- For chromatic bases, generate complementary, analogous ±30°, triadic, and
  split-complementary swatches — all computed from the actual dominant hue so
  every garment produces a unique set of recommendations.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

from modules.color_analysis import DominantColor, is_neutral, rgb_to_lab


@dataclass(frozen=True)
class ColorRecommendation:
    """A suggested coordinating color."""
    rgb: tuple[int, int, int]
    hex_code: str
    label: str
    reason: str
    harmony_type: str  # complementary / analogous / triadic / split-comp / neutral


def _from_hsv(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    rgb_f = colorsys.hsv_to_rgb(hue % 1.0,
                                 max(0.0, min(1.0, saturation)),
                                 max(0.0, min(1.0, value)))
    return tuple(round(c * 255) for c in rgb_f)


def _make(label: str, rgb: tuple, reason: str, harmony_type: str) -> ColorRecommendation:
    return ColorRecommendation(
        rgb=rgb,
        hex_code="#{:02X}{:02X}{:02X}".format(*rgb),
        label=label,
        reason=reason,
        harmony_type=harmony_type,
    )


def _best_chromatic(palette: list[DominantColor]) -> DominantColor:
    """Return the palette entry with the highest saturation × proportion score.

    This avoids always picking palette[0] (most-common pixel cluster) which is
    often a background grey or a muted shadow tone after GrabCut.
    """
    meaningful = [color for color in palette if color.proportion >= 0.05]
    candidates = meaningful or palette
    return max(
        candidates,
        key=lambda c: c.hsv[1] * c.proportion * max(c.hsv[2], 0.2),
    )


def _warmth(color: DominantColor) -> float:
    """CIELAB a* channel: positive = warm/red, negative = cool/green."""
    _, a, _ = rgb_to_lab(color.rgb)
    return a


def recommend_colors(palette: list[DominantColor]) -> list[ColorRecommendation]:
    """Generate 5 coordinating color suggestions driven by the actual garment color.

    Algorithm:
    1. Pick the most chromatic palette entry (highest sat × proportion).
    2. If chromatic → generate color-wheel rotations unique to its hue.
    3. If truly neutral (sat < 0.12) → generate warmth-adaptive suggestions
       that are still based on the color's luminosity and lab warmth,
       so dark neutrals and light neutrals get different outputs.
    """
    if not palette:
        raise ValueError("Palette must not be empty.")

    base = _best_chromatic(palette)
    hue, sat, val = base.hsv

    # ── Chromatic path ───────────────────────────────────────────────────
    if not is_neutral(base):
        # Ensure suggestions have reasonable saturation even if base is slightly muted
        out_sat = max(sat, 0.55)
        out_val = max(val, 0.55)

        hue_deg = round(hue * 360)
        comp_hue = hue + 0.5
        split1 = hue + 5 / 12   # 150°
        split2 = hue - 5 / 12   # -150°
        tri    = hue + 1 / 3    # 120°
        ana_p  = hue + 1 / 12   # +30°
        ana_m  = hue - 1 / 12   # -30°

        return [
            _make(
                f"Complementary (~{(hue_deg + 180) % 360}deg)",
                _from_hsv(comp_hue, out_sat * 0.90, min(1.0, out_val * 1.05)),
                "Exact opposite hue on the color wheel -- bold, high-contrast pairing.",
                "complementary",
            ),
            _make(
                f"Analogous +30deg (~{(hue_deg + 30) % 360}deg)",
                _from_hsv(ana_p, out_sat * 0.88, out_val),
                "Neighbor hue -- cohesive, tonal outfit harmony.",
                "analogous",
            ),
            _make(
                f"Analogous -30deg (~{(hue_deg - 30) % 360}deg)",
                _from_hsv(ana_m, out_sat * 0.88, out_val),
                "Other neighbor hue -- equally harmonious but with a different mood.",
                "analogous",
            ),
            _make(
                f"Triadic (~{(hue_deg + 120) % 360}deg)",
                _from_hsv(tri, out_sat * 0.80, out_val),
                "120deg apart -- vibrant three-way balance.",
                "triadic",
            ),
            _make(
                f"Split-Comp (~{(hue_deg + 150) % 360}deg)",
                _from_hsv(split1, out_sat * 0.85, out_val),
                "Softer contrast than full complementary -- dynamic yet wearable.",
                "split-comp",
            ),
        ]

    # ── Neutral / achromatic path ────────────────────────────────────────
    # Vary suggestions by perceived warmth (Lab a*) and luminosity (val)
    warmth = _warmth(base)
    dark = val < 0.35   # dark neutral (charcoal, dark navy, black)
    light = val > 0.75  # light neutral (white, cream, light gray)

    if warmth > 5:          # warm neutral (camel, tan, beige, off-white)
        accent1 = _make("Terracotta",    (195, 90, 65),   "Earthy warm contrast for a warm neutral.",  "complementary")
        accent2 = _make("Dusty Sage",    (130, 165, 120), "Cool green earthy tone — grounded pairing.", "analogous")
        accent3 = _make("Deep Burgundy", (120, 30, 55),   "Rich warm accent that elevates neutrals.",   "triadic")
    elif warmth < -5:       # cool neutral (gray, silver, cool white)
        accent1 = _make("Slate Blue",    (80, 110, 165),  "Cool-toned complement for a cool neutral.", "complementary")
        accent2 = _make("Soft Lavender", (175, 155, 210), "Gentle hue lift for cool/gray garments.",   "analogous")
        accent3 = _make("Deep Teal",     (20, 120, 115),  "Bold cool contrast — confident pairing.",   "triadic")
    else:                   # true neutral (pure gray, pure white, pure black)
        accent1 = _make("Cobalt Blue",   (30, 80, 180),   "A classic chromatic pop on any neutral.",   "complementary")
        accent2 = _make("Forest Green",  (40, 115, 65),   "Natural, earthy contrast for true neutrals.", "analogous")
        accent3 = _make("Warm Coral",    (230, 105, 80),  "Vibrant warmth against a neutral base.",    "triadic")

    # Light vs dark neutrals pair differently for tonal options
    if dark:
        tonal1 = _make("Charcoal",  (60, 62, 65),    "Dark tonal layering — monochromatic depth.",  "neutral")
        tonal2 = _make("Warm White", (245, 240, 228), "Maximum light-dark contrast for a dark neutral.", "neutral")
    elif light:
        tonal1 = _make("Ivory",     (240, 234, 210), "Tonal depth — barely-there warmth.",          "neutral")
        tonal2 = _make("Light Gray", (190, 191, 196), "Minimal tonal shift for layered neutrals.",   "neutral")
    else:
        tonal1 = _make("Mid Gray",  (130, 132, 138), "Tonal mid-point for neutral layering.",       "neutral")
        tonal2 = _make("Sand",      (205, 185, 145), "Warm earthy neutral for understated style.",  "neutral")

    return [accent1, accent2, accent3, tonal1, tonal2]
