"""Color harmony scoring: complementary, analogous, triadic, split-complementary.

Harmony theory overview:
  - Monochromatic  : hue distance ≤ 30°  — same hue, different sat/val
  - Analogous      : hue distance 31–60°  — adjacent on color wheel
  - Triadic        : hue distance ~120°   — three equidistant hues
  - Split-comp     : hue distance ~150°   — split complementary
  - Complementary  : hue distance ~180°   — opposite on wheel
  - Neutral pairing: one side is achromatic → always high score
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.color_analysis import DominantColor, delta_e_76, is_neutral


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompatibilityResult:
    score: int                  # 0–100
    category: str               # harmony label
    explanation: str
    dominant_a_hex: str
    dominant_b_hex: str
    hue_distance_deg: float


# ---------------------------------------------------------------------------
# Hue-wheel helpers
# ---------------------------------------------------------------------------

def _hue_distance(h1: float, h2: float) -> float:
    """Circular hue distance in degrees (0–180)."""
    d = abs(h1 - h2)
    return min(d, 1.0 - d) * 360.0


def _relationship_score(dist: float) -> float:
    """Base harmony score in [0, 1] based on hue distance."""
    if dist <= 30:
        return 0.84        # monochromatic — safe
    if dist <= 60:
        return 0.78        # analogous — harmonious
    if 105 <= dist <= 135:
        return 0.75        # triadic — vibrant
    if 140 <= dist <= 160:
        return 0.80        # split-complementary — dynamic
    if dist >= 165:
        return 0.88        # complementary — bold contrast
    return 0.52            # random interval — clashing


def _harmony_label(dist: float, neutral: bool) -> str:
    if neutral:
        return "Neutral pairing"
    if dist <= 30:
        return "Monochromatic"
    if dist <= 60:
        return "Analogous"
    if 105 <= dist <= 135:
        return "Triadic"
    if 140 <= dist <= 160:
        return "Split-complementary"
    if dist >= 165:
        return "Complementary"
    return "Contrasting / mixed"


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compare_palettes(
    first: list[DominantColor],
    second: list[DominantColor],
) -> CompatibilityResult:
    """Compute a weighted heuristic compatibility score between two palettes.

    Algorithm:
    1. For every cross-product pair (color_a, color_b) compute:
       - weight = proportion_a × proportion_b
       - relationship_score based on hue distance
       - brightness balance penalty
       - saturation balance penalty
    2. Weighted sum / total weight → raw score
    3. Dominant-color pair determines harmony label
    4. Clamp and round to 0–100

    Args:
        first:  Palette for garment A (top/full).
        second: Palette for garment B (bottom).

    Returns:
        ``CompatibilityResult`` with score, category, explanation, and debug fields.
    """
    if not first or not second:
        raise ValueError("Both palettes must contain at least one color.")

    evidence: list[tuple[float, float, float]] = []
    for ca in first:
        ha, sa, va = ca.hsv
        for cb in second:
            hb, sb, vb = cb.hsv
            weight = ca.proportion * cb.proportion
            dist = _hue_distance(ha, hb)
            rel = _relationship_score(dist)
            if is_neutral(ca) or is_neutral(cb):
                rel = 0.96
            brightness_balance = 1.0 - min(abs(va - vb) * 0.15, 0.10)
            sat_balance = 1.0 - min(abs(sa - sb) * 0.15, 0.10)
            evidence.append((weight, rel, brightness_balance * sat_balance))

    total_w = sum(e[0] for e in evidence) or 1e-9
    raw = sum(w * r * b for w, r, b in evidence) / total_w
    score = round(max(0, min(100, raw * 100)))

    dom_a = first[0]
    dom_b = second[0]
    dom_dist = _hue_distance(dom_a.hsv[0], dom_b.hsv[0])
    neutral = is_neutral(dom_a) or is_neutral(dom_b)
    category = _harmony_label(dom_dist, neutral)
    de = delta_e_76(dom_a, dom_b)

    explanation = (
        f"Dominant colors are **{dom_a.name.title()}** ({dom_a.hex_code}) and "
        f"**{dom_b.name.title()}** ({dom_b.hex_code}). "
        f"Hue separation ≈ {round(dom_dist)}° → **{category}**. "
        f"Perceptual distance (ΔE) = {de:.1f}. "
        f"Weighted palette harmony score: **{score}/100**."
    )
    return CompatibilityResult(
        score=score,
        category=category,
        explanation=explanation,
        dominant_a_hex=dom_a.hex_code,
        dominant_b_hex=dom_b.hex_code,
        hue_distance_deg=dom_dist,
    )
