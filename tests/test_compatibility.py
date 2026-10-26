from modules.color_detection import DominantColor
from modules.compatibility import compare_palettes


def color(rgb: tuple[int, int, int], name: str) -> DominantColor:
    return DominantColor(rgb, "#000000", name, 1.0)


def test_neutral_pairing_is_explained() -> None:
    result = compare_palettes([color((20, 40, 100), "navy")], [color((220, 200, 160), "beige")])

    assert result.category == "Neutral pairing"
    assert result.score >= 85
    assert "heuristic" in result.explanation


def test_similar_hues_are_monochromatic() -> None:
    result = compare_palettes([color((210, 30, 30), "red")], [color((150, 25, 25), "red")])

    assert result.category == "Monochromatic"


def test_empty_palette_is_rejected() -> None:
    try:
        compare_palettes([], [color((0, 0, 0), "black")])
    except ValueError as error:
        assert "palettes" in str(error)
    else:
        raise AssertionError("Expected empty palette to be rejected")
