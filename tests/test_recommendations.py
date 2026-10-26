from modules.color_detection import DominantColor
from modules.recommendations import recommend_colors


def test_recommendations_are_derived_for_colored_palette() -> None:
    recommendations = recommend_colors(
        [DominantColor((220, 30, 30), "#DC1E1E", "red", 1.0)]
    )

    assert len(recommendations) == 3
    assert all(item.hex_code.startswith("#") for item in recommendations)


def test_recommendations_handle_neutral_palette() -> None:
    recommendations = recommend_colors(
        [DominantColor((20, 20, 20), "#141414", "black", 1.0)]
    )

    assert recommendations[0].label == "Soft blue"
