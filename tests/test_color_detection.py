from PIL import Image

from modules.color_detection import extract_dominant_colors


def test_extract_dominant_colors_ignores_background() -> None:
    image = Image.new("RGB", (10, 10), "white")
    for x in range(5):
        for y in range(10):
            image.putpixel((x, y), (220, 20, 20))
    mask = Image.new("L", (10, 10), 0)
    for x in range(5):
        for y in range(10):
            mask.putpixel((x, y), 255)

    colors = extract_dominant_colors(image, mask, cluster_count=2)

    assert colors[0].rgb == (220, 20, 20)
    assert colors[0].proportion == 1.0
    assert colors[0].hex_code == "#DC1414"


def test_extract_dominant_colors_reduces_clusters_for_uniform_mask() -> None:
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    mask = Image.new("L", (4, 4), 255)

    colors = extract_dominant_colors(image, mask, cluster_count=5)

    assert len(colors) == 1
    assert colors[0].proportion == 1.0


def test_extract_dominant_colors_rejects_empty_mask() -> None:
    image = Image.new("RGB", (2, 2), "red")
    mask = Image.new("L", (2, 2), 0)

    try:
        extract_dominant_colors(image, mask)
    except ValueError as error:
        assert "no pixels" in str(error)
    else:
        raise AssertionError("Expected an empty mask to be rejected")
