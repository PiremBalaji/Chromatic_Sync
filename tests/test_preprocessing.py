from io import BytesIO

import pytest
from PIL import Image

from modules.preprocessing import crop_image, load_image, resize_for_analysis


def image_bytes(image_format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGBA", (20, 10), (12, 34, 56, 128)).save(output, format=image_format)
    return output.getvalue()


def test_load_image_converts_to_rgb() -> None:
    image = load_image(image_bytes())

    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_load_image_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError, match="valid image"):
        load_image(b"not an image")


def test_resize_for_analysis_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (200, 100))

    resized = resize_for_analysis(image, max_size=100)

    assert resized.size == (100, 50)


def test_resize_for_analysis_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        resize_for_analysis(Image.new("RGB", (10, 10)), max_size=0)


def test_crop_image_uses_pixel_bounds() -> None:
    image = Image.new("RGB", (20, 10), "red")

    cropped = crop_image(image, 2, 1, 18, 9)

    assert cropped.size == (16, 8)
