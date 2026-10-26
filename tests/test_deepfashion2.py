from PIL import Image

from modules.deepfashion2 import create_garment_mask, isolate_garment


def test_create_garment_mask_rasterizes_polygons() -> None:
    annotation = {"item1": {"segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]]}}

    mask = create_garment_mask((10, 10), annotation)

    assert mask.getpixel((4, 4)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_create_garment_mask_rejects_missing_polygons() -> None:
    try:
        create_garment_mask((10, 10), {"item1": {}})
    except ValueError as error:
        assert "polygons" in str(error)
    else:
        raise AssertionError("Expected missing polygons to be rejected")


def test_isolate_garment_applies_transparency() -> None:
    image = Image.new("RGB", (2, 2), "red")
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((0, 0), 255)

    isolated = isolate_garment(image, mask)

    assert isolated.mode == "RGBA"
    assert isolated.getpixel((0, 0))[3] == 255
    assert isolated.getpixel((1, 1))[3] == 0
