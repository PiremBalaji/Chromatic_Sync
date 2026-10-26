from pathlib import Path

import pytest

from modules.deepfashion2 import list_training_images


def test_list_training_images_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        list_training_images(tmp_path, limit=0)
