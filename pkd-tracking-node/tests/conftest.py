import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def create_image():
    def _create_image(size):
        img = np.random.randint(255, size=size, dtype=np.uint8)
        return img

    return _create_image
