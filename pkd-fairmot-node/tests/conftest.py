import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import tensorflow.keras.backend as K
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[3] / "PeekingDuck"))

FORTESTS_DIR = Path(__file__).resolve().parents[2] / "fortests"


HUMAN_IMAGES = ["t1.jpg", "t2.jpg", "t4.jpg"]
NO_HUMAN_IMAGES = ["black.jpg", "t3.jpg"]
HUMAN_VIDEOS = ["humans_mot.mp4"]


@pytest.fixture(params=HUMAN_IMAGES)
def human_images(request):
    """Yields images with humans in them."""
    test_img_dir = FORTESTS_DIR / "images"

    yield str(test_img_dir / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture(params=NO_HUMAN_IMAGES)
def no_human_images(request):
    """Yields images without humans in them."""
    test_img_dir = FORTESTS_DIR / "images"

    yield str(test_img_dir / request.param)
    K.clear_session()
    gc.collect()


@pytest.fixture
def two_people_seq():
    sequence_dir = FORTESTS_DIR / "video_sequences" / "two_people_crossing"
    with open(sequence_dir / "detections.yml") as infile:
        detections = yaml.safe_load(infile.read())
    return [
        {
            "img": cv2.imread(str(sequence_dir / f"{key}.jpg")),
            "bboxes": np.array(val["bboxes"]),
            "bbox_scores": np.array(val["bbox_scores"]),
        }
        for key, val in detections.items()
    ]
