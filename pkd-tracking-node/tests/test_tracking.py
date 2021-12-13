from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from custom_nodes.dabble.tracking import Node

FORTESTS_DIR = Path(__file__).resolve().parents[2] / "fortests"
SIZE = (400, 600, 3)


@pytest.fixture(name="tracking_config")
def fixture_tracking_config():
    filepath = Path(__file__).resolve().parent / "test_tracking.yml"
    with open(filepath) as file:
        node_config = yaml.safe_load(file)

    return node_config


@pytest.fixture(name="tracker", params=["iou", "mosse"])
def fixture_tracker(request, tracking_config):
    tracking_config["tracking_type"] = request.param
    node = Node(tracking_config)
    return node


@pytest.fixture(name="two_people_crossing_sequence")
def fixture_two_people_crossing_sequence():
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


class TestTracking:
    def test_no_tags(self, create_image, tracker):
        img1 = create_image(SIZE)

        inputs = {
            "img": img1,
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        outputs = tracker.run(inputs)

        assert not outputs["obj_tags"]

    def test_multi_tags(self, tracker, two_people_crossing_sequence):
        prev_tags = None
        for i, inputs in enumerate(two_people_crossing_sequence):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            if i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]
