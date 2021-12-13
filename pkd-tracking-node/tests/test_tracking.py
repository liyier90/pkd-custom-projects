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


@pytest.fixture(name="two_people_seq")
def fixture_two_people_seq():
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
    def test_should_raise_for_invalid_tracking_type(self, tracking_config):
        tracking_config["tracking_type"] = "invalid type"
        with pytest.raises(ValueError) as excinfo:
            _ = Node(tracking_config)
        assert str(excinfo.value) == "tracker_type must be one of ['iou', 'mosse']"

    def test_no_tags(self, create_image, tracker):
        img1 = create_image(SIZE)

        inputs = {
            "img": img1,
            "bboxes": np.empty((0, 4), dtype=np.float32),
            "bbox_scores": np.empty((0), dtype=np.float32),
        }
        outputs = tracker.run(inputs)

        assert not outputs["obj_tags"]

    def test_tracking_ids_should_be_consistent_across_frames(
        self, tracker, two_people_seq
    ):
        prev_tags = []
        for i, inputs in enumerate(two_people_seq):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            if i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]

    def test_should_track_new_detection(self, tracker, two_people_seq):
        seq_idx = 4
        # Add a new detection at the specified seq_idx
        two_people_seq[seq_idx]["bboxes"] = np.append(
            two_people_seq[seq_idx]["bboxes"],
            [[0.1, 0.2, 0.3, 0.4]],
            axis=0,
        )
        two_people_seq[seq_idx]["bbox_scores"] = np.append(
            two_people_seq[seq_idx]["bbox_scores"], [0.4], axis=0
        )
        prev_tags = []
        for i, inputs in enumerate(two_people_seq):
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == seq_idx:
                assert outputs["obj_tags"] == prev_tags + ["2"]
            elif i == seq_idx + 1:
                assert outputs["obj_tags"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]

    def test_should_remove_lost_tracks(self, tracking_config, two_people_seq):
        """This only applies to IOU Tracker.

        NOTE: We are manually making a track to be lost since we don't
        have enough frames for it to occur naturally.
        """
        seq_idx = 4
        # Add a new detection at the specified seq_idx
        two_people_seq[seq_idx]["bboxes"] = np.append(
            two_people_seq[seq_idx]["bboxes"],
            [[0.1, 0.2, 0.3, 0.4]],
            axis=0,
        )
        two_people_seq[seq_idx]["bbox_scores"] = np.append(
            two_people_seq[seq_idx]["bbox_scores"], [0.4], axis=0
        )
        tracking_config["tracking_type"] = "iou"
        tracker = Node(tracking_config)
        prev_tags = []
        for i, inputs in enumerate(two_people_seq):
            # Set the track which doesn't have a detection to be "lost"
            # by setting `lost > max_lost`
            if i == seq_idx + 1:
                tracker.tracker.tracker.tracks[2].lost = (
                    tracker.tracker.tracker.max_lost + 1
                )
            outputs = tracker.run(inputs)
            assert len(outputs["obj_tags"]) == len(inputs["bboxes"])
            # This happens to be true for the test case, not a guaranteed
            # behaviour during normal operation.
            assert len(tracker.tracker.tracker.tracks) == len(inputs["bboxes"])
            # Special handling of comparing tag during and right after
            # seq_idx since a detection got added and removed
            if i == seq_idx:
                assert outputs["obj_tags"] == prev_tags + ["2"]
            elif i == seq_idx + 1:
                assert outputs["obj_tags"] == prev_tags[:-1]
            elif i > 0:
                assert outputs["obj_tags"] == prev_tags
            prev_tags = outputs["obj_tags"]
