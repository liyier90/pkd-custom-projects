from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR

from custom_nodes.model.fairmot import Node
from custom_nodes.model.fairmotv1.fairmot_files.matching import (
    fuse_motion,
    iou_distance,
)

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6


@pytest.fixture(name="fairmot_config")
def fixture_fairmot_config():
    """Yields config while forcing the model to run on CPU."""
    file_path = Path(__file__).resolve().parent / "test_fairmot.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    node_config["weights_parent_dir"] = "~/code"

    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture(name="fairmot_config_gpu")
def fixture_fairmot_config_gpu():
    """Yields config which allows the model to run on GPU on CUDA-enabled
    devices.
    """
    file_path = Path(__file__).resolve().parent / "test_fairmot.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    node_config["weights_parent_dir"] = "~/code"

    yield node_config


@pytest.fixture(
    name="fairmot_bad_config_value",
    params=[
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def fixture_fairmot_bad_config_value(request, fairmot_config):
    """Various invalid config values."""
    fairmot_config[request.param["key"]] = request.param["value"]
    return fairmot_config


@pytest.fixture(
    name="fairmot_negative_config_value",
    params=[
        {"key": "K", "value": -0.5},
        {"key": "min_box_area", "value": -0.5},
        {"key": "track_buffer", "value": -0.5},
    ],
)
def fixture_fairmot_negative_config_value(request, fairmot_config):
    """Various invalid config values."""
    fairmot_config[request.param["key"]] = request.param["value"]
    return fairmot_config


def replace_download_weights(*_):
    pass


def replace_fuse_motion(*args):
    """Manipulate the computed embedding distance so they are too large and
    cause none of the detections to be associated. This forces the Tracker to
    associate with IoU costs.
    """
    return np.ones_like(fuse_motion(*args))


def replace_iou_distance(*args):
    """Manipulate the computed IoU-based costs so they are too large and
    cause none of the detections to be associated. This forces the Tracker to
    mark tracks for removal.
    """
    return np.ones_like(iou_distance(*args))


class TestFairMOT:
    def test_should_give_empty_output_for_no_human_images(
        self, no_human_images, fairmot_config
    ):
        """Input images either contain nothing or non-humans."""
        blank_image = cv2.imread(no_human_images)
        fairmot = Node(fairmot_config)
        output = fairmot.run({"img": blank_image})
        expected_output = {
            "bboxes": [],
            "bbox_labels": [],
            "bbox_scores": [],
            "obj_tags": [],
        }
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output["bboxes"], expected_output["bboxes"])
        npt.assert_equal(output["bbox_labels"], expected_output["bbox_labels"])
        npt.assert_equal(output["bbox_scores"], expected_output["bbox_scores"])
        npt.assert_equal(output["obj_tags"], expected_output["obj_tags"])

    def test_tracking_ids_should_be_consistent_across_frames(
        self, two_people_seq, fairmot_config
    ):
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            output = fairmot.run(inputs)
            if i > 1:
                for track_id, track in enumerate(fairmot.model.tracker.tracked_stracks):
                    assert repr(track) == f"OT_{track_id + 1}_(1-{i + 1})"
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_tracking_ids_should_be_consistent_across_frames_gpu(
        self, two_people_seq, fairmot_config_gpu
    ):
        fairmot = Node(fairmot_config_gpu)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    def test_reactivate_tracks(self, two_people_seq, fairmot_config):
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            if i == SEQ_IDX:
                # These STrack should get re-activated
                for track in fairmot.model.tracker.tracked_stracks:
                    track.mark_lost()
            output = fairmot.run(inputs)
            if i > 1:
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    def test_associate_with_iou(self, two_people_seq, fairmot_config):
        fairmot = Node(fairmot_config)
        prev_tags = []
        with mock.patch(
            "custom_nodes.model.fairmotv1.fairmot_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
                output = fairmot.run(inputs)
                if i > 1:
                    assert output["obj_tags"] == prev_tags
                prev_tags = output["obj_tags"]

    def test_mark_unconfirmed_tracks_for_removal(self, two_people_seq, fairmot_config):
        """Manipulate both embedding and iou distance to be above the cost
        limit so nothing gets associated and all gets marked for removal. As a
        result, the Tracker should no produce any track IDs.
        """
        fairmot = Node(fairmot_config)
        with mock.patch(
            "custom_nodes.model.fairmotv1.fairmot_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ), mock.patch(
            "custom_nodes.model.fairmotv1.fairmot_files.matching.iou_distance",
            wraps=replace_iou_distance,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
                output = fairmot.run(inputs)
                if i == 0:
                    # Skipping the assert on the first frame FairMOT sets
                    # STrack to is_activated=True on when frame_id=1 but JDE
                    # doesn't
                    continue
                assert not output["obj_tags"]

    def test_remove_lost_tracks(self, two_people_seq, fairmot_config):
        # Set buffer and as a result `max_time_lost` to extremely short so
        # lost tracks will get removed
        fairmot_config["track_buffer"] = 1
        fairmot = Node(fairmot_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            if i >= SEQ_IDX:
                inputs["img"] = np.zeros_like(inputs["img"])
            output = fairmot.run(inputs)
            # switched to black image from SEQ_IDX onwards, nothing should be
            # detected on this frame ID
            if i == SEQ_IDX:
                assert not output["obj_tags"]
            elif i > 1:
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    @pytest.mark.parametrize(
        "mot_metadata",
        [
            {"frame_rate": 30.0, "reset_model": True},
            {"frame_rate": 10.0, "reset_model": False},
        ],
    )
    def test_new_video_frame_rate(self, two_people_seq, fairmot_config, mot_metadata):
        fairmot = Node(config=fairmot_config)
        prev_tags = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.fairmot_mot.fairmot_model.logger"
        ) as captured:
            for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
                # Insert mot_metadata in input to signal a new model should be
                # created
                if i == 0:
                    inputs["mot_metadata"] = mot_metadata
                output = fairmot.run(inputs)
                if i == 0:
                    assert captured.records[0].getMessage() == (
                        "Creating new model with frame rate: "
                        f"{mot_metadata['frame_rate']:.2f}..."
                    )
                if i > 1:
                    assert output["obj_tags"] == prev_tags
                assert fairmot._frame_rate == pytest.approx(mot_metadata["frame_rate"])
                prev_tags = output["obj_tags"]

    def test_invalid_config_value(self, fairmot_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=fairmot_bad_config_value)
        assert "_threshold must be in [0, 1]" in str(excinfo.value)

    def test_negative_config_value(self, fairmot_negative_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=fairmot_negative_config_value)
        assert "must be more than 0" in str(excinfo.value)

    def test_invalid_config_model_files(self, fairmot_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError) as excinfo:
            fairmot_config["weights"]["model_file"]["dla_34"] = "some/invalid/path"
            _ = Node(config=fairmot_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_images, fairmot_config):
        blank_image = cv2.imread(no_human_images)
        # Potentially passing in a file path or a tuple from image reader
        # output
        fairmot = Node(fairmot_config)
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": Path.cwd()})
        assert str(excinfo.value) == "image must be a np.ndarray"
        with pytest.raises(TypeError) as excinfo:
            _ = fairmot.run({"img": ("image name", blank_image)})
        assert str(excinfo.value) == "image must be a np.ndarray"

    def test_no_weights(self, fairmot_config):
        # weights_dir = fairmot_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        weights_dir = (
            Path(fairmot_config["weights_parent_dir"]).expanduser()
            / PEEKINGDUCK_WEIGHTS_SUBDIR
        )
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.fairmot_mot.fairmot_model.logger"
        ) as captured:
            fairmot = Node(config=fairmot_config)
            print(captured)
            # records 0 - 20 records are updates to configs
            assert (
                captured.records[0].getMessage()
                == "No weights detected. Proceeding to download..."
            )
            assert (
                captured.records[1].getMessage()
                == f"Weights downloaded to {weights_dir}."
            )
            assert fairmot is not None
