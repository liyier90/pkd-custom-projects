from pathlib import Path
from unittest import TestCase, mock

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import torch
import yaml
from peekingduck.weights_utils.finder import PEEKINGDUCK_WEIGHTS_SUBDIR

from custom_nodes.model.jde import Node
from custom_nodes.model.jdev1.jde_files.matching import fuse_motion, iou_distance

# Frame index for manual manipulation of detections to trigger some
# branches
SEQ_IDX = 6


@pytest.fixture(name="jde_config")
def fixture_jde_config():
    """Yields config while forcing the model to run on CPU."""
    file_path = Path(__file__).resolve().parent / "test_jde.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    node_config["weights_parent_dir"] = "~/code"

    with mock.patch("torch.cuda.is_available", return_value=False):
        yield node_config


@pytest.fixture(name="jde_config_gpu")
def fixture_jde_config_gpu():
    """Yields config which allows the model to run on GPU on CUDA-enabled
    devices.
    """
    file_path = Path(__file__).resolve().parent / "test_jde.yml"
    with open(file_path) as file:
        node_config = yaml.safe_load(file)
    node_config["root"] = Path.cwd()
    node_config["weights_parent_dir"] = "~/code"

    yield node_config


@pytest.fixture(
    name="jde_bad_config_value",
    params=[
        {"key": "iou_threshold", "value": -0.5},
        {"key": "iou_threshold", "value": 1.5},
        {"key": "nms_threshold", "value": -0.5},
        {"key": "nms_threshold", "value": 1.5},
        {"key": "score_threshold", "value": -0.5},
        {"key": "score_threshold", "value": 1.5},
    ],
)
def fixture_jde_bad_config_value(request, jde_config):
    """Various invalid config values."""
    jde_config[request.param["key"]] = request.param["value"]
    return jde_config


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


class TestJDE:
    def test_should_give_empty_output_for_no_human_images(
        self, no_human_images, jde_config
    ):
        """Input images either contain nothing or non-humans."""
        blank_image = cv2.imread(no_human_images)
        jde = Node(jde_config)
        output = jde.run({"img": blank_image})
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
        self, two_people_seq, jde_config
    ):
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            output = jde.run(inputs)
            if i > 1:
                for track_id, track in enumerate(jde.model.tracker.tracked_stracks):
                    assert repr(track) == f"OT_{track_id + 1}_(1-{i + 1})"
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
    def test_tracking_ids_should_be_consistent_across_frames_gpu(
        self, two_people_seq, jde_config_gpu
    ):
        jde = Node(jde_config_gpu)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            output = jde.run(inputs)
            if i > 1:
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    def test_reactivate_tracks(self, two_people_seq, jde_config):
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            if i == SEQ_IDX:
                # These STrack should get re-activated
                for track in jde.model.tracker.tracked_stracks:
                    track.mark_lost()
            output = jde.run(inputs)
            if i > 1:
                assert output["obj_tags"] == prev_tags
            prev_tags = output["obj_tags"]

    def test_associate_with_iou(self, two_people_seq, jde_config):
        jde = Node(jde_config)
        prev_tags = []
        with mock.patch(
            "custom_nodes.model.jdev1.jde_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ):
            for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
                output = jde.run(inputs)
                if i > 1:
                    assert output["obj_tags"] == prev_tags
                prev_tags = output["obj_tags"]

    def test_mark_unconfirmed_tracks_for_removal(self, two_people_seq, jde_config):
        """Manipulate both embedding and iou distance to be above the cost
        limit so nothing gets associated and all gets marked for removal. As a
        result, the Tracker should no produce any track IDs.
        """
        jde = Node(jde_config)
        with mock.patch(
            "custom_nodes.model.jdev1.jde_files.matching.fuse_motion",
            wraps=replace_fuse_motion,
        ), mock.patch(
            "custom_nodes.model.jdev1.jde_files.matching.iou_distance",
            wraps=replace_iou_distance,
        ):
            for inputs in ({"img": x["img"]} for x in two_people_seq):
                output = jde.run(inputs)
                assert not output["obj_tags"]

    def test_remove_lost_tracks(self, two_people_seq, jde_config):
        # Set buffer and as a result `max_time_lost` to extremely short so
        # lost tracks will get removed
        jde_config["track_buffer"] = 1
        jde = Node(jde_config)
        prev_tags = []
        for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
            if i >= SEQ_IDX:
                inputs["img"] = np.zeros_like(inputs["img"])
            output = jde.run(inputs)
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
    def test_new_video_frame_rate(self, two_people_seq, jde_config, mot_metadata):
        jde = Node(config=jde_config)
        prev_tags = []
        with TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.jde_mot.jde_model.logger"
        ) as captured:
            for i, inputs in enumerate({"img": x["img"]} for x in two_people_seq):
                # Insert mot_metadata in input to signal a new model should be
                # created
                if i == 0:
                    inputs["mot_metadata"] = mot_metadata
                output = jde.run(inputs)
                if i == 0:
                    assert captured.records[0].getMessage() == (
                        "Creating new model with frame rate: "
                        f"{mot_metadata['frame_rate']:.2f}..."
                    )
                if i > 1:
                    assert output["obj_tags"] == prev_tags
                assert jde._frame_rate == pytest.approx(mot_metadata["frame_rate"])
                prev_tags = output["obj_tags"]

    def test_invalid_config_value(self, jde_bad_config_value):
        with pytest.raises(ValueError) as excinfo:
            _ = Node(config=jde_bad_config_value)
        assert "_threshold must be in [0, 1]" in str(excinfo.value)

    def test_invalid_config_model_files(self, jde_config):
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=True
        ), pytest.raises(ValueError) as excinfo:
            jde_config["weights"]["model_file"]["864x480"] = "some/invalid/path"
            _ = Node(config=jde_config)
        assert "Model file does not exist. Please check that" in str(excinfo.value)

    def test_invalid_image(self, no_human_images, jde_config):
        blank_image = cv2.imread(no_human_images)
        # Potentially passing in a file path or a tuple from image reader
        # output
        jde = Node(jde_config)
        with pytest.raises(TypeError) as excinfo:
            _ = jde.run({"img": Path.cwd()})
        assert str(excinfo.value) == "image must be a np.ndarray"
        with pytest.raises(TypeError) as excinfo:
            _ = jde.run({"img": ("image name", blank_image)})
        assert str(excinfo.value) == "image must be a np.ndarray"

    def test_no_weights(self, jde_config):
        # weights_dir = jde_config["root"].parent / PEEKINGDUCK_WEIGHTS_SUBDIR
        weights_dir = (
            Path(jde_config["weights_parent_dir"]).expanduser()
            / PEEKINGDUCK_WEIGHTS_SUBDIR
        )
        with mock.patch(
            "peekingduck.weights_utils.checker.has_weights", return_value=False
        ), mock.patch(
            "peekingduck.weights_utils.downloader.download_weights",
            wraps=replace_download_weights,
        ), TestCase.assertLogs(
            "peekingduck.pipeline.nodes.model.jde_mot.jde_model.logger"
        ) as captured:
            jde = Node(config=jde_config)
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
            assert jde is not None
