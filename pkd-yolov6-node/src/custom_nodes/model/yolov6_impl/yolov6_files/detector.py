"""YOLOv6 detector implementation."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from peekingduck.pipeline.utils.bbox.transforms import xyxy2xyxyn

from .data.data_augment import letterbox
from .layers.common import RepVGGBlock
from .models.yolo import YOLOv6
from .utils.nms import non_max_suppression
from .utils.torch_utils import fuse_model


class Detector:  # pylint: disable=too-many-instance-attributes
    """Object detection class using YOLOv6 to predict object bboxes."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_dir: Path,
        class_names: List[str],
        detect_ids: List[int],
        model_type: str,
        num_classes: int,
        model_file: Dict[str, str],
        max_detections: int,
        agnostic_nms: bool,
        fuse: bool,
        half: bool,
        multi_label: bool,
        input_size: int,
        iou_threshold: float,
        score_threshold: float,
    ):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = class_names
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_config_path = model_dir / f"{self.model_type}.yml"
        self.model_path = model_dir / model_file[self.model_type]
        self.max_detections = max_detections
        self.agnostic_nms = agnostic_nms
        self.fuse = fuse
        self.half = half and self.device.type == "cuda"
        self.multi_label = multi_label
        self.input_size = (input_size, input_size)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.update_detect_ids(detect_ids)

        self.yolov6 = self._create_yolov6_model()
        self.stride = int(self.yolov6.stride.max())

    @torch.no_grad()
    def predict_object_bbox_from_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detects bounding boxes of selected object categories from an image.

        The input image is first scaled according to the `input_size`
        configuration option. Detection results will be filtered according to
        `iou_threshold`, `score_threshold`, and `detect_ids` configuration
        options. Bounding boxes coordinates are then normalized w.r.t. the
        input `image` size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        processed_image = self._preprocess(image)
        processed_image = processed_image.to(self.device)
        if len(processed_image.shape) == 3:
            # expand for batch dim
            processed_image = processed_image[None]
        prediction = self.yolov6(processed_image)
        bboxes, classes, scores = self._postprocess(prediction, image, processed_image)
        return bboxes, classes, scores

    def update_detect_ids(self, ids: List[int]) -> None:
        """Updates list of selected object category IDs. When the list is
        empty, all available object category IDs are detected.
        Args:
            ids: List of selected object category IDs
        """
        self.detect_ids = torch.Tensor(ids).to(self.device)  # type: ignore
        if self.half:
            self.detect_ids = self.detect_ids.half()

    def _create_yolov6_model(self) -> YOLOv6:
        """Creates a YOLOv6 model and loads its weights.

        Logs model configurations.

        Returns:
            (YOLOv6Model): YOLOv6 model.
        """
        self.logger.info(
            "YOLOv6 model loaded with the following configs:\n\t"
            f"Model type: {self.model_type}\n\t"
            f"Input resolution: {self.input_size}\n\t"
            f"IDs being detected: {self.detect_ids.int().tolist()}\n\t"
            f"IOU threshold: {self.iou_threshold}\n\t"
            f"Score threshold: {self.score_threshold}\n\t"
            f"Half-precision floating-point: {self.half}\n\t"
        )
        return self._load_yolov6_weights()

    def _get_model(self) -> YOLOv6:
        """Constructs YOLOv6 model based on parsed configuration.

        Args:
            model_size (Dict[str, float]): Depth and width of the model.

        Returns:
            (YOLOX): YOLOX model.
        """
        with open(self.model_config_path) as infile:
            config = yaml.safe_load(infile.read())
        model = YOLOv6(
            config,
            channels=3,
            num_classes=self.num_classes,
            anchors=config["model"]["head"]["anchors"],
        )
        return model

    def _load_yolov6_weights(self) -> YOLOv6:
        """Loads YOLOv6 model weights.

        Returns:
            (YOLOv6): YOLOv6 model.

        Raises:
            ValueError: `model_path` does not exist.
        """
        if self.model_path.is_file():
            ckpt = torch.load(str(self.model_path), map_location=self.device)
            model = self._get_model().to(self.device).float()
            model.load_state_dict(ckpt)
            if self.fuse:
                fuse_model(model)
            model.eval()
            if self.device.type != "cpu":
                # warmup
                model(
                    torch.zeros(1, 3, *self.input_size)
                    .to(self.device)
                    .type_as(next(model.parameters()))
                )
            # switch to deploy
            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()

            if self.half:
                model.half()
            model.eval()

            return model

        raise ValueError(
            f"Model file does not exist. Please check that {self.model_path} exists."
        )

    def _postprocess(
        self,
        prediction: torch.Tensor,
        orig_image: np.ndarray,
        processed_image: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocesses the model detections. Performs non maximum suppression
        and filters by score threshold and detect IDs.

        Args:
            prediction (torch.Tensor): Model predictions.
            orig_image (np.ndarray): Original input image.
            processed_image (np.ndarray): Preprocessed input image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores
        """
        det = non_max_suppression(
            prediction,
            self.score_threshold,
            self.iou_threshold,
            self.detect_ids,
            self.agnostic_nms,
            self.multi_label,
            self.max_detections,
        )[0]
        if not det.size(0):
            return np.empty((0, 4)), np.empty(0), np.empty(0)
        det[:, :4] = self.rescale(
            det[:, :4], processed_image.shape[2:], orig_image.shape
        ).round()
        # print(det[:, :4], det[:, 4])

        output_np = det.cpu().detach().numpy()
        bboxes = xyxy2xyxyn(output_np[:, :4], *orig_image.shape[:2])
        scores = output_np[:, 4]
        classes = np.array([self.class_names[int(i)] for i in output_np[:, 5]])
        return bboxes, classes, scores

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image before feeding it to the model. Pads image to the
        required shape and flips its dimensions and channels.

        Args:
            image (np.ndarray): The input image;

        Returns:
            (torch.Tensor): The preprocessed image tensor.
        """
        padded_img = letterbox(image, self.input_size, stride=self.stride)

        # Convert
        padded_img = padded_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_tensor = torch.from_numpy(np.ascontiguousarray(padded_img))
        # uint8 to fp16/32
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255  # 0 - 255 to 0.0 - 1.0

        return img_tensor

    @staticmethod
    def rescale(
        boxes: torch.Tensor,
        processed_shape: Tuple[int, ...],
        orig_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Rescales the bounding box coordinates to the original image shape.

        Args:
            boxes (torch.Tensor): Bounding boxes of detected objects.
            processed_shape (Tuple[int, ...]): Shape of the preprocessed image.
            orig_shape (Tuple[int, ...]): Shape of the original image.
        """
        ratio = min(
            processed_shape[0] / orig_shape[0], processed_shape[1] / orig_shape[1]
        )
        padding = (
            (processed_shape[1] - orig_shape[1] * ratio) / 2,
            (processed_shape[0] - orig_shape[0] * ratio) / 2,
        )

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, orig_shape[1])  # x1
        boxes[:, 1].clamp_(0, orig_shape[0])  # y1
        boxes[:, 2].clamp_(0, orig_shape[1])  # x2
        boxes[:, 3].clamp_(0, orig_shape[0])  # y2

        return boxes
