"""Classes for motion detection"""
import cv2
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import numpy as np
import torch
import logging
from multiprocessing import Pipe, Process
from threading import Thread
from PIL import Image, ImageDraw, ImageFont
from birdhub.yolo_utils import DetectMultiBackend
from birdhub.yolo_utils import non_max_suppression
from birdhub.yolo_utils import select_device
from birdhub.orchestration import Mediator
from birdhub.video import Frame


class Detection:
    """Class to represent a detection"""

    def __init__(
        self,
        frame_timestamp: datetime,
        labels: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None,
        bboxes: Optional[List[np.ndarray]] = None,
        meta_information: Dict[str, str] = None,
    ):
        self.frame_timestamp = frame_timestamp
        self.labels = labels
        self.confidences = confidences
        self.bboxes = bboxes
        self.meta_information = meta_information

    def __str__(self):
        return f"Detection(timestamp={self.frame_timestamp}, labels={self.labels}, confidences={self.confidences}, meta_information={self.meta_information})"

    def set(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        if key not in self.__dict__ or getattr(self, key) is None:
            return default
        return getattr(self, key)

    def __repr__(self):
        return self.__str__()


# TODO: create detection sequence that holds multiple detections and associated meta information


class Detector(ABC):
    """Base class for detectors"""

    def __init__(self) -> None:
        self._event_manager_connection = None

    def add_event_manager(self, event_manager: Mediator):
        # create commuinication pipe
        self._event_manager_connection, child_connection = Pipe()
        # register pipe with event manager
        event_manager.register_pipe("detector", child_connection)

    @abstractmethod
    def detect(self, frame: Frame) -> Optional[List[Detection]]:
        raise NotImplementedError

    def run(self):
        """Start the detector process"""
        self._process = Thread(target=self._run)
        self._process.start()

    def instantiate_model(self):
        """Instantiate the model"""
        pass

    def _run(self):
        # instantiate detection model
        self.instantiate_model()
        while True:
            if self._event_manager_connection.poll(1):
                frame = self._event_manager_connection.recv()
                self.detect(frame)


class SimpleMotionDetector(Detector):
    """Simple motion detector that compares the current frame with the previous frame"""

    def __init__(
        self,
        threshold=20,
        blur=21,
        dilation_kernel=np.ones((5, 5)),
        threshold_area=50,
        activation_frames: int = 5,
    ):
        super().__init__()
        self._threshold = threshold
        self._blur = blur
        self._dilation_kernel = dilation_kernel
        self._threshold_area = threshold_area
        self._previous_frame = None
        self._activation_frames = activation_frames
        self._motion_frames = 0
        self._detections = []

    def _preprocess_image(self, image):
        """Preprocess the image"""
        # Convert the image to grayscale
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # TODO: send event that reinitialization of stream is needed
            return None
        # Apply a blur to the image
        blur = cv2.GaussianBlur(gray_image, (self._blur, self._blur), 0)

        return blur

    def _update_detections(self, frame: Frame, rects: List[np.ndarray]):
        """Update the detection with the given rects if they are not empty"""
        if len(rects) > 0:
            detection = Detection(
                frame.timestamp,
                labels=["motion"] * len(rects),
                confidences=[1.0] * len(rects),
                bboxes=rects,
                meta_information={
                    "type": "motion",
                    "frame_timestamp": frame.timestamp.isoformat(
                        sep=" ", timespec="milliseconds"
                    ),
                },
            )
            self._detections.append(detection)

    def detect(self, frame: Frame) -> Optional[List[Detection]]:
        """Detect motion between the current frame and the previous frame"""
        # add first frame
        if self._previous_frame is None:
            self._previous_frame = frame
            return None
        # Convert the frames to grayscale
        prep_frame, prep_previous = self._preprocess_image(
            frame.image
        ), self._preprocess_image(self._previous_frame.image)
        if prep_frame is None or prep_previous is None:
            return None
        # Calculate the absolute difference between the current frame and the previous frame
        frame_delta = cv2.absdiff(prep_previous, prep_frame)
        # Apply a threshold to the frame delta
        _, threshold = cv2.threshold(
            frame_delta, self._threshold, 255, cv2.THRESH_BINARY
        )
        # Dilate the thresholded image to fill in holes
        dilated = cv2.dilate(threshold, self._dilation_kernel, iterations=2)
        # Find contours on the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < self._threshold_area:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append([x, y, x + w, y + h])
        # update detections
        self._update_detections(frame, rects)
        # check whether motion is detected
        if len(rects) > 0 and self._motion_frames < self._activation_frames:
            self._motion_frames += 1
        if len(rects) > 0 and self._motion_frames >= self._activation_frames:
            if self._event_manager_connection is not None:
                self._event_manager_connection.send(("detection", self._detections))
            self._motion_frames = 0
            output = self._detections.copy()
            self._detections = []
            return output
        if len(rects) == 0 and self._motion_frames > 0:
            self._motion_frames = 0
            self._detections = []
        # Update the previous frame
        self._previous_frame = frame
        return None


class BirdDetectorYolov5(Detector):
    def __init__(
        self,
        model_path: str,
        image_size: Tuple[int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threhsold: float = 0.45,
        max_delay: int = 10 # in seconds
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._device = None
        self._model = None
        self._classes = None
        self._stride = None
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._iou_threhsold = iou_threhsold
        self._max_delay = max_delay

    def _load_model(self, model_path):
        model = DetectMultiBackend(model_path, device=self._device)
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.to(self._device)
        return model

    def instantiate_model(self):
        self._device = select_device("")
        self._model = self._load_model(self._model_path)
        self._stride = self._model.stride
        self._classes = self._model.names

    def _extract_birds_from_prediction(self, prediction):
        return [self._classes[int(i)] for i in prediction.numpy()[:, -1]]

    def _get_confidences(self, prediction):
        return prediction[:, -2].numpy().tolist()

    def _get_boxes(self, prediction, original_size: Tuple[int]):
        return [
            self._convert_bbox_to_original_size(i, original_size)
            for i in prediction[:, :4].numpy().tolist()
        ]

    def _convert_bbox_to_original_size(self, bbox, original_size):
        original_width, original_height = original_size
        resized_width, resized_height = self._image_size

        scale_width = original_width / resized_width
        scale_height = original_height / resized_height

        x1, y1, x2, y2 = bbox

        bbox_original = [
            x1 * scale_width,
            y1 * scale_height,
            x2 * scale_width,
            y2 * scale_height,
        ]

        return bbox_original

    def detect(self, frame: Frame) -> Optional[List[Detection]]:
        if self._model is None:
            raise ValueError("Model not instantiated")
        if frame.image is None:
            return None
        # check for delay and drop frame if delay is too high
        current_time = datetime.now()
        if (current_time - frame.timestamp).seconds > self._max_delay:
            return None
        original_size = frame.image.shape[1], frame.image.shape[0]
        resized = cv2.resize(frame.image, (self._image_size))
        # assumes im is in opencv BGR format
        im = resized.transpose(2, 0, 1)[::-1]  # BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self._model.device)
        im = im.half() if self._model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self._model(im, augment=False, visualize=False)
        results = non_max_suppression(
            results, self._confidence_threshold, self._iou_threhsold, max_det=1000
        )
        stacked = torch.cat(results, 0).cpu()
        birds = self._extract_birds_from_prediction(stacked)
        confidences = self._get_confidences(stacked)
        boxes = self._get_boxes(stacked, original_size)
        if len(birds) > 0:
            detection = [
                Detection(
                    frame_timestamp=frame.timestamp,
                    labels=birds,
                    confidences=confidences,
                    bboxes=boxes,
                    meta_information={
                        "type": "bird detected",
                        "timestamp": frame.timestamp.isoformat(
                            sep=" ", timespec="milliseconds"
                        ),
                    },
                )
            ]
            if self._event_manager_connection is not None:
                self._event_manager_connection.send(("detection", self._detections))
            return detection

    @staticmethod
    def show_bbox(image, boxes, labels):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            # Draw the bounding box with red lines
            draw.rectangle(
                (x1, y1, x2, y2), outline=(255, 0, 0), width=5  # Red in RGB
            )  # Line width
            font = ImageFont.truetype("arial.ttf", 30)
            draw.text((x1, y1 - 30), label, fill=(255, 0, 0), font=font)  # Black in RGB
        image.show()


# This is actually a composition of detectors -> make different base class?
class SingleClassSequenceDetector(Detector):
    """Accumulates object predictions and implements functionality to
    determine the most likely class of the object in the sequence.
    When there are multiple objects in the seuqence, the class with the
    highest cumulative confidence is returned.
    For example, if there are 2 predictions for pigeon with confidence 0.2 each
    and one prediction for crow, then crow will be returned.
    If the pigeions in the example above had confidence 0.5 each, then pigeon would
    be returned.
    If there are multiple objects within the same frame, they areaccumulated together.
    """

    def __init__(self, detector: Detector, minimum_number_detections: int = 5) -> None:
        super().__init__()
        self._detections = []
        self._object_detections = {}
        self._number_detections = 0
        self._minimum_number_detections = minimum_number_detections
        self._detector = detector

    def _accumulate_detections(self, detections: Optional[List[Detection]]):
        if detections is None:
            return
        # extend detections
        for detection in detections:
            objects, confidences = detection.get("labels", default=[]), detection.get(
                "confidences", default=[]
            )
            for obj, conf in zip(objects, confidences):
                self._number_detections += 1
                if obj not in self._object_detections:
                    self._object_detections[obj] = conf
                else:
                    self._object_detections[obj] = self._object_detections[obj] + conf
        self._detections.extend(detections)

    def _blank_detections(self):
        self._detections = []
        self._object_detections = {}
        self._number_detections = 0

    def _rewrite_to_consensus(self):
        # get most likely object
        most_likely_object = self._get_most_likely_object()
        for detection in self._detections:
            meta = detection.get("meta_information", {})
            meta["most_likely_object"] = most_likely_object
            meta["mean_confidence"] = (
                self._object_detections[most_likely_object] / self._number_detections
            )
            detection.set("meta_information", meta)

    def detect(self, frame: Frame) -> Optional[List[Detection]]:
        # accumulate detections
        self._accumulate_detections(self._detector.detect(frame))
        # determine if we have reached consensus
        if self._has_reached_consensus() and len(self._detections) > 0:
            self._rewrite_to_consensus()
            # copy output
            output = self._detections.copy()
            if self._event_manager_connection is not None:
                self._event_manager_connection.send(("detection", output))
            self._blank_detections()
            return output

    def _has_reached_consensus(self):
        return self._number_detections >= self._minimum_number_detections

    def _get_most_likely_object(self):
        if self._number_detections < self._minimum_number_detections:
            return None
        return max(self._object_detections, key=self._object_detections.get)
    
    def instantiate_model(self):
        """Delegate model instantiation to the detector."""
        self._detector.instantiate_model()


class MotionActivatedSingleClassDetector(SingleClassSequenceDetector):
    """Detector that is a composition of a motion detector and another detctor.
    The second detector only kicks in when motion is detected."""

    def __init__(
        self,
        detector: Detector,
        motion_detector: Detector,
        minimum_number_detections: int = 5,
        slack: int = 5,
    ) -> None:
        super().__init__(detector, minimum_number_detections)
        self._motion_detector = motion_detector
        self._slack = slack
        self._stop_detecting_in = 0

    def _set_slack(self, motion_detections: Optional[List[Detection]]):
        if motion_detections is not None:
            # log detection
            self._event_manager_connection.send(
                (
                    "log_request",
                    (
                        "detection",
                        {
                            "type": "motion",
                            "detail": "Class detector activated",
                            "timestamp": motion_detections[-1]
                            .get("frame_timestamp")
                            .isoformat(sep=" ", timespec="milliseconds"),
                        },
                        logging.DEBUG,
                    ),
                )
            )
            self._stop_detecting_in = self._slack
        elif self._stop_detecting_in > 0:
            self._stop_detecting_in -= 1
        else:
            self._blank_detections()

    def detect(self, frame: Frame) -> Optional[List[Detection]]:
        # pass image to motion detector
        motion_detections = self._motion_detector.detect(frame)
        self._set_slack(motion_detections)
        # if motion is detected, pass detection to other detector
        if motion_detections is not None or self._stop_detecting_in > 0:
            result = super().detect(frame)
            if result is not None:
                # Think about whether we want to reset the detector here -> this effectively means that we only detect one object per motion event
                self._blank_detections()
                return result
    
    def instantiate_model(self):
        """Delegate model instantiation to the two detector."""
        self._detector.instantiate_model()
        self._motion_detector.instantiate_model()
