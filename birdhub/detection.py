"""Classes for motion detection"""
import cv2
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import numpy as np
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from birdhub.orchestration import Mediator

class Detection:
    """Class to represent a detection"""

    def __init__(self, 
                       source_image:np.ndarray,
                       labels:Optional[List[str]]=None,
                       confidences:Optional[List[float]]=None,
                       bboxes:Optional[List[np.ndarray]]=None,
                       meta_information:Dict[str, str]=None):
        self.source_image = source_image
        self.labels = labels
        self.confidences = confidences
        self.bboxes = bboxes
        self.meta_information = meta_information

    def __str__(self):
        return f"Detection(label={self.labels}, confidence={self.confidences}, bbox={self.bboxes}, meta_information={self.meta_information})"

    def set(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        if key not in self.__dict__:
            return default
        return getattr(self, key)

    def __repr__(self):
        return self.__str__()


class Detector(ABC):
    """Base class for detectors"""

    def __init__(self) -> None:
        self._event_manager = None

    def add_event_manager(self, event_manager: Mediator):
        self._event_manager = event_manager

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Detection:
        raise NotImplementedError


class SimpleMotionDetector(Detector):
    """Simple motion detector that compares the current frame with the previous frame"""

    def __init__(self, threshold=20, blur=21, dilation_kernel=np.ones((5,5)), threshold_area=50, activation_frames:int=5):
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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a blur to the image
        blur = cv2.GaussianBlur(gray_image, (self._blur, self._blur), 0)

        return blur

    def _update_detections(self, detection: Detection, rects: List[np.ndarray]):
        """Update the detection with the given rects if they are not empty"""
        if len(rects) > 0:
            detection.set("labels", ["motion"]*len(rects))
            detection.set("bboxes", rects)
        self._detections.append(detection)

    def detect(self, frame: np.ndarray) -> Detection:
        """Detect motion between the current frame and the previous frame"""
        detection = Detection(source_image=frame)
        # add first frame
        if self._previous_frame is None:
            self._previous_frame = frame
            return detection
        # Convert the frames to grayscale
        prep_frame, prep_previous = self._preprocess_image(frame), self._preprocess_image(self._previous_frame)
        # Calculate the absolute difference between the current frame and the previous frame
        frame_delta = cv2.absdiff(prep_previous, prep_frame)
        # Apply a threshold to the frame delta
        _, threshold = cv2.threshold(frame_delta, self._threshold, 255, cv2.THRESH_BINARY)
        # Dilate the thresholded image to fill in holes
        dilated = cv2.dilate(threshold, self._dilation_kernel, iterations=2)
        # Find contours on the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < self._threshold_area:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append([x, y, x + w, y + h])
        # update detections
        self._update_detections(detection, rects)
        # check whether motion is detected
        if len(rects) > 0 and self._motion_frames < self._activation_frames:
            self._motion_frames += 1
        if len(rects) > 0 and self._motion_frames >= self._activation_frames:
            if self._event_manager is not None:
                self._event_manager.notify("detection", self._detections)
            self._motion_frames = 0
            self._detections = []
        if len(rects) == 0 and self._motion_frames > 0:
            self._motion_frames = 0
            self._detections = []
        # Update the previous frame
        self._previous_frame = frame
        return detection

# Adjust this to detector interface
class BirdDetectorYolov5(Detector):

    def __init__(self,
                 model_path,
                 image_size=(640,640),
                 confidence_threshold=0.25,
                 iou_threhsold=0.45) -> None:
        # Load model
        self._device = select_device('')
        self._model = self._load_model(model_path)
        self._classes = self._model.names
        self._stride = self._model.stride
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold
        self._iou_threhsold = iou_threhsold
        # define data loader

    def _load_model(self, model_path):
        model = DetectMultiBackend(model_path, device=self._device)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.to(self._device)
        return model
    
    def _extract_birds_from_prediction(self, prediction):
        return [self._classes[int(i)] for i in prediction.numpy()[:, -1]]

    def _get_confidences(self, prediction):
        return prediction[:, -2].numpy().tolist()
    
    def _get_boxes(self, prediction):
        return prediction[:, :4].numpy().tolist()

    def detect(self, frame):
        # TODO: resize if needed
        # assumes im is in opencv BGR format
        im = frame.transpose(2,0,1)[::-1] # BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self._model.device)
        im = im.half() if self._model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        results = self._model(im, augment=False, visualize=False)
        results = non_max_suppression(results,
                                      self._confidence_threshold,
                                      self._iou_threhsold,
                                      max_det=1000)
        stacked = torch.cat(results, 0).cpu()
        birds = self._extract_birds_from_prediction(stacked)
        confidences = self._get_confidences(stacked)
        boxes = self._get_boxes(stacked)
        return birds, confidences, boxes, stacked

    @staticmethod
    def show_bbox(image, boxes, labels):
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            # Draw the bounding box with red lines
            draw.rectangle((x1, y1, x2, y2),
                            outline=(255, 0, 0), # Red in RGB
                            width=5)             # Line width
            font = ImageFont.truetype("arial.ttf", 30)
            draw.text((x1, y1 - 30), label, fill=(255, 0, 0), font=font) # Black in RGB
        image.show()



# TODO: adjust single this to conform to the detector interface
class SingleClassImageSequence(Detector):
    """Accumulates object predictions and implements functionality to
    determine to most likely class of the object in the sequence.
    When there are multiple objects in the seuqence, the class with the
    highest cumulative confidence is returned.
    For example, if there are 2 predictions for pigeon with confidence 0.2 each
    and one prediction for crow, then crow will be returned.
    If the pigeions in the example above had confidence 0.5 each, then pigeon would
    be returned. 
    If there are multiple objects within the same frame, they areaccumulated together.
    """


    def __init__(self, detector: Detector, minimum_number_detections:int=5) -> None:
        self._detections = {}
        self._number_detections = 0
        self._minimum_number_detections = minimum_number_detections
    

    def detect(self, frame: np.ndarray) -> Detection:
        detection = self._detector.detect(frame)
        objects, confidences = detection.get("labels", default=[]), detection.get("confidences", default=[])
        for obj, conf in zip(objects, confidences):
            self._number_detections += 1
            if obj not in self._detections:
                self._detections[obj] = conf
            self._detections[obj] = self._detections[obj] + conf


    def add_detections(self, objects, confidences):
        for obj, conf in zip(objects, confidences):
            self._number_detections += 1
            if obj not in self._detections:
                self._detections[obj] = conf
            self._detections[obj] = self._detections[obj] + conf
    
    def _has_reached_consensus(self):
        return self._number_detections >= self._minimum_number_detections
    
    def _get_most_likely_object(self):
        if self._number_detections < self._minimum_number_detections:
            return None
        return max(self._detections, key=self._detections.get)
        