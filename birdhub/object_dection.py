""""""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


class BirdDetectorYolov5:

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

    def detect_birds(self, im):
        # TODO: resize if needed
        # assumes im is in opencv BGR format
        im = im.transpose(2,0,1)[::-1] # BGR to RGB
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


class SingleClassImageSequence():
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


    def __init__(self, minimum_number_detections:int=5, ) -> None:
        self._detections = {}
        self._number_detections = 0
        self._minimum_number_detections = minimum_number_detections
    
    def add_detections(self, objects, confidences):
        for obj, conf in zip(objects, confidences):
            self._number_detections += 1
            if obj not in self._detections:
                self._detections[obj] = conf
            self._detections[obj] = self._detections[obj] + conf
    
    def has_reached_consensus(self):
        return self._number_detections >= self._minimum_number_detections
    
    def get_most_likely_object(self):
        if self._number_detections < self._minimum_number_detections:
            return None
        return max(self._detections, key=self._detections.get)
        