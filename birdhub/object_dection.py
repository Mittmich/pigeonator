""""""

import torch
from yolov5.models.common import DetectMultiBackend, Detections
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


class BirdDetectorYolov5:

    def __init__(self,
                 model_path,
                 image_size=(640,640),
                 confidence_trheshold=0.25,
                 iou_threhsold=0.45) -> None:
        # Load model
        self._device = select_device('')
        self._model = self._load_model(model_path)
        self._classes = self._model.names
        self._stride = self._model.stride
        self._image_size = image_size
        self._confidence_trheshold = confidence_trheshold
        self._iou_threhsold = iou_threhsold
        # define data loader

    def _load_model(self, model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.to(self._device)
        return model
    
    def detect_birds(self, im):
        results = self._model(im, augment=False)
        # TODO; extract only birds
        return results


