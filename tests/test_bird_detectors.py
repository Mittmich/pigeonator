import pytest
import torch
import datetime
import numpy as np
import cv2
from birdhub.detection import BirdDetectorYolov5, Frame

class TestBirdDetectorYolov5:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.bird_detector = BirdDetectorYolov5("weights/bh_v1.onnx")

    def test_extract_birds_from_prediction(self):
        prediction = torch.tensor([[0, 0, 0, 0, 0.5, 1]])

        birds = self.bird_detector._extract_birds_from_prediction(prediction)
        assert birds == ['Crow']

    def test_get_confidences(self):
        prediction = torch.tensor([[0, 0, 0, 0, 0.5, 1]])

        confidences = self.bird_detector._get_confidences(prediction)
        assert confidences == [0.5]

    def test_get_boxes(self):
        prediction = torch.tensor([[1, 2, 3, 4, 0.5 , 1]])

        boxes = self.bird_detector._get_boxes(prediction, (640, 640))
        assert boxes == [[1, 2, 3, 4]]

    def test_detect_bird(self):
        # load test jpg image
        image = cv2.resize(cv2.imread("tests/test_data/pigeon.jpg"), (640, 640))
        frame = Frame(image=image, timestamp=datetime.datetime.now())
        detections = self.bird_detector.detect(frame)
        assert detections[0].labels == ['Pigeon']
        assert len(detections) == 1

