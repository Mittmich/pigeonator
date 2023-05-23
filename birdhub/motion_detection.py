"""Classes for motion detection"""
import cv2
from abc import ABC, abstractmethod
import numpy as np

class MotionDetector(ABC):
    """Base class for motion detectors"""

    @abstractmethod
    def detect(self, frame, previous_frame):
        """Detect motion between the current frame and the previous frame"""
        raise NotImplementedError


class SimpleMotionDetector(MotionDetector):
    """Simple motion detector that compares the current frame with the previous frame"""

    def __init__(self, threshold=20, blur=21, dilation_kernel=np.ones((5,5)), threshold_area=50):
        self._threshold = threshold
        self._blur = blur
        self._dilation_kernel = dilation_kernel
        self._threshold_area = threshold_area

    def _preprocess_image(self, image):
        """Preprocess the image"""
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a blur to the image
        blur = cv2.GaussianBlur(gray_image, (self._blur, self._blur), 0)

        return blur

    def detect(self, frame, previous_frame):
        """Detect motion between the current frame and the previous frame"""
        # guard against None
        if frame is None or previous_frame is None:
            return []
        # Convert the frames to grayscale
        prep_frame, prep_previous = self._preprocess_image(frame), self._preprocess_image(previous_frame)

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

        # Return the rectangles
        return rects