"""Collection of recorder objects"""
from abc import ABC, abstractmethod
import os
from datetime import datetime
import cv2
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from birdhub.video import VideoWriter, Frame
from birdhub.orchestration import Mediator
from birdhub.detection import Detection
from birdhub.logging import logger


class Recorder(ABC):
    def __init__(
        self,
        outputDir: str,
        frame_size: Tuple[int],
        fps: int = 10,
        writer_factory: VideoWriter = VideoWriter,
    ) -> None:
        self._outputDir = outputDir
        self._frame_size = frame_size
        self._fps = fps
        self._event_manager = None
        self._writer_factory = writer_factory

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_recording_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}.avi")

    def add_event_manager(self, event_manager: Mediator):
        self._event_manager = event_manager

    @abstractmethod
    def register_frame(self, frame: Frame):
        pass

    def register_detection(self, detection: List[Detection]):
        pass

    def register_effect_activation(self, data: dict):
        pass


class ContinuousRecorder(Recorder):
    def __init__(
        self,
        outputDir: str,
        frame_size: Tuple[int],
        fps: int = 10,
        writer_factory: VideoWriter = VideoWriter,
    ) -> None:
        super().__init__(outputDir, frame_size, fps, writer_factory)
        logger.log_event("recording_started", "Continuous recording started")
        self._writer = self._writer_factory(
            self._get_recording_output_file(), self._fps, frame_size
        )

    def register_frame(self, frame:Frame):
        self._writer.write(frame.image)


class EventRecorder(Recorder):
    def __init__(
        self,
        outputDir: str,
        frame_size: Tuple[int],
        fps: int = 10,
        slack: int = 100,
        look_back_frames: int = 3,
        writer_factory: VideoWriter = VideoWriter,
        detection_writer_factory: VideoWriter = VideoWriter,
    ) -> None:
        super().__init__(outputDir, frame_size, fps, writer_factory)
        self._slack = slack
        self._look_back_frames = []
        self._look_back_frames_limit = look_back_frames
        self._writer = None
        self._detection_writer = None
        self._stop_recording_in = 0
        self._detection_writer_factory = detection_writer_factory
        self._detection_image_buffer = []
        self._recording = False

    def _get_detection_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}_detections.avi")

    def _create_detection_frame(self, detections: List[Detection], frame: Frame) -> np.ndarray:
        image = frame.image
        candidate_detection = [d for d in detections if d.frame_timestamp == frame.timestamp]
        if len(candidate_detection) == 0:
            return image
        detection = candidate_detection[0]
        boxes = detection.get("bboxes", default=[])
        labels = detection.get("labels", default=[])
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = [int(i) for i in box]
            # Draw the bounding box with red lines
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            cv2.putText(
                image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, 255
            )
        return image

    def create_detection_frames(self, detections: List[Detection]) -> List[np.ndarray]:
        # if we are recording, thake detection image buffer, otherwise take lookback frames
        if self._recording:
            images = [self._create_detection_frame(detections, frame) for frame in self._detection_image_buffer]
        else:
            images = [self._create_detection_frame(detections, frame) for frame in self._look_back_frames]
        self._detection_image_buffer = []
        return images

    def _update_lookback_frames(self, frame:Frame):
        self._look_back_frames.append(frame)
        self._detection_image_buffer.append(frame)
        # lookbackframes can be shorter than the longest slack
        if len(self._look_back_frames) > self._look_back_frames_limit:
            self._look_back_frames = self._look_back_frames[
                -self._look_back_frames_limit :
            ]
        # dettection image buffer should be bounded by slack
        if len(self._detection_image_buffer) > self._slack:
            self._detection_image_buffer = self._detection_image_buffer[
                -self._slack :
            ]

    def _destroy_writers(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        if self._detection_writer:
            self._detection_writer.release()
            self._detection_writer = None

    def _write_detections(self, detection_data):
        if self._detection_writer is None:
            self._detection_writer = self._detection_writer_factory(
                self._get_detection_output_file(), self._fps, self._frame_size
            )
        detection_frames = self.create_detection_frames(detection_data)
        if detection_frames is not None:
            for detection_frame in detection_frames:
                self._detection_writer.write(detection_frame)

    def register_frame(self, frame:Frame):
        self._update_lookback_frames(frame)
        if self._stop_recording_in > 0:
            self._writer.write(frame.image)
            self._stop_recording_in -= 1
        elif self._writer is not None:
            self._event_manager.log("recording_stopped", "event recording stopped")
            # write detection buffer images (if any)
            self._write_detections([])
            self._recording = False
            self._destroy_writers()

    def register_detection(self, detection_data):
        if self._writer:
            self._stop_recording_in = self._slack
            self._write_detections(detection_data)
            self._look_back_frames = []
        else:
            self._event_manager.log("recording_started", "event recording started")
            self._writer = self._writer_factory(
                self._get_recording_output_file(), self._fps, self._frame_size
            )
            self._stop_recording_in = self._slack
            # write look back frames
            for frame in self._look_back_frames:
                self._writer.write(frame.image)
            # write detection data to a file
            self._write_detections(detection_data)
            self._look_back_frames = []
            self._recording = True
