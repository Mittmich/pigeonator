"""Collection of recorder objects"""
import os
from datetime import datetime
import cv2
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from birdhub.video import Stream, VideoWriter
from birdhub.orchestration import Mediator
from birdhub.detection import Detection
from birdhub.logging import logger


class Recorder:
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

    def register_frame(self):
        pass

    def register_detection(self):
        pass

    def register_effect_action(self):
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

    def register_frame(self, frame):
        self._writer.write(frame)


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

    def _get_detection_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}_detections.avi")

    def _create_detection_frame(self, detection: Detection):
        image = detection.get("source_image")
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
        return [self._create_detection_frame(d) for d in detections]

    def _update_lookback_frames(self, frame):
        self._look_back_frames.append(frame)
        if len(self._look_back_frames) > self._look_back_frames_limit:
            self._look_back_frames = self._look_back_frames[
                -self._look_back_frames_limit :
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

    def register_frame(self, frame):
        self._update_lookback_frames(frame)
        if self._stop_recording_in > 0:
            self._writer.write(frame)
            self._stop_recording_in -= 1
        elif self._writer is not None:
            self._event_manager.log("recording_stopped", "event recording stopped")
            self._destroy_writers()

    def register_detection(self, detection_data):
        if self._writer:
            self._stop_recording_in = self._slack
            self._write_detections(detection_data)
        else:
            self._event_manager.log("recording_started", "event recording started")
            self._writer = self._writer_factory(
                self._get_recording_output_file(), self._fps, self._frame_size
            )
            self._stop_recording_in = self._slack
            self._recording = True
            # write look back frames
            for frame in self._look_back_frames:
                self._writer.write(frame)
            self._look_back_frames = []
            # write detection data to a file
            self._write_detections(detection_data)
