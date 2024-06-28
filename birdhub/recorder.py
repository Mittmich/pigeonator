"""Collection of recorder objects"""
from abc import ABC, abstractmethod
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import cv2
from typing import List, Tuple
from multiprocessing import Pipe
from threading import Thread
import numpy as np
from birdhub.video import VideoWriter, Frame, ImageStore
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
        self._image_store = None

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_recording_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}.mp4")

    def add_event_manager(self, event_manager: Mediator):
        # create commuinication pipe
        self._event_manager_connection, child_connection = Pipe()
        # register pipe with event manager
        event_manager.register_pipe("recorder", child_connection)

    def run(self, image_store: ImageStore):
        """Start the detector process"""
        self._image_store = image_store
        self._process = Thread(target=self._run)
        self._process.start()

    def _run(self):
        while True:
            data = self._event_manager_connection.recv()
            # check if data is a frame or detection
            if isinstance(data, Frame):
                self.register_frame(data)

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
            self._get_recording_output_file() ,self._fps, frame_size
        )

    def register_frame(self, frame: Frame):
        # retrieve image from image store
        image = self._image_store.get(frame.timestamp)
        if image is not None:
            self._writer.write(image)


class EventRecorder(Recorder):
    def __init__(
        self,
        outputDir: str,
        frame_size: Tuple[int],
        fps: int = 10,
        slack: int = 100,
        look_back_frames: int = 3,
        detection_slack: int = 200,
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
        self._detection_frame_buffer = []
        self._detections = []
        self._recording = False
        self._detection_slack = detection_slack
        self._activations = []
        self._current_detection_video_file = None

    def _get_detection_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}_detections.mp4")

    def _create_detection_frame(
        self, detections: List[Detection], frame: Frame
    ) -> Frame:
        # retrieve image from buffer
        image = self._image_store.get(frame.timestamp)
        if image is None:
            return frame
        candidate_detection = [
            d for d in detections if d.frame_timestamp == frame.timestamp
        ]
        if len(candidate_detection) > 0:
            detection = candidate_detection[0]
            boxes = detection.get("bboxes", default=[])
            labels = detection.get("labels", default=[])
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = [int(i) for i in box]
                # Draw the bounding box with red lines
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            # put image back to buffer
            self._image_store.put(frame.timestamp, image)
        return frame

    def create_detection_frames(self, detections: List[Detection]) -> List[np.ndarray]:
        # if we are recording, take detection image buffer, otherwise take lookback frames
        if self._recording:
            images = [
                self._create_detection_frame(detections, frame)
                for frame in self._detection_frame_buffer
            ]
        else:
            images = [
                self._create_detection_frame(detections, frame)
                for frame in self._look_back_frames
            ]
        self._detection_frame_buffer = []
        return images

    def _update_lookback_frames(self, frame: Frame):
        self._look_back_frames.append(frame)
        self._detection_frame_buffer.append(frame)
        # lookbackframes can be shorter than the longest slack
        if len(self._look_back_frames) > self._look_back_frames_limit:
            self._look_back_frames = self._look_back_frames[
                -self._look_back_frames_limit :
            ]
        # dettection frame buffer should be bounded by slack
        if len(self._detection_frame_buffer) > self._slack:
            self._detection_frame_buffer = self._detection_frame_buffer[-self._slack :]

    def _destroy_writers(self):
        if self._writer:
            self._writer.release()
            self._writer = None
        if self._detection_writer:
            self._detection_writer.release()
            self._detection_writer = None
            self._current_detection_video_file = None

    def _update_detections(self, detection_data):
        detection_frames = self.create_detection_frames(detection_data)
        current_timestamps = set(i.timestamp for i in self._detections)
        filtered_detection_frames = [
            i for i in detection_frames if i.timestamp not in current_timestamps
        ]
        self._detections.extend(filtered_detection_frames)

    def _write_detections(self):
        # add all activations that are recorded to the detections
        for activation in self._activations:
            write_timestamps = [
                i.timestamp
                for i in self._detections
                if i.timestamp > activation["timestamp"]
                and (i.timestamp - activation["timestamp"]) < timedelta(seconds=2)
            ]
            for frame in self._detections:
                self._add_activation(frame, activation, write_timestamps)
        if self._detection_writer is None:
            self._current_detection_video_file = self._get_detection_output_file()
            self._detection_writer = self._detection_writer_factory(
                self._current_detection_video_file, self._fps, self._frame_size
            )
        for detection_frame in self._detections:
            # retrive image from buffer
            image = self._image_store.get(detection_frame.timestamp)
            if image is not None:
                self._detection_writer.write(image)
        self._detections = []
        self._activations = []

    def _add_activation(
        self, frame: Frame, data: dict, write_timestamps: List[datetime.timestamp]
    ):
        if frame.timestamp in write_timestamps:
            # get image from buffer
            image = self._image_store.get(frame.timestamp)
            if image is not None:
                # get boundary of this text
                textsize = cv2.getTextSize(data["type"], cv2.FONT_HERSHEY_DUPLEX, 7, 2)[0]
                # get coords based on boundary
                textX = (image.shape[1] - textsize[0]) // 2
                textY = (image.shape[0] + textsize[1]) // 2
                cv2.putText(
                    image,
                    data["type"],
                    (textX, textY),
                    cv2.FONT_HERSHEY_DUPLEX,
                    7,
                    (0, 0, 255),
                    2,
                )
                # write image back to buffer
                self._image_store.put(frame.timestamp, image)

    def register_effect_activation(self, data: dict):
        self._activations.append(data)

    def register_frame(self, frame: Frame):
        # get image from image store
        image = self._image_store.get(frame.timestamp)
        if image is None:
            return
        # add frame to persisted cache
        self._image_store.put(frame.timestamp, image)
        self._update_lookback_frames(frame)
        # decide whether to write detection frames
        if len(self._detections) > self._detection_slack:
            self._write_detections()
        if self._stop_recording_in > 0:
            self._writer.write(image)
            self._stop_recording_in -= 1
        elif self._writer is not None:
            # log end of recording
            logger.log_event(
                        "recording_stopped",
                        "event recording stopped",
                        logging.INFO
            )
            # sebd end of recording event to event manager
            self._event_manager_connection.send(
                ("recording_stopped", 
                    {
                        "recording_end_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "recording_file": self._current_detection_video_file,
                        "recording_timestamp": Path(self._current_detection_video_file).name.split("_")[0]
                    }
                )
            )
            # write detections
            self._update_detections([])
            self._write_detections()
            self._recording = False
            self._destroy_writers()

    def register_detection(self, detection_data):
        if self._writer:
            self._stop_recording_in = self._slack
            self._update_detections(detection_data)
            self._look_back_frames = []
        else:
            # log start of recording
            logger.log_event(
                        "recording_started",
                        "event recording started",
                        logging.INFO
            )
            self._writer = self._writer_factory(
                self._get_recording_output_file(), self._fps, self._frame_size
            )
            self._stop_recording_in = self._slack
            # write look back frames
            for frame in self._look_back_frames:
                # get image from buffer
                image = self._image_store.get(frame.timestamp)
                if image is not None:
                    self._writer.write(image)
            # write detection data to a file
            self._update_detections(detection_data)
            self._look_back_frames = []
            self._recording = True
    
    def _run(self):
        while True:
            data = self._event_manager_connection.recv()
            # check data type
            if isinstance(data, Frame):
                self.register_frame(data)
            elif isinstance(data, list):
                # list of detections
                self.register_detection(data)
            elif isinstance(data, dict):
                # activation data
                self.register_effect_activation(data)
