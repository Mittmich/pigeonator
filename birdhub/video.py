"""A module to handle video streams and video files."""
import datetime
import logging
from typing import Optional
import cv2
import numpy as np
import torch
from birdhub.orchestration import Mediator
from birdhub.logging import logger
from birdhub.timestamp_extraction import DigitModel

class Frame:
    def __init__(self, image: np.ndarray, timestamp: datetime.datetime, capture_time: Optional[datetime.datetime] = None):
        self.image = image
        self.timestamp = timestamp
        if capture_time is None:
            self.capture_time = datetime.datetime.now()
        else:
            self.capture_time = capture_time


class Stream:

    def __init__(self, streamurl, ocr_weights="../weights/ocr_v3.pt", write_timestamps=True):
        self.streamurl = streamurl
        self.cap = cv2.VideoCapture(self.streamurl)
        self._event_manager = None
        self._previous_timestamp = None
        self._frame_index = 0
        self._digit_model = DigitModel()
        self._digit_model.load_state_dict(torch.load(ocr_weights, map_location=torch.device('cpu')))
        self._write_timestamps = write_timestamps

    def get_frame(self):
        ret, frame = self.cap.read()
        if self._frame_index % 10 == 0:
            timestamp = self._get_timestamp(frame)
            if timestamp is None:
                timestamp = self._previous_timestamp
            self._previous_timestamp = timestamp
            self._frame_index = 0
        else:
            # add index to microsecond part of timestamp to make it unique
            timestamp = self._previous_timestamp + datetime.timedelta(microseconds=self._frame_index)
        self._frame_index += 1
        frame = Frame(frame, timestamp, datetime.datetime.now())
        if self._write_timestamps:
            self._write_timestamp(frame)
        return frame

    def _get_timestamp(self, frame):
        try:
            timestamp = self._digit_model.get_timestamp(frame)
        except ValueError as e:
            self._event_manager.log("timestamp_error", None, level=logging.INFO)
            logger.warning("Could not extract timestamp from frame: {}".format(e))
            # write frame to file for debugging
            now = datetime.datetime.now()
            cv2.imwrite(f"train_model/raw_data/timestamp_errors/timestamp_error_{now.strftime('%H:%M:%S')}.jpg", frame)
            timestamp = None
        return timestamp

    def add_event_manager(self, event_manager: Mediator):
        self._event_manager = event_manager

    def _write_timestamp(self, frame):
        if frame.timestamp is None:
            return
        cv2.putText(frame.image, 'O: ' + frame.timestamp.strftime("%H:%M:%S,%f"), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame.image, 'C: ' + frame.capture_time.strftime("%H:%M:%S,%f"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def stream(self):
        self._event_manager.log("stream_started", None)
        while True:
            self._event_manager.notify("video_frame", self.get_frame())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()

    def __next__(self):
        return self.get_frame()
    
    @property
    def frameSize(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self):
        return self
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


class VideoWriter:
    """A class to write video frames to a file."""

    def __init__(self, filename, fps, frameSize):
        """Initialize the VideoWriter object."""
        self.filename = filename
        self.fps = fps
        self.frameSize = frameSize
        self.videoWriter = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'), fps, frameSize)

    def write(self, frame):
        """Write a frame to the video file."""
        self.videoWriter.write(frame)

    def __enter__(self):
        """Return the VideoWriter object."""
        return self
    
    def release(self):
        """Release the VideoWriter object."""
        self.videoWriter.release()

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the VideoWriter object."""
        self.videoWriter.release()

    def __del__(self):
        """Destroy the VideoWriter object."""
        self.videoWriter.release()