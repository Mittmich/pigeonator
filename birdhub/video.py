"""A module to handle video streams and video files."""
import asyncio
from asyncio import Queue
import datetime
import logging
from typing import Optional
import cv2
import numpy as np
import torch
from cachetools import LRUCache
from shelved_cache import PersistentCache
from tempfile import NamedTemporaryFile
from threading import Lock
from threading import Thread
from birdhub.logging import logger
from birdhub.timestamp_extraction import DigitModel


class Frame:
    def __init__(
        self,
        timestamp: datetime.datetime,
        capture_time: Optional[datetime.datetime] = None,
    ):
        self.timestamp = timestamp
        if capture_time is None:
            self.capture_time = datetime.datetime.now()
        else:
            self.capture_time = capture_time


class ImageStore:
    """Object that holds the a maximum number of for a given delay. Meant to be shared
    between one producer and multiple consumers."""

    def __init__(self, number_images: int, persist=False):
        if persist:
            # create named temporary file
            filename = NamedTemporaryFile().name
            self._images = PersistentCache(LRUCache, filename=filename, maxsize=number_images)
        else:
            self._images = LRUCache(maxsize=number_images)
        self._lock = Lock()
    
    def put(self, timestamp: datetime.datetime, image: np.ndarray) -> None:
        with self._lock:
            self._images[timestamp] = image

    def get(self, timestamp: datetime.datetime) -> Optional[np.ndarray]:
        with self._lock:
            return self._images.get(timestamp)


class Stream:
    def __init__(
        self, streamurl, ocr_weights="../weights/ocr_v3.pt", write_timestamps=True
    ):
        self.streamurl = streamurl
        self.cap = None
        self._event_manager = None
        self._previous_timestamp = None
        self._frame_index = 0
        self._digit_model = DigitModel()
        self._digit_model.load_state_dict(
            torch.load(ocr_weights, map_location=torch.device("cpu"))
        )
        self._write_timestamps = write_timestamps
        self._process = None
        # start and stop stream to get frame size
        self.cap = cv2.VideoCapture(self.streamurl)
        self._frameSize = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.cap.release()
        self.cap = None


    def get_frame(self, image_store: ImageStore):
        ret, image = self.cap.read()
        timestamp = self._get_timestamp(image)
        # if self._frame_index % 10 == 0:
        #     timestamp = self._get_timestamp(frame)
        #     if timestamp is None:
        #         timestamp = self._previous_timestamp
        #     self._previous_timestamp = timestamp
        #     self._frame_index = 0
        # else:
        #     # add index to microsecond part of timestamp to make it unique
        #     timestamp = self._previous_timestamp + datetime.timedelta(
        #         microseconds=self._frame_index
        #     )
        # self._frame_index += 1
        # create frame messag object
        frame = Frame(timestamp, datetime.datetime.now())
        if self._write_timestamps:
            self._write_timestamp(frame, image)
        # image to store
        image_store.put(timestamp, image)
        return frame

    def _get_timestamp(self, frame):
        try:
            timestamp = datetime.datetime.now()
        except ValueError as e:
            self._event_manager.log("timestamp_error", None, level=logging.INFO)
            logger.warning("Could not extract timestamp from frame: {}".format(e))
            # write frame to file for debugging
            now = datetime.datetime.now()
            cv2.imwrite(
                f"train_model/raw_data/timestamp_errors/timestamp_error_{now.strftime('%H:%M:%S')}.jpg",
                frame,
            )
            timestamp = None
        return timestamp

    def run(self, event_queue: Queue, image_store: ImageStore):
        """Start the stream and add new frames to the queue."""
        self._process = Thread(target=self._async_wrapper, args=(event_queue,image_store,))
        self._process.start()

    def _async_wrapper(self, event_queue: Queue, image_store: ImageStore):
        """Start the stream and add new frames to the queue."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run(event_queue, image_store))
        loop.close()

    async def _run(self, event_queue: Queue, image_store: ImageStore):
        """Start the stream and add new frames to the queue."""
        # initialize stream
        self.cap = cv2.VideoCapture(self.streamurl)
        logger.log_event("stream_started", None, level=logging.INFO)
        while True:
            await event_queue.put(("video_frame", self.get_frame(image_store)))

    def _write_timestamp(self, frame, image):
        cv2.putText(
            image,
            "O: " + frame.timestamp.strftime("%H:%M:%S"),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "C: " + frame.capture_time.strftime("%H:%M:%S,%f"),
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()

    def __next__(self):
        return self.get_frame()

    @property
    def frameSize(self):
        return self._frameSize

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
        self.videoWriter = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"MJPG"), fps, frameSize
        )

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
