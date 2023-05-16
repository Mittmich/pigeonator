"""Collection of recorder objects"""
from abc import ABC, abstractmethod
import os
from datetime import datetime
from typing import List, Tuple, Optional
from birdhub.video import Stream, VideoWriter
from birdhub.motion_detection import MotionDetector
from birdhub.logging import logger

class Recorder(ABC):
    
    def __init__(self, stream_url: str) -> None:
        self._stream_url = stream_url
    

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def record(self, outputDir: str, fps: int = 10) -> None:
        raise NotImplementedError

class ContinuousRecorder(Recorder):

    def __init__(self, stream_url: str) -> None:
        self._stream_url = stream_url
    
    def record(self, outputDir: str, fps: int = 10) -> None:
        with Stream(self._stream_url) as stream:
            logger.info(f"Recording to {outputDir}")
            output_file = os.path.join(outputDir, f"{self._get_timestamp()}.avi")
            with VideoWriter(output_file, fps, stream.frameSize) as writer:
                for frame in stream:
                    writer.write(frame)


class MotionRecoder(Recorder):

    def __init__(self, stream_url: str, motion_detector: MotionDetector, file_prefix:Optional[str]=None, slack:int=10) -> None:
        self._stream_url = stream_url
        self._detector = motion_detector
        self._slack = slack