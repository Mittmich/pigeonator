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

    def __init__(self, stream_url: str, motion_detector: MotionDetector, slack:int=100) -> None:
        self._stream_url = stream_url
        self._detector = motion_detector
        self._slack = slack
    
    def record(self, outputDir: str, fps: int = 10) -> None:
        with Stream(self._stream_url) as stream:
            logger.info(f"Recording to {outputDir}")
            previous_frame = None
            writer = None
            stop_recording_in = 0
            for frame in stream:
                if previous_frame is not None and self._detector.detect(frame, previous_frame) and stop_recording_in == 0:
                    logger.info("Motion detected")
                    output_file = os.path.join(outputDir, f"{self._get_timestamp()}.avi")
                    writer = VideoWriter(output_file, fps, stream.frameSize)
                    stop_recording_in = self._slack
                elif stop_recording_in > 0:
                    stop_recording_in -= 1
                if stop_recording_in > 0:
                    writer.write(frame)
                else:
                    if writer is not None:
                        logger.info("   Recording stopped")
                        writer.release()
                        writer = None
                previous_frame = frame