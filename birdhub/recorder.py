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

    def __init__(self, stream_url: str, motion_detector: MotionDetector, slack:int=100, activation_frames:int=3) -> None:
        self._stream_url = stream_url
        self._detector = motion_detector
        self._slack = slack
        self._activation_frames = activation_frames
    
    def record(self, outputDir: str, fps: int = 10) -> None:
        with Stream(self._stream_url) as stream:
            logger.info(f"Recording to {outputDir}")
            previous_frame = None
            writer = None
            stop_recording_in = 0
            motion_frames = 0
            look_back_frames = []
            for frame in stream:
                look_back_frames.append(frame)
                if previous_frame is not None:
                    rect = self._detector.detect(frame, previous_frame)
                else:
                    rect = []
                if rect and writer is None:
                    if motion_frames < self._activation_frames:
                        motion_frames += 1
                    else:
                        logger.info("Motion detected")
                        output_file = os.path.join(outputDir, f"{self._get_timestamp()}.avi")
                        writer = VideoWriter(output_file, fps, stream.frameSize)
                        logger.info("   Writing lookback frames")
                        for look_back_frame in look_back_frames:
                            writer.write(look_back_frame)
                if rect and writer is not None:
                    stop_recording_in = self._slack
                else:
                    stop_recording_in -= 1
                # write frame to file if needed
                if stop_recording_in > 0:
                    writer.write(frame)
                else:
                    if writer is not None:
                        logger.info("   Recording stopped")
                        motion_frames = 0
                        writer.release()
                        writer = None
                previous_frame = frame
                if len(look_back_frames) > self._activation_frames:
                    look_back_frames = look_back_frames[-self._activation_frames:]