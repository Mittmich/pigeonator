"""Collection of recorder objects"""
from abc import ABC, abstractmethod
import os
from datetime import datetime
from typing import List, Tuple, Optional
from birdhub.video import Stream, VideoWriter
from birdhub.orchestration import Mediator
from birdhub.logging import logger

class Recorder():
    
    def __init__(self, outputDir: str, frame_size: Tuple[int], fps: int = 10) -> None:
        self._outputDir = outputDir
        self._frame_size = frame_size
        self._fps = fps
        self._event_manager = None


    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_recording_output_file(self):
        return os.path.join(self._outputDir, f"{self._get_timestamp()}.avi")

    def add_event_manager(self, event_manager: Mediator):
        self._event_manager = event_manager

    def register_start_recording(self):
        pass

    def register_stop_recording(self):
        pass

    def register_frame(self):
        pass

    def register_detection(self):
        pass

    def register_effect_action(self):
        pass


class ContinuousRecorder(Recorder):

    def __init__(self, outputDir: str, frame_size: Tuple[int], fps: int = 10) -> None:
        super().__init__(outputDir, fps)
        logger.log_event("recording_started", "Continuous recording started")
        self._writer = VideoWriter(self._get_recording_output_file(), self._fps, frame_size)

    def register_frame(self, frame):
        self._writer.write(frame)


# class MotionRecoder(Recorder):

#     def __init__(self, stream_url: str, motion_detector: MotionDetector, slack:int=100, activation_frames:int=3) -> None:
#         self._stream_url = stream_url
#         self._detector = motion_detector
#         self._slack = slack
#         self._activation_frames = activation_frames
#         self._previous_frame = None
#         self._stop_recording_in = 0
#         self._motion_frames = 0
#         self._look_back_frames = []
#         self._writer = None

#     def _initialize_video_writer(self, outputDir, fps, frameSize):
#         logger.log_event("motion_detected","Motion detected")
#         output_file = os.path.join(outputDir, f"{self._get_timestamp()}.avi")
#         self._writer = VideoWriter(output_file, fps, frameSize)
#         for look_back_frame in self._look_back_frames:
#             self._writer.write(look_back_frame)
    
#     def _destroy_video_writer(self):
#         if self._writer is not None:
#             logger.log_event("recording_stopped","Recording stopped")
#             self._motion_frames = 0
#             self._writer.release()
#             self._writer = None

    
#     def record(self, outputDir: str, fps: int = 10) -> None:
#         with Stream(self._stream_url) as stream:
#             logger.log_event("recording_init",f"Recording to {outputDir}")
#             for frame in stream:
#                 self._look_back_frames.append(frame)
#                 rect = self._detector.detect(frame, self._previous_frame)
#                 if rect and self._writer is None:
#                     if self._motion_frames < self._activation_frames:
#                         self._motion_frames += 1
#                     else:
#                         self._initialize_video_writer( outputDir, fps, stream.frameSize)
#                 if rect and self._writer is not None:
#                     self._stop_recording_in = self._slack
#                 if not rect and self._writer is not None:
#                     self._stop_recording_in -= 1
#                 if not rect and self._writer is None:
#                     self._motion_frames = 0
#                 # write frame to file if needed
#                 if self._stop_recording_in > 0:
#                     self._writer.write(frame)
#                 else:
#                     self._destroy_video_writer()
#                 self._previous_frame = frame
#                 if len(self._look_back_frames) > self._activation_frames:
#                     self._look_back_frames = self._look_back_frames[-self._activation_frames:]