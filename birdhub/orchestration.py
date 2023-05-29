"""Functionality to orchestrate streams, detectors, recorders, and other components."""

from abc import ABC, abstractmethod
from typing import Optional
from birdhub.logging import logger


class Mediator(ABC):
    """
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    """
    @abstractmethod
    def notify(self, event: str, data: object) -> None:
        pass


class VideoEventManager(Mediator):
    
    def __init__(self, stream: 'Stream', recorder: Optional['Recorder']=None, detector: Optional['Detector']=None, effector:Optional['Effector']=None) -> None:
        self._stream = stream
        self._recorder = recorder
        self._detector = detector
        self._effector = effector
        # register mediator object
        self._stream.add_event_manager(self)
        if self._recorder is not None:
            self._recorder.add_event_manager(self)
        if self._detector is not None:
            self._detector.mediater = self
        if self._effector is not None:
            self._effector.mediater = self


    def notify(self, event: str, data: object) -> None:
        if event == "video_frame":
            if self._detector is not None:
                self._detector.detect(data)
            if self._recorder is not None:
                self._recorder.register_frame(data) # This is needed for lookback recording
        if event == "detection_start":
            logger.log_event("detection_start", data.get("meta_information", None))
            if self._recorder is not None:
                self._recorder.register_start_recording(data)
                self._recorder.register_detection(data)
            if self._effector is not None:
                self._effector.activate(data)
        if event == "detection":
            logger.log_event("detection", data.get("meta_information", None))
            if self._recorder is not None:
                self._recorder.register_detection(data)
        if event == "detetction_stop":
            logger.log_event("detection_stop", data.get("meta_information", None))
            if self._effector is not None:
                self._effector.deactivate()
            if self._recorder is not None:
                self._recorder.register_stop_recording()
        if event == "effect_activated":
            logger.log_event("effect_activated", data.get("meta_information", None))
            if self._recorder is not None:
                self._recorder.register_effect_action(data)