"""Functionality to orchestrate streams, detectors, recorders, and other components."""

from abc import ABC, abstractmethod
from typing import Optional
from multiprocessing import Queue, Pipe
import logging
from birdhub.logging import logger


class Mediator(ABC):
    """
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    """

    @abstractmethod
    def log(self, event: str, message: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def notify(self, event: str, data: object) -> None:
        pass


class VideoEventManager(Mediator):
    def __init__(
        self,
        stream: "Stream",
        recorder: Optional["Recorder"] = None,
        detector: Optional["Detector"] = None,
        effector: Optional["Effector"] = None,
        throttle_detection: int = 10,
    ) -> None:
        self._stream = stream
        self._recorder = recorder
        self._detector = detector
        self._effector = effector
        self._throttle_detection = throttle_detection
        self._detections_logged = 0
        self._pipes = {}
        # register mediator object
        if self._recorder is not None:
            self._recorder.add_event_manager(self)
        if self._detector is not None:
            self._detector.add_event_manager(self)
        if self._effector is not None:
            self._effector.add_event_manager(self)

    def log(
        self, event: str, message: Optional[str] = None, level=logging.INFO
    ) -> None:
        if event == "detection":
            self._detections_logged += 1
            if self._detections_logged % self._throttle_detection == 0:
                message["accumulation_count"] = self._throttle_detection
                logger.log_event(event, message, level=level)
        elif event == "recording_stopped":
            self._detections_logged = 0
            logger.log_event(event, message, level=level)
        else:
            logger.log_event(event, message, level=level)

    def notify(self, event: str, data: object) -> None:
        if event == "video_frame":
            if self._detector is not None:
                self._pipes["detector"].send(data)
            if self._recorder is not None:
                self._pipes["recorder"].send(data)
        if event == "detection":
            self.log("detection", data[-1].get("meta_information", None))
            if self._recorder is not None:
                self._pipes["recorder"].send(data)
            if self._effector is not None:
                self._pipes["effector"].send(data)
        if event == "effect_activated":
            self.log("effect_activated", data.get("meta_information", None))
            if self._recorder is not None:
                self._recorder.register_effect_activation(data)
        if event == "log_request":
            log_event, message, level = data
            self.log(log_event, message, level)

    def register_pipe(self, name: str, pipe: Pipe):
        """Registers pipe with event manager."""
        self._pipes[name] = pipe

    def run(self):
        """Start orchestration loop and notify components about events."""
        event_queue = Queue()
        log_queue = Queue()
        # start all components
        self._stream.run(event_queue, log_queue)
        if self._detector is not None:
            self._detector.run()
        if self._recorder is not None:
            self._recorder.run()
        if self._effector is not None:
            self._effector.run()
        while True:
            # check stream
            if not event_queue.empty():
                event, data = event_queue.get()
                self.notify(event, data)
            if not log_queue.empty():
                event, message = log_queue.get()
                self.log(event, message)
            # check pipes
            for pipe in self._pipes.values():
                if pipe.poll():
                    event, data = pipe.recv()
                    self.notify(event, data)
