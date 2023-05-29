"""Functionality to orchestrate streams, detectors, recorders, and other components."""

from abc import ABC, abstractmethod
from typing import Optional
from video import Stream
from recorder import Recorder
from detection import Detector
from effectors import Effector

class Mediator(ABC):
    """
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    """
    @abstractmethod
    def notify(self, sender: object, event: str) -> None:
        pass


class VideoEventManager(Mediator):
    
    def __init__(self, stream: Stream, recorder: Optional[Recorder]=None, detector: Optional[Detector]=None, effector:Optional[Effector]=None) -> None:
        super().__init__()
        self._stream = stream
        self._recorder = recorder
        self._detector = detector
        self._effector = effector

