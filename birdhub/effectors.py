"""Effectors that can be used to deter birds"""
from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import timedelta, datetime
from birdhub.orchestration import Mediator
from birdhub.detection import Detection


class Effector(ABC):
    def __init__(self, target_class: str, cooldown_time: timedelta) -> None:
        self._event_manager = None
        self._target_class = target_class
        self._cooldown_time = cooldown_time
        self._last_activation = None

    def add_event_manager(self, event_manager: Mediator):
        self._event_manager = event_manager

    @abstractmethod
    def register_detection(self, data: Optional[List[Detection]]) -> None:
        """Register detection"""
        raise NotImplementedError

    def _get_time_since_last_activation(self) -> timedelta:
        """Get time since last activation"""
        if self._last_activation is None:
            return timedelta.max
        else:
            return datetime.now() - self._last_activation

    def is_activation_allowed(self) -> bool:
        """Check if activation is allowed"""
        return self._get_time_since_last_activation() > self._cooldown_time


class MockEffector(Effector):
    """Mock effector that only logs when it is activated or deactivated"""

    def register_detection(self, data: Optional[List[Detection]]) -> None:
        """Register detection"""
        if data is None or len(data) == 0:
            return
        for detection in data:
            if (
                detection.get("most_likely_object", None) == self._target_class
                and self.is_activation_allowed()
            ):
                self._event_manager.log(
                    "effect_activated", {"type": "mock", "target_class": self._target_class}
                )
                self._last_activation = datetime.now()
