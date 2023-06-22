"""Effectors that can be used to deter birds"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from multiprocessing import Process
from datetime import timedelta, datetime
from playsound import playsound
from birdhub.orchestration import Mediator
from birdhub.detection import Detection


class Effector(ABC):
    def __init__(self, target_class: str, cooldown_time: timedelta, config: Optional[Dict]=None) -> None:
        self._event_manager = None
        self._target_class = target_class
        self._cooldown_time = cooldown_time
        self._last_activation = None
        self._config = config

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

    def _get_most_likely_object(self, data: Detection) -> Optional[str]:
        meta_data = data.get("meta_information", {})
        return meta_data.get("most_likely_object", None)

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
                self._get_most_likely_object(detection) == self._target_class
                and self.is_activation_allowed()
            ):
                activation_time = datetime.now()
                self._event_manager.notify("effect_activated", {'timestamp': activation_time, 'type': 'Mock Effect',
                                                                'meta_information': {"type": "mock", "target_class": self._target_class}})
                self._last_activation = activation_time


class SoundEffector(Effector):
    """Plays speciried sound"""

    def register_detection(self, data: Optional[List[Detection]]) -> None:
        """Register detection"""
        if data is None or len(data) == 0:
            return
        for detection in data:
            if (
                self._get_most_likely_object(detection) == self._target_class
                and self.is_activation_allowed()
            ):
                activation_time = datetime.now()
                playsound(self._config["sound_file"])
                self._event_manager.notify("effect_activated", {'timestamp': datetime.now(), 'type': 'Audio Effector',
                                                                'meta_information': {"type": "audio_effector", "target_class": self._target_class, "sound_file": self._config["sound_file"]}})
                self._last_activation = activation_time



EFFECTORS = {"mock": MockEffector, "sound": SoundEffector}