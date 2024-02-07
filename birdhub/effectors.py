"""Effectors that can be used to deter birds"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from multiprocessing import Process, Pipe
from datetime import timedelta, datetime
from playsound import playsound
from birdhub.orchestration import Mediator
from birdhub.detection import Detection


class Effector(ABC):
    def __init__(
        self, target_class: str, cooldown_time: timedelta, config: Optional[Dict] = None
    ) -> None:
        self._event_manager_connection = None
        self._target_class = target_class
        self._cooldown_time = cooldown_time
        self._last_activation = None
        self._config = config

    def add_event_manager(self, event_manager: Mediator):
        # create commuinication pipe
        self._event_manager_connection, child_connection = Pipe()
        # register pipe with event manager
        event_manager.register_pipe("detector", child_connection)

    def run(self):
        """Start the detector process"""
        self._process = Process(target=self._run)
        self._process.start()

    def _run(self):
        """Run the effector"""
        while True:
            if self._event_manager_connection.poll(1):
                data = self._event_manager_connection.recv()
                self.register_detection(data)

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
                detection_time = detection.get("frame_timestamp", None)
                if detection_time is not None and isinstance(detection_time, datetime):
                    detection_time = detection_time.isoformat(
                        sep=" ", timespec="milliseconds"
                    )
                self._event_manager_connection.send(
                    (
                        "effect_activated",
                        {
                            "timestamp": activation_time,
                            "type": "Mock Effect",
                            "meta_information": {
                                "type": "mock",
                                "target_class": self._target_class,
                                "detection_timestamp": detection_time,
                            },
                        },
                    )
                )
                self._last_activation = activation_time


class SoundEffector(Effector):
    """Plays specified sound"""

    def register_detection(self, data: Optional[List[Detection]]) -> None:
        """Register detection"""
        if data is None or len(data) == 0:
            return
        # iterate over detections in reverse order to get the most recent detection
        for detection in data[::-1]:
            if (
                self._get_most_likely_object(detection) == self._target_class
                and self.is_activation_allowed()
            ):
                activation_time = datetime.now()
                playsound(self._config["sound_file"])
                end_time = datetime.now()
                detection_time = detection.get("frame_timestamp", None)
                if detection_time is not None and isinstance(detection_time, datetime):
                    detection_time = detection_time.isoformat(
                        sep=" ", timespec="milliseconds"
                    )
                self._event_manager_connection.send(
                    (
                        "effect_activated",
                        {
                            "timestamp": datetime.now(),
                            "type": "Audio Effector",
                            "meta_information": {
                                "type": "audio_effector",
                                "target_class": self._target_class,
                                "sound_file": self._config["sound_file"],
                                "detecton_timestamp": detection_time,
                                "activation_timestamp": activation_time.isoformat(
                                    sep=" ", timespec="milliseconds"
                                ),
                                "end_timestamp": end_time.isoformat(
                                    sep=" ", timespec="milliseconds"
                                ),
                            },
                        },
                    )
                )
                self._last_activation = activation_time


EFFECTORS = {"mock": MockEffector, "sound": SoundEffector}
