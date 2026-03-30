"""Effectors that can be used to deter birds"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from multiprocessing import Pipe
import os
import subprocess
import pygame
from threading import Thread
from datetime import timedelta, datetime
from birdhub.orchestration import Mediator
from birdhub.detection import Detection


class Effector(ABC):
    def __init__(
        self, target_classes: List[str], cooldown_time: timedelta, config: Optional[Dict] = None
    ) -> None:
        self._event_manager_connection = None
        self._target_classes = target_classes
        self._cooldown_time = cooldown_time
        self._last_activation = None
        self._config = config

    def add_event_manager(self, event_manager: Mediator):
        # create commuinication pipe
        self._event_manager_connection, child_connection = Pipe()
        # register pipe with event manager
        event_manager.register_pipe("effector", child_connection)

    def run(self):
        """Start the detector process"""
        self._process = Thread(target=self._run)
        self._process.start()

    def _run(self):
        """Run the effector"""
        while True:
            if self._event_manager_connection.poll():
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
                self._get_most_likely_object(detection) in self._target_classes
                and self.is_activation_allowed()
            ):
                activation_time = datetime.now()
                detection_time = detection.get("frame_timestamp", None)
                if detection_time is not None and isinstance(detection_time, datetime):
                    detection_time = detection_time.strftime("%Y-%m-%dT%H:%M:%S")
                self._event_manager_connection.send(
                    (
                        "effect_activated",
                        {
                            "timestamp": activation_time,
                            "type": "Mock Effect",
                            "meta_information": {
                                "type": "mock",
                                "target_classes": self._target_classes,
                                "detection_timestamp": detection_time,
                            },
                        },
                    )
                )
                self._last_activation = activation_time


class SoundEffector(Effector):
    """Plays specified sound"""

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

        audio_driver = self._config.get("sdl_audio_driver", "alsa")
        audio_device = self._config.get("alsa_device", "plughw:2,0")
        self._alsa_card_id = str(self._config.get("alsa_card_id", "2"))
        self._sound_volume = int(self._config.get("sound_volume", 90))
        # Map each ALSA control to its target volume percentage.
        # Channels is always fixed at 100%; Master follows sound_volume.
        self._volume_controls = self._config.get(
            "alsa_volume_controls",
            {"Master": self._sound_volume, "Channels": 100},
        )

        os.environ["SDL_AUDIODRIVER"] = audio_driver
        os.environ["AUDIODEV"] = audio_device

        pygame.init()
        # channels=1 forces mono mixing so the same signal is sent to every
        # physical speaker (important when only one speaker is connected).
        pygame.mixer.init(channels=1)
        self._sound = pygame.mixer.Sound(self._config["sound_file"])
        self._set_hardware_volume()

    def _set_hardware_volume(self) -> None:
        """Set hardware mixer volume for each configured control."""
        for control, percent in self._volume_controls.items():
            percent = max(0, min(100, int(percent)))
            try:
                subprocess.run(
                    ["amixer", "-c", self._alsa_card_id, "sset", str(control), f"{percent}%"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                # Keep running even if ALSA tools are unavailable on the current host.
                continue

    def register_detection(self, data: Optional[List[Detection]]) -> None:
        """Register detection"""
        if data is None or len(data) == 0:
            return
        # iterate over detections in reverse order to get the most recent detection
        for detection in data[::-1]:
            if (
                self._get_most_likely_object(detection) in self._target_classes
                and self.is_activation_allowed()
            ):
                activation_time = datetime.now()
                self._set_hardware_volume()
                self._sound.play()
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
                                "target_classes": self._target_classes,
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
