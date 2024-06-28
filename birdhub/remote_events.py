"""Module responsble for sending events to the remote server."""

import requests
from threading import Thread
from typing import Optional
from pathlib import Path
from multiprocessing import Pipe
from birdhub.orchestration import Mediator
from requests.exceptions import RequestException
from birdhub.detection import Detection
from birdhub.logging import logger

# Define functions to send events to the remote server


def log_request_error(func):
    """Decorator to log request errors."""
    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            response.raise_for_status()
            return response
        except RequestException as e:
            logger.error(f"Error sending data to remote server: {e}")

    return wrapper


@log_request_error
def send_detection(server_address: str, data: list[Detection]):
    """Send detection data to the remote server."""
    # get first detection to derive the timestamp
    data = data[0]
    return requests.post(
        f"{server_address}/detections/",
        json={
            "detections": [
                {
                    "detected_class": data.get('meta_information')["most_likely_object"],
                    "detection_timestamp": data.get("frame_timestamp"),
                    "confidence": data.get("mean_confidence"),
                }
            ],
        },
    )


@log_request_error
def send_effect_activated(server_address: str, data: dict):
    """Send effect activated data to the remote server."""
    return requests.post(
        f"{server_address}/effectorAction/",
        json={
            "action": data["type"],
            "action_metadata": data["meta_information"],
            "detection_timestamp": data["meta_information"]["detection_timestamp"],
            "action_timestamp": data["timestamp"],
        },
    )


@log_request_error
def send_recording_stopped(server_address: str, data: dict):
    """Send recording stopped data to the remote server."""
    with open(data["recording_file"], "rb") as f:
        return requests.post(
            f"{server_address}/recordings/",
            files={
                "file": (Path(data["recording_file"].name), f, "video/mp4"),
            },
            data={
                "recording_timestamp": data["recording_time"],
                "recording_end_timestamp": data["recording_end_timestamp"],
            },
        )


class EventDispatcher:
    """Responsible for sending events to the remote server."""

    EVENT_HANDLERS = {
        "detection": send_detection,
        "effect_activated": send_effect_activated,
        "recording_stopped": send_recording_stopped,
    }

    def __init__(self, server_address: str, listening_for: Optional[list[str]] = None):
        """Initialize the EventDispatcher object."""
        self._server_address = server_address
        if listening_for is not None and not all(
            [event in self.EVENT_HANDLERS for event in listening_for]
        ):
            raise ValueError("Invalid event type.")
        else:
            listening_for = self.EVENT_HANDLERS.keys()
        self._listening_for = listening_for

    def run(self):
        """Start the detector process"""
        self._process = Thread(target=self._run)
        self._process.start()

    def _run(self):
        """Run the effector"""
        while True:
            if self._event_manager_connection.poll():
                self.register_event(*self._event_manager_connection.recv())

    def add_event_manager(self, event_manager: Mediator):
        # create commuinication pipe
        self._event_manager_connection, child_connection = Pipe()
        # register pipe with event manager
        event_manager.register_pipe("event_dispatcher", child_connection)

    def register_event(self, event: str, data: dict):
        """Register event and send to remote server if dispatcher listens to events."""
        if event in self._listening_for:
            self.EVENT_HANDLERS[event](self._server_address, data)
