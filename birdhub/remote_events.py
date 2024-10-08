"""Module responsble for sending events to the remote server."""

import requests
from threading import Thread
from typing import Optional
from pathlib import Path
import shutil
import subprocess
from multiprocessing import Pipe
from birdhub.orchestration import Mediator
from requests.exceptions import RequestException
from requests.auth import HTTPBasicAuth
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
            logger.error(f"Error sending data to remote server: {e}\nRequest body: {response.json()}")

    return wrapper


@log_request_error
def send_detection(server_address: str, data: list[Detection], user: str = None, password: str = None, verify_ssl: bool = True):
    """Send detection data to the remote server."""
    # get first detection to derive the timestamp
    data = data[0]
    return requests.post(
        f"{server_address}/detections/",
        json={
            "detections": [
                {
                    "detected_class": data.get('meta_information')["most_likely_object"],
                    "detection_timestamp": data.get("frame_timestamp").strftime("%Y-%m-%dT%H:%M:%S"),
                    "confidence": data.get("meta_information")['mean_confidence'],
                }
            ],
        },
        verify=verify_ssl,
        auth=HTTPBasicAuth(user, password) if user and password else None,
    )


@log_request_error
def send_effect_activated(server_address: str, data: dict, user: str = None, password: str = None, verify_ssl: bool = True):
    """Send effect activated data to the remote server."""
    return requests.post(
        f"{server_address}/effectorAction/",
        json={
            "action": data["type"],
            "action_metadata": data["meta_information"],
            "detection_timestamp": data["meta_information"]["detection_timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
            "action_timestamp": data["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
        },
        verify=verify_ssl,
        auth=HTTPBasicAuth(user, password) if user and password else None,
    )


@log_request_error
def send_recording_stopped(server_address: str, data: dict, user: str = None, password: str = None, verify_ssl: bool = True):
    """Send recording stopped data to the remote server."""
    # create smaller video using ffmpeg
    small_video_file = data["recording_file"] + "_small.mp4"
    subprocess.run(
        [
            # ffmpeg -i test.mp4 -c:v libx265 -s 640x320 -crf 27 -c:a copy output.mp4
            "ffmpeg",
            "-i",
            data["recording_file"],
            "-c:v",
            "libx265",
            "-s",
            "640x320",
            "-crf",
            "27",
            "-c:a",
            "copy",
            small_video_file,
        ],
        check=True,
    )
    with open(small_video_file, "rb") as f:
        return requests.post(
            f"{server_address}/recordings/",
            files={
                "file": (Path(data["recording_file"]).name, f, "video/mp4"),
            },
            data={
                "recording_timestamp": data["recording_timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "recording_end_timestamp": data["recording_end_timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
            },
            verify=verify_ssl,
            auth=HTTPBasicAuth(user, password) if user and password else None,
        )
    # remove small video file
    shutil.rmtree(small_video_file)


class EventDispatcher:
    """Responsible for sending events to the remote server."""

    EVENT_HANDLERS = {
        "detection": send_detection,
        "effect_activated": send_effect_activated,
        "recording_stopped": send_recording_stopped,
    }

    def __init__(self, server_address: str, listening_for: Optional[list[str]] = None, user: str = None, password: str = None, verify_ssl: bool = True):
        """Initialize the EventDispatcher object."""
        self._server_address = server_address
        self._user = user
        self._password = password
        self._verify_ssl = verify_ssl
        if listening_for is None:
            listening_for = self.EVENT_HANDLERS.keys()
        if not all(
            [event in self.EVENT_HANDLERS for event in listening_for]
        ):
            raise ValueError("Invalid event type.")
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
