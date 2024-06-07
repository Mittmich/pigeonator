"""Functionality to orchestrate streams, detectors, recorders, and other components."""

from abc import ABC, abstractmethod
from typing import Optional
import asyncio
from asyncio import Queue
from multiprocessing import Pipe
from datetime import timedelta, datetime
import logging
from birdhub.logging import logger
from birdhub.video import ImageStore


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
    def __init__(
        self,
        stream: "Stream",
        recorder: Optional["Recorder"] = None,
        detector: Optional["Detector"] = None,
        effector: Optional["Effector"] = None,
        max_buffer_size: int = 500,
        max_delay: int = 5 # seconds
    ) -> None:
        self._stream = stream
        self._recorder = recorder
        self._detector = detector
        self._effector = effector
        self._detections_logged = 0
        self._pipes = {}
        self._event_queue = None
        self._max_delay = max_delay
        self._image_store = ImageStore(number_images=max_buffer_size)
        # register mediator object
        if self._recorder is not None:
            self._recorder.add_event_manager(self)
        if self._detector is not None:
            self._detector.add_event_manager(self)
        if self._effector is not None:
            self._effector.add_event_manager(self)


    async def notify(self, event: str, data: object) -> None:
        if event == "video_frame":
            # log size of queue
            #logger.log_event('queue_size', self._event_queue.qsize(), level=logging.DEBUG)
            # drop frames if they are not recent
            if datetime.now() - data.timestamp > timedelta(seconds=self._max_delay):
                return
            if self._detector is not None:
                self._pipes["detector"].send(data)
            if self._recorder is not None:
                self._pipes["recorder"].send(data)
        if event == "detection":
            logger.log_event("detection", data[-1].get("meta_information", None))
            if self._recorder is not None:
                self._pipes["recorder"].send(data)
            if self._effector is not None:
                self._pipes["effector"].send(data)
        if event == "effect_activated":
            logger.log_event("effect_activated", data.get("meta_information", None))
            if self._recorder is not None:
                self._pipes["recorder"].send(data)

    def register_pipe(self, name: str, pipe: Pipe):
        """Registers pipe with event manager."""
        self._pipes[name] = pipe

    async def process_notify(self, event_queue: Queue):
        """Notification worker"""
        while True:
            if not event_queue.empty():
                event, data = await event_queue.get()
                await self.notify(event, data)
                event_queue.task_done()
            # check pipes
            for pipe in self._pipes.values():
                if pipe.poll():
                    event, data = pipe.recv()
                    await self.notify(event, data)

    async def run(self):
        """Start orchestration loop and notify components about events."""
        self._event_queue = Queue(maxsize=500)
        # start all components
        self._stream.run(self._event_queue, self._image_store)
        if self._detector is not None:
            self._detector.run(self._image_store)
        if self._recorder is not None:
            self._recorder.run(self._image_store)
        if self._effector is not None:
            self._effector.run()
        # start workers
        tasks = [
            asyncio.create_task(self.process_notify(self._event_queue)),
            asyncio.create_task(self.process_notify(self._event_queue)),
            asyncio.create_task(self.process_notify(self._event_queue)),
        ]
        # await queue workers
        await asyncio.gather(*tasks)