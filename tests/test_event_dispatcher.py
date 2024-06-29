"""Tests for event dispatcher."""
import pytest
import asyncio
import datetime
from pathlib import Path
from unittest.mock import MagicMock
from birdhub.remote_events import EventDispatcher
from birdhub.orchestration import VideoEventManager
from birdhub.detection import Detection


@pytest.fixture
def mock_requests():
    """requests module with mock post method"""
    mock_requests = MagicMock()
    mock_post = MagicMock()
    mock_requests.post = mock_post
    return mock_requests


@pytest.mark.parametrize(
    "event_type, success, exp", [
        (["detection"], True, ["detection"]),
        (["effect_activated", 'detection'], True, ["effect_activated", 'detection']),
        (['bad_type'], False, None),
        (None, True, ['detection',"effect_activated", 'recording_stopped'])
    ]
)
def test_instantiation(event_type, success, exp):
    """Test that instantiation fails with invalid event type."""
    if not success:
        with pytest.raises(ValueError):
            EventDispatcher("http://localhost:5000", ["invalid_event_type"])
    else:
        disptacher = EventDispatcher("http://localhost:5000", event_type)
        assert sorted(disptacher._listening_for) == sorted(exp)



@pytest.mark.parametrize(
    "detections", [
        [Detection(
            frame_timestamp=datetime.datetime.now(),
            labels=["bird"],
            confidences=[0.9],
            bboxes=[(0, 0, 100, 100)],
            meta_information={'most_likely_object': 'pigeon', 'mean_confidence': 0.9}
        )],
        [Detection(
            frame_timestamp=datetime.datetime.now(),
            labels=["bird"],
            confidences=[0.9],
            bboxes=[(0, 0, 100, 100)],
            meta_information={'most_likely_object': 'crow', 'mean_confidence': 0.9}
        )]
    ]
)
@pytest.mark.asyncio
async def test_detections_from_event_manager_sent_correctly(detections, monkeypatch, mock_requests):
    """Test whether detections are sent correctly."""
    monkeypatch.setattr("birdhub.remote_events.requests", mock_requests)
    dispatcher = EventDispatcher("http://localhost:5000", ["detection"])
    event_manager = VideoEventManager(stream=MagicMock(),event_dispatcher=dispatcher)
    # start dispatcher thread
    dispatcher.run()
    # send detection via event manager
    await event_manager.notify("detection", detections)
    # sleep to wait for dispatcher to process the request
    await asyncio.sleep(1)
    # check if request was sent correctly
    mock_requests.post.assert_called_once()
    mock_requests.post.assert_called_with(
        "http://localhost:5000/detections/",
        json={
            "detections": [
                {
                    "detected_class": detections[0].get('meta_information')["most_likely_object"],
                    "detection_timestamp": detections[0].get("frame_timestamp").strftime("%Y-%m-%dT%H:%M:%S"),
                    "confidence": detections[0].get("meta_information")["mean_confidence"],
                }
            ],
        },
    )


@pytest.mark.parametrize(
    "effect_activation", [
        {
            "type": "sound",
            "meta_information": {"detection_timestamp": datetime.datetime.now()},
            "timestamp": datetime.datetime.now(),
        }
    ]
)
@pytest.mark.asyncio
async def test_effect_activated_from_event_manager_sent_correctly(effect_activation,monkeypatch, mock_requests):
    """Test whether effect activated events are sent correctly."""
    monkeypatch.setattr("birdhub.remote_events.requests", mock_requests)
    dispatcher = EventDispatcher("http://localhost:5000", ["effect_activated"])
    event_manager = VideoEventManager(stream=MagicMock(),event_dispatcher=dispatcher)
    # start dispatcher thread
    dispatcher.run()
    # send effect activated event via event manager
    await event_manager.notify("effect_activated",effect_activation)
    # sleep to wait for dispatcher to process the request
    await asyncio.sleep(1)
    # check if request was sent correctly
    mock_requests.post.assert_called_once()
    mock_requests.post.assert_called_with(
        "http://localhost:5000/effectorAction/",
        json={
            "action": effect_activation["type"],
            "action_metadata": effect_activation["meta_information"],
            "detection_timestamp": effect_activation["meta_information"]["detection_timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
            "action_timestamp": effect_activation["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

@pytest.mark.parametrize(
    "recording_data", [
        {
            "recording_file": r"tests\testfiles\test_video.mp4",
            "recording_timestamp": datetime.datetime.now(),
            "recording_end_timestamp": datetime.datetime.now(),
        }
    ]
)
@pytest.mark.asyncio
async def test_recording_stopped_from_event_manager_sent_correctly(recording_data, monkeypatch, mock_requests):
    """Test whether recording stopped events are sent correctly."""
    monkeypatch.setattr("birdhub.remote_events.requests", mock_requests)
    dispatcher = EventDispatcher("http://localhost:5000", ["recording_stopped"])
    event_manager = VideoEventManager(stream=MagicMock(),event_dispatcher=dispatcher)
    # start dispatcher thread
    dispatcher.run()
    # send recording stopped event via event manager
    await event_manager.notify("recording_stopped",recording_data)
    # sleep to wait for dispatcher to process the request
    await asyncio.sleep(1)
    # check if request was sent correctly
    mock_requests.post.assert_called_once()