"""Tests for event dispatcher."""
import pytest
import asyncio
import datetime
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
    "event_type, success", [
        (["detection"], True),
        (["effect_activated", 'detection'], True),
        (['bad_type'], False),
    ]
)
def test_instantiation(event_type, success):
    """Test that instantiation fails with invalid event type."""
    if not success:
        with pytest.raises(ValueError):
            EventDispatcher("http://localhost:5000", ["invalid_event_type"])
    else:
        EventDispatcher("http://localhost:5000", event_type)



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
                    "detection_timestamp": detections[0].get("frame_timestamp"),
                    "confidence": detections[0].get("mean_confidence"),
                }
            ],
        },
    )