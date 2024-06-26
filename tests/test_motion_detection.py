import numpy as np
import datetime
import pytest
from unittest.mock import MagicMock, patch

from birdhub.detection import SimpleMotionDetector
from birdhub.video import Frame, ImageStore


@pytest.fixture
def mock_pipe_connection():
    return MagicMock(), MagicMock()

@pytest.fixture
def image_store():
    return ImageStore(number_images=100)

@pytest.fixture
def motion_detector(mock_pipe_connection, image_store):
    connection, child_connection = mock_pipe_connection
    with patch("birdhub.detection.Pipe", return_value=(connection, child_connection)):
        detector = SimpleMotionDetector(activation_frames=3, max_delay=1_000)
        detector.add_event_manager(MagicMock())
        detector._image_store = image_store
        yield detector, connection

@pytest.fixture
def dummy_frame():
    # Create a 100x100 black image and a timestamp
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    timestamp = datetime.datetime.now()
    return Frame(timestamp), image


@pytest.fixture
def diff_frame():
    # Create a 100x100 white image and a timestamp
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    timestamp = datetime.datetime.now() + datetime.timedelta(seconds=10)
    return Frame(timestamp), image


def test_no_previous_image(motion_detector, dummy_frame, image_store):
    # No previous image, should not detect motion
    motion_detector, connection = motion_detector
    dummy_frame, image = dummy_frame
    image_store.put(dummy_frame.timestamp, image)
    assert motion_detector.detect(dummy_frame) is None
    connection.send.assert_not_called()


def test_no_motion(motion_detector, dummy_frame, image_store):
    # Call detect twice with the same frame, should not detect motion
    motion_detector, connection = motion_detector
    dummy_frame, image = dummy_frame
    image_store.put(dummy_frame.timestamp, image)
    motion_detector.detect(dummy_frame)
    assert motion_detector.detect(dummy_frame) is None
    connection.send.assert_not_called()


def test_no_activation(motion_detector, dummy_frame, diff_frame, image_store):
    motion_detector, connection = motion_detector
    dummy_frame, dummy_image = dummy_frame
    diff_frame, diff_image = diff_frame
    image_store.put(dummy_frame.timestamp, dummy_image)
    image_store.put(diff_frame.timestamp, diff_image)
    # Set activation frames to a large number
    motion_detector._activation_frames = 100
    # Call detect with different frames, should not detect motion
    motion_detector.detect(dummy_frame)
    assert motion_detector.detect(diff_frame) is None
    connection.send.assert_not_called()


def test_detects_motion(motion_detector, dummy_frame, diff_frame, image_store):
    motion_detector, connection = motion_detector
    dummy_frame, dummy_image = dummy_frame
    diff_frame, diff_image = diff_frame
    image_store.put(dummy_frame.timestamp, dummy_image)
    image_store.put(diff_frame.timestamp, diff_image)
    # Call detect with different frames until reaches the activation frames
    motion_sequence = [dummy_frame, diff_frame, dummy_frame]
    for image in motion_sequence:
        motion_detector.detect(image)
    # Now it should detect motion
    detections = motion_detector.detect(diff_frame)
    assert detections is not None
    assert len(detections) == 3
    connection.send.assert_called_once_with(("detection", detections))
    for detection in detections:
        assert detection.labels[0] == "motion"
