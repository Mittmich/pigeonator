import pytest
from unittest.mock import MagicMock
from datetime import timedelta
from birdhub.detection import Detection
from birdhub.effectors import MockEffector


@pytest.fixture
def mock_detection_correct_class():
    detection = MagicMock()
    detection.get.return_value = {"most_likely_object":"correct_class"}
    return detection


@pytest.fixture
def mock_detection_wrong_class():
    detection = MagicMock()
    detection.get.return_value = {"most_likely_object":"wrong_class"}
    return detection


def test_no_detection():
    event_manager = MagicMock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection(None)

    event_manager.log.assert_not_called()


def test_wrong_class_detection(mock_detection_wrong_class):
    event_manager = MagicMock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection([mock_detection_wrong_class])

    event_manager.log.assert_not_called()


def test_correct_class_detection(mock_detection_correct_class):
    event_manager = MagicMock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection([mock_detection_correct_class])

    event_manager.notify.assert_called_once()


def test_activation_time_too_short(mock_detection_correct_class):
    event_manager = MagicMock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    # Assuming that `_last_activation` is set at the moment of activation
    # First detection should trigger activation
    effector.register_detection([mock_detection_correct_class])
    # Second detection within the cooldown time shouldn't trigger activation
    effector.register_detection([mock_detection_correct_class])

    assert event_manager.notify.call_count == 1
