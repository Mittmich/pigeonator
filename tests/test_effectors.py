import pytest
from unittest.mock import Mock
from datetime import timedelta
from birdhub.detection import Detection
from birdhub.effectors import MockEffector


@pytest.fixture
def mock_detection_correct_class():
    detection = Mock(spec=Detection)
    detection.get.return_value = "correct_class"
    return detection


@pytest.fixture
def mock_detection_wrong_class():
    detection = Mock(spec=Detection)
    detection.get.return_value = "wrong_class"
    return detection


def test_no_detection():
    event_manager = Mock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection(None)

    event_manager.log.assert_not_called()


def test_wrong_class_detection(mock_detection_wrong_class):
    event_manager = Mock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection([mock_detection_wrong_class])

    event_manager.log.assert_not_called()


def test_correct_class_detection(mock_detection_correct_class):
    event_manager = Mock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    effector.register_detection([mock_detection_correct_class])

    event_manager.log.assert_called_with(
        "effect_activated", {"type": "mock", "target_class": "correct_class"}
    )


def test_activation_time_too_short(mock_detection_correct_class):
    event_manager = Mock()
    effector = MockEffector("correct_class", timedelta(minutes=10))
    effector.add_event_manager(event_manager)

    # Assuming that `_last_activation` is set at the moment of activation
    # First detection should trigger activation
    effector.register_detection([mock_detection_correct_class])
    # Second detection within the cooldown time shouldn't trigger activation
    effector.register_detection([mock_detection_correct_class])

    assert event_manager.log.call_count == 1
