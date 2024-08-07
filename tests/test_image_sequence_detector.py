import pytest
from unittest.mock import patch, MagicMock
from birdhub.detection import SingleClassSequenceDetector, Detection
import numpy as np


@pytest.fixture
def small_detection(empty_array):
    return [Detection(empty_array, ["cat", "dog"], [0.2, 0.3])]


@pytest.fixture
def big_detection(empty_array):
    return [
        Detection(
            empty_array,
            ["cat", "dog", "bird", "elephant", "tiger", 'tiger'],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
    ]


@pytest.fixture
def small_sequence_detector(small_detection):
    mock = MagicMock()
    mock.detect.return_value = small_detection
    return mock


@pytest.fixture
def long_sequence_detector(big_detection):
    mock = MagicMock()
    mock.detect.return_value = big_detection
    return mock


@pytest.fixture
def empty_array():
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_event_manager():
    return MagicMock()


@pytest.fixture
def mock_pipe_connection():
    return MagicMock(), MagicMock()


def test_single_class_sequence_detector_obeys_threshold(
    small_sequence_detector, mock_event_manager, empty_array
):
    seq = SingleClassSequenceDetector(
        minimum_number_detections=5, detector=small_sequence_detector
    )
    seq.add_event_manager(mock_event_manager)
    assert seq.detect(empty_array) == None
    mock_event_manager.notify.assert_not_called()


def test_single_class_sequence_has_reached_consensus(
    long_sequence_detector, mock_event_manager, mock_pipe_connection, empty_array
):
    connection, child_connection = mock_pipe_connection
    with patch("birdhub.detection.Pipe", return_value=(connection, child_connection)):
        seq = SingleClassSequenceDetector(
            minimum_number_detections=5, detector=long_sequence_detector
        )
        seq.add_event_manager(mock_event_manager)
        result = seq.detect(empty_array)
        assert result[0].get("meta_information")["most_likely_object"] == "tiger"
        assert np.isclose(result[0].get("meta_information")["mean_confidence"], 0.55)
        connection.send.assert_called()


def test_single_class_sequence_detector_resets_after_detection(
    small_sequence_detector, mock_event_manager, mock_pipe_connection, empty_array
):
    connection, child_connection = mock_pipe_connection
    with patch("birdhub.detection.Pipe", return_value=(connection, child_connection)):
        seq = SingleClassSequenceDetector(
            minimum_number_detections=3, detector=small_sequence_detector
        )
        seq.add_event_manager(mock_event_manager)
        seq.detect(empty_array)
        result = seq.detect(empty_array)
        assert result[0].get("meta_information")["most_likely_object"] == "dog"
        connection.send.assert_called()
        # check that further detections are not emitted
        connection.reset_mock()
        seq.detect(empty_array)
        connection.send.assert_not_called()
