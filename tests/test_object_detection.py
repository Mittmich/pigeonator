import pytest
from unittest.mock import patch, MagicMock
from birdhub.detection import SingleClassSequenceDetector, Detection
import numpy as np

@pytest.fixture
def small_detection(empty_array):
    return [Detection(empty_array, ["cat", "dog"], [0.2, 0.3])]

@pytest.fixture
def big_detection(empty_array):
    return [Detection(empty_array, ["cat", "dog", "bird", "elephant", "tiger"], [0.1, 0.2, 0.3, 0.4, 0.5])]

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

def test_single_class_sequence_detector_obeys_threshold(small_sequence_detector, mock_event_manager, empty_array):
    seq = SingleClassSequenceDetector(minimum_number_detections=5, detector=small_sequence_detector)
    seq.add_event_manager(mock_event_manager)
    assert seq.detect(empty_array) == None
    mock_event_manager.notify.assert_not_called()

def test_single_class_sequence_detector_has_reached_consensus(long_sequence_detector, mock_event_manager, empty_array):
    seq = SingleClassSequenceDetector(minimum_number_detections=5, detector=long_sequence_detector)
    seq.add_event_manager(mock_event_manager)
    result = seq.detect(empty_array)
    assert result[0].get("meta_information") == {"most_likely_object": "tiger"}
    mock_event_manager.notify.assert_called()