import pytest
from unittest.mock import MagicMock
from birdhub.detection import MotionActivatedSingleClassDetector


@pytest.fixture
def detectors():
    """Create mock detector and motion detector."""
    detector = MagicMock()
    motion_detector = MagicMock()
    return detector, motion_detector


@pytest.fixture
def frame():
    """Create a mock frame."""
    return MagicMock()


def test_no_motion_no_detection(detectors, frame):
    """Test that the detector is not called if no motion is detected."""
    detector, motion_detector = detectors
    motion_detector.detect.return_value = None
    detector.detect.return_value = None

    motion_activated_detector = MotionActivatedSingleClassDetector(
        detector, motion_detector
    )
    motion_activated_detector.detect(frame)

    detector.detect.assert_not_called()


def test_motion_detected(detectors, frame):
    """Test that the detector is called if motion is detected."""
    detector, motion_detector = detectors
    motion_detections = [MagicMock()]
    motion_detector.detect.return_value = motion_detections
    detector.detect.return_value = motion_detections

    motion_activated_detector = MotionActivatedSingleClassDetector(
        detector, motion_detector
    )
    motion_activated_detector.add_event_manager(MagicMock())
    motion_activated_detector.detect(frame)

    detector.detect.assert_called_once_with(frame)


def test_slack_time_no_motion(detectors, frame):
    """Test that the detector is called if slack is larger than 0 and no motion is detected."""
    detector, motion_detector = detectors
    motion_detections = [MagicMock()]
    motion_detector.detect.side_effect = [motion_detections, None]
    detector.detect.return_value = motion_detections

    motion_activated_detector = MotionActivatedSingleClassDetector(
        detector, motion_detector, slack=10
    )
    motion_activated_detector.add_event_manager(MagicMock())
    motion_activated_detector.detect(frame)
    detector.detect.assert_called_once_with(frame)
    detector.reset_mock()

    motion_activated_detector.detect(frame)
    detector.detect.assert_called_once_with(frame)


def test_reset_after_slack_expired(detectors, frame):
    """test that detector is reset after slack time has expired"""
    detector, motion_detector = detectors
    motion_detections = [MagicMock()]
    motion_detector.detect.side_effect = [motion_detections, None]
    detector.detect.return_value = motion_detections

    motion_activated_detector = MotionActivatedSingleClassDetector(
        detector, motion_detector, slack=0
    )
    motion_activated_detector.add_event_manager(MagicMock())
    motion_activated_detector.detect(frame)
    detector.detect.assert_called_once_with(frame)
    detector.reset_mock()

    motion_activated_detector.detect(frame)
    detector.detect.assert_not_called()
    assert len(motion_activated_detector._detections) == 0
