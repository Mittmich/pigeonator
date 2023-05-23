import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from birdhub.recorder import ContinuousRecorder, MotionRecoder
from birdhub.motion_detection import SimpleMotionDetector

@pytest.fixture
def empty_array():
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def random_array():
    np.random.seed(42)
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def no_motion_stream(empty_array):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_array, empty_array]
    frameSequence.frameSize.return_value = (640, 640)
    stream.__enter__.return_value = frameSequence
    return stream

@pytest.fixture
def single_frame_motion_stream(empty_array, random_array):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_array, random_array]
    stream.__enter__.return_value = frameSequence
    return stream


@pytest.fixture
def long_single_frame_motion_stream(empty_array, random_array):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_array] + [random_array] * 100
    stream.__enter__.return_value = frameSequence
    return stream


def test_ContinuousRecorder_record():
    with patch('birdhub.recorder.Stream') as MockStream, \
         patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        cr = ContinuousRecorder('http://example.com')
        cr.record('/path/to/output/dir', fps=10)
        MockStream.assert_called_once_with('http://example.com')
        MockVideoWriter.assert_called_once()

def test_MotionRecoder_does_not_record_if_no_motion(no_motion_stream):
    with patch('birdhub.recorder.Stream') as MockStream, \
         patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        MockStream.return_value = no_motion_stream
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=10)
        mr.record('/path/to/output/dir', fps=10)
        MockStream.assert_called_once_with('http://example.com')
        writer.write.assert_not_called()

def test_MotionRecoder_records_motion(single_frame_motion_stream):
    with patch('birdhub.recorder.Stream') as MockStream, \
         patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        MockStream.return_value = single_frame_motion_stream
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=0)
        mr.record('/path/to/output/dir', fps=10)
        MockStream.assert_called_once_with('http://example.com')
        writer.write.assert_called()
        assert writer.write.call_count == 3


def test_MotionRecoder_does_not_record_below_activation_frames(single_frame_motion_stream):
    with patch('birdhub.recorder.Stream') as MockStream, \
         patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        MockStream.return_value = single_frame_motion_stream
        mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=1)
        mr.record('/path/to/output/dir', fps=10)
        MockStream.assert_called_once_with('http://example.com')
        MockVideoWriter.assert_not_called()

def test_MotionRecoder_respects_slack(long_single_frame_motion_stream, empty_array):
    with patch('birdhub.recorder.Stream') as MockStream, \
         patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        MockStream.return_value = long_single_frame_motion_stream
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=10, activation_frames=0)
        mr.record('/path/to/output/dir', fps=10)
        MockStream.assert_called_once_with('http://example.com')
        writer.write.assert_called()
        assert writer.write.call_count == 12
        # ensure lockbackframes have been written
        writer.write.assert_any_call(empty_array)