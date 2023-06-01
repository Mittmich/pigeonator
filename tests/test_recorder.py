import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from birdhub.recorder import ContinuousRecorder, EventRecorder
from birdhub.detection import Detection

@pytest.fixture
def empty_array():
    return np.zeros((640, 640, 3), dtype=np.uint8)

@pytest.fixture
def random_array():
    np.random.seed(42)
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def example_detection(empty_array):
    return Detection(['bird'], [0.9], [[(0, 0, 100, 100)]], [empty_array])

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

@pytest.fixture
def multiple_motion_events_stream(empty_array, random_array):
    # Creating a mock stream with intermittent motion frames
    stream = MagicMock()
    frameSequence = MagicMock()
    # stream starts wih 3 motions, then stills, then one motion frame
    frameSequence.__iter__.return_value = [empty_array] + [random_array] + [empty_array] + [random_array] + 10 * [random_array] + 10*[empty_array]
    stream.__enter__.return_value = frameSequence
    return stream


def test_ContinuousRecorder_records(empty_array):
    with  patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        cr = ContinuousRecorder("output_dir", frame_size=(640, 480 ), fps=10)
        cr.register_frame(empty_array)
        writer.write.assert_called_with(empty_array)

def test_Event_Recorder_does_not_record_without_detection(empty_array):
    with patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        er = EventRecorder("output_dir", frame_size=(640, 480 ), fps=10)
        er.register_frame(empty_array)
        writer.write.assert_not_called()

def test_Event_Recorder_records_after_detection(empty_array, example_detection):
    with patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        er = EventRecorder("output_dir", frame_size=(640, 480 ), fps=10)
        mock_event_manager = MagicMock()
        er.add_event_manager(mock_event_manager)
        er.register_detection(example_detection)
        er.register_frame(empty_array)
        assert writer.write.call_count == 2
        assert len(er.create_detection_frames(example_detection)) == 1
        writer.write.assert_any_call(empty_array)
        writer.write.assert_any_call(er.create_detection_frames(example_detection)[0])

def test_Event_Recorder_records_look_back_frames_correctly(empty_array, random_array, example_detection):
    with patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
        writer = MagicMock()
        MockVideoWriter.return_value = writer
        mock_event_manager = MagicMock()
        er = EventRecorder("output_dir", frame_size=(640, 480 ), fps=10, look_back_frames=5)
        er.add_event_manager(mock_event_manager)
        frames = [empty_array] + 5 * [random_array]
        # register first 5 frames
        for frame in frames[:5]:
            er.register_frame(frame)
        # register first detection
        er.register_detection(example_detection)
        assert writer.write.call_count == 6
        # check that the first 5 frames are written
        for actual, expected in zip(writer.write.mock_calls,frames[:5]):
            np.testing.assert_array_equal(actual.args[0], expected)


# def test_MotionRecoder_does_not_record_if_no_motion(no_motion_stream):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = no_motion_stream
#         writer = MagicMock()
#         MockVideoWriter.return_value = writer
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=10)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         writer.write.assert_not_called()

# def test_MotionRecoder_records_motion(single_frame_motion_stream):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = single_frame_motion_stream
#         writer = MagicMock()
#         MockVideoWriter.return_value = writer
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=0)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         writer.write.assert_called()
#         assert writer.write.call_count == 3


# def test_MotionRecoder_does_not_record_below_activation_frames(single_frame_motion_stream):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = single_frame_motion_stream
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=100, activation_frames=1)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         MockVideoWriter.assert_not_called()

# def test_MotionRecorder_does_not_record_below_activation_frames_multiple_sequences(multiple_motion_events_stream):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = multiple_motion_events_stream
#         writer = MagicMock()
#         MockVideoWriter.return_value = writer
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=5, activation_frames=3)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         writer.write.assert_not_called()

# def test_MotionRecorder_respects_activation_frames_multiple_sequences(multiple_motion_events_stream):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = multiple_motion_events_stream
#         writer = MagicMock()
#         MockVideoWriter.return_value = writer
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=2, activation_frames=2)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         writer.write.assert_called()
#         assert writer.write.call_count == 5


# def test_MotionRecoder_respects_slack(long_single_frame_motion_stream, empty_array):
#     with patch('birdhub.recorder.Stream') as MockStream, \
#          patch('birdhub.recorder.VideoWriter') as MockVideoWriter:
#         MockStream.return_value = long_single_frame_motion_stream
#         writer = MagicMock()
#         MockVideoWriter.return_value = writer
#         mr = MotionRecoder('http://example.com', SimpleMotionDetector(), slack=10, activation_frames=0)
#         mr.record('/path/to/output/dir', fps=10)
#         MockStream.assert_called_once_with('http://example.com')
#         writer.write.assert_called()
#         assert writer.write.call_count == 12
#         # ensure lockbackframes have been written
#         writer.write.assert_any_call(empty_array)