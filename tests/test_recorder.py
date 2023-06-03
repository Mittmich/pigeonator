import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import datetime
from birdhub.recorder import ContinuousRecorder, EventRecorder
from birdhub.detection import Detection, Frame

@pytest.fixture
def empty_frame():
    return Frame(timestamp=datetime.datetime(year=2023, month=5, day=8), image=np.zeros((640, 640, 3), dtype=np.uint8))

@pytest.fixture
def random_frame():
    np.random.seed(42)
    return Frame(timestamp=datetime.datetime(year=2023, month=5, day=8), image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))

@pytest.fixture
def example_detections(empty_frame):
    return [Detection(empty_frame.timestamp, labels=['bird'], confidences=[0.9], bboxes=[(0, 0, 100, 100)])]

@pytest.fixture
def mock_writer():
    writer = MagicMock()
    MockVideoWriter = MagicMock()
    MockVideoWriter.return_value = writer
    return MockVideoWriter, writer

@pytest.fixture
def mock_detection_writer():
    writer = MagicMock()
    MockVideoWriter = MagicMock()
    MockVideoWriter.return_value = writer
    return MockVideoWriter, writer

@pytest.fixture
def mock_event_manager():
    return MagicMock()

@pytest.fixture
def event_recorder(mock_writer, mock_detection_writer, mock_event_manager):
    er = EventRecorder("output_dir", frame_size=(640, 480 ), fps=10, writer_factory=mock_writer[0], detection_writer_factory=mock_detection_writer[0])
    er.add_event_manager(mock_event_manager)
    return er

@pytest.fixture
def no_motion_stream(empty_frame):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_frame, empty_frame]
    frameSequence.frameSize.return_value = (640, 640)
    stream.__enter__.return_value = frameSequence
    return stream

@pytest.fixture
def single_frame_motion_stream(empty_frame, random_frame):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_frame, random_frame]
    stream.__enter__.return_value = frameSequence
    return stream


@pytest.fixture
def long_single_frame_motion_stream(empty_frame, random_frame):
    stream = MagicMock()
    frameSequence = MagicMock()
    frameSequence.__iter__.return_value = [empty_frame] + [random_frame] * 100
    stream.__enter__.return_value = frameSequence
    return stream

@pytest.fixture
def multiple_motion_events_stream(empty_frame, random_frame):
    # Creating a mock stream with intermittent motion frames
    stream = MagicMock()
    frameSequence = MagicMock()
    # stream starts wih 3 motions, then stills, then one motion frame
    frameSequence.__iter__.return_value = [empty_frame] + [random_frame] + [empty_frame] + [random_frame] + 10 * [random_frame] + 10*[empty_frame]
    stream.__enter__.return_value = frameSequence
    return stream


def test_ContinuousRecorder_records(empty_frame, mock_writer):
    cr = ContinuousRecorder("output_dir", frame_size=(640, 480 ), fps=10, writer_factory=mock_writer[0])
    cr.register_frame(empty_frame)
    mock_writer[1].write.assert_called_with(empty_frame.image)

def test_Event_Recorder_does_not_record_without_detection(empty_frame, mock_writer, mock_detection_writer, event_recorder):
    event_recorder.register_frame(empty_frame)
    mock_writer[1].write.assert_not_called()
    mock_detection_writer[1].write.assert_not_called()

def test_Event_Recorder_records_after_detection(empty_frame, example_detections, mock_writer, event_recorder):
        event_recorder.register_detection(example_detections)
        event_recorder.register_frame(empty_frame)
        assert mock_writer[1].write.call_count == 1
        mock_writer[1].write.assert_any_call(empty_frame.image)

def test_Event_Recorder_records_look_back_frames_correctly(empty_frame, random_frame, example_detections, mock_writer, event_recorder):
    frames = [empty_frame] + [random_frame]
    # register first 5 frames
    for frame in frames[:5]:
        event_recorder.register_frame(frame)
    # register first detection
    event_recorder.register_detection(example_detections)
    assert mock_writer[1].write.call_count == 2
    # check that the first 5 frames are written
    for actual, expected in zip(mock_writer[1].write.mock_calls,frames):
        np.testing.assert_array_equal(actual.args[0], expected.image)

# Write tests for detection writing