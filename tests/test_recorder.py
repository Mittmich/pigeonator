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
def example_detections(empty_array):
    return [Detection(source_image=empty_array, labels=['bird'], confidences=[0.9], bboxes=[(0, 0, 100, 100)])]

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


def test_ContinuousRecorder_records(empty_array, mock_writer):
    cr = ContinuousRecorder("output_dir", frame_size=(640, 480 ), fps=10, writer_factory=mock_writer[0])
    cr.register_frame(empty_array)
    mock_writer[1].write.assert_called_with(empty_array)

def test_Event_Recorder_does_not_record_without_detection(empty_array, mock_writer, mock_detection_writer, event_recorder):
    event_recorder.register_frame(empty_array)
    mock_writer[1].write.assert_not_called()
    mock_detection_writer[1].write.assert_not_called()

def test_Event_Recorder_records_after_detection(empty_array, example_detections, mock_writer, mock_detection_writer, event_recorder):
        event_recorder.register_detection(example_detections)
        event_recorder.register_frame(empty_array)
        assert mock_writer[1].write.call_count == 1
        assert mock_detection_writer[1].write.call_count == 1
        assert len(event_recorder.create_detection_frames(example_detections)) == 1
        mock_writer[1].write.assert_any_call(empty_array)
        mock_detection_writer[1].write.assert_any_call(event_recorder.create_detection_frames(example_detections)[0])

def test_Event_Recorder_records_look_back_frames_correctly(empty_array, random_array, example_detections, mock_detection_writer, mock_writer, event_recorder):
    frames = [empty_array] + [random_array]
    # register first 5 frames
    for frame in frames[:5]:
        event_recorder.register_frame(frame)
    # register first detection
    event_recorder.register_detection(example_detections)
    assert mock_writer[1].write.call_count == 2
    # check that the first 5 frames are written
    for actual, expected in zip(mock_writer[1].write.mock_calls,frames):
        np.testing.assert_array_equal(actual.args[0], expected)