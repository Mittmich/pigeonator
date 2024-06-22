import click
from birdhub.recorder import ContinuousRecorder, EventRecorder
import asyncio
from birdhub.orchestration import VideoEventManager
from birdhub.detection import (
    SimpleMotionDetector,
    BirdDetectorYolov5,
    MotionActivatedSingleClassDetector,
)
from birdhub.video import RTSPStream, RaspberryPiStream
from birdhub.logging import logger
from birdhub.effectors import EFFECTORS
from datetime import timedelta


@click.group()
def cli():
    pass


@click.command()
def test():
    """Test command"""
    logger.log_event("message", "Hello world!")


@click.group()
def record():
    pass


@click.command()
@click.argument("url")
@click.argument("outputdir")
@click.option("--fps", type=int, default=10)
def continuous(url, outputdir, fps):
    """Record from video stream and save to file"""
    stream = RTSPStream(url)
    recorder = ContinuousRecorder(
        outputDir=outputdir, frame_size=stream.frameSize, fps=fps
    )
    VideoEventManager(stream=stream, recorder=recorder)
    stream.stream()


@click.command()
@click.argument("url")
@click.argument("outputdir")
@click.option("--fps", type=int, default=10)
@click.option("--slack", type=int, default=100)
def motion(url, outputdir, fps, slack):
    """Record from video stream and save to file"""
    # TODO: make threshold dependent on image size
    stream = RTSPStream(url)
    recorder = EventRecorder(
        outputDir=outputdir, frame_size=stream.frameSize, fps=fps, slack=slack
    )
    detector = SimpleMotionDetector(threshold_area=5_000)
    VideoEventManager(stream=stream, recorder=recorder, detector=detector)
    stream.stream()


@click.command()
@click.argument("url")
@click.argument("outputdir")
@click.option("--fps", type=int, default=10)
@click.option("--slack", type=int, default=100)
@click.option("--model", type=str, default="weights/bh_v1.onnx")
def birds(url, outputdir, fps, slack, model):
    stream = RTSPStream(url)
    recorder = EventRecorder(
        outputDir=outputdir, frame_size=stream.frameSize, fps=fps, slack=slack
    )
    motion_detector = SimpleMotionDetector(threshold_area=5_000)
    bird_detector = BirdDetectorYolov5(model, confidence_threshold=0.6)
    motion_activated_detector = MotionActivatedSingleClassDetector(
        bird_detector, motion_detector, minimum_number_detections=20
    )
    VideoEventManager(
        stream=stream,
        recorder=recorder,
        detector=motion_activated_detector,
        throttle_detection=1,
    )
    stream.stream()


@click.command()
@click.argument("url")
@click.argument("outputdir")
@click.option("--target_class", type=str, default="Pigeon")
@click.option("--fps", type=int, default=10)
@click.option("--slack", type=int, default=100)
@click.option("--model", type=str, default="weights/bh_v2.onnx")
@click.option("--effector", type=str, default="mock")
@click.option("--record", type=bool, default=True)
@click.option("--sound_path", type=str, default="sounds/crow_1.mp3")
@click.option("--minimum_number_detections", type=int, default=5)
@click.option("--stream_type", type=str, default="rtsp")
@click.option("--motion_th_area", type=int, default=2_000)
def deter(
    url,
    outputdir,
    target_class,
    fps,
    slack,
    model,
    effector,
    record,
    minimum_number_detections,
    sound_path,
    stream_type,
    motion_th_area
):
    if stream_type == 'rtsp':
        stream = RTSPStream(url)
    elif stream_type == 'raspi':
        stream = RaspberryPiStream(fps=fps)
    if record:
        recorder = EventRecorder(
            outputDir=outputdir,
            frame_size=stream.frameSize,
            fps=fps,
            slack=slack,
            look_back_frames=20,
        )
    else:
        recorder = None
    motion_detector = SimpleMotionDetector(threshold_area=motion_th_area, activation_frames=1, max_delay=2)
    bird_detector = BirdDetectorYolov5(model, confidence_threshold=0.6, max_delay=2)
    motion_activated_detector = MotionActivatedSingleClassDetector(
        bird_detector,
        motion_detector,
        minimum_number_detections=minimum_number_detections,
        slack=slack,
    )
    # instantiate effector
    effector = EFFECTORS[effector](
        target_class=target_class,
        cooldown_time=timedelta(seconds=10),
        config={"sound_file": sound_path},
    )
    event_manager = VideoEventManager(
        stream=stream,
        recorder=recorder,
        detector=motion_activated_detector,
        effector=effector,
    )
    asyncio.run(event_manager.run())


record.add_command(motion)
record.add_command(continuous)
record.add_command(birds)
cli.add_command(record)
cli.add_command(test)
cli.add_command(deter)

if __name__ == "__main__":
    cli()
