import click
from birdhub.recorder import ContinuousRecorder
from birdhub.orchestration import VideoEventManager
from birdhub.video import Stream
from birdhub.logging import logger

@click.group()
def cli():
    pass

@click.command()
def test():
    """Test command"""
    logger.log_event("message","Hello world!")


@click.group()
def record():
    pass

@click.command()
@click.argument('url')
@click.argument('outputdir')
@click.option('--fps', type=int, default=10)
def continuous(url, outputdir, fps):
    """Record from video stream and save to file"""
    stream = Stream(url)
    recorder = ContinuousRecorder(outputDir=outputdir, frame_size=stream.frameSize, fps=fps)
    VideoEventManager(stream=stream, recorder=recorder)
    stream.stream()

# @click.command()
# @click.argument('url')
# @click.argument('outputdir')
# @click.option('--fps', type=int, default=10)
# @click.option('--slack', type=int, default=100)
# def motion(url, outputdir, fps, slack):
#     """Record from video stream and save to file"""
#     # TODO: make threshold dependent on image size
#     detector = SimpleMotionDetector(threshold_area=5_000)
#     recorder = MotionRecoder(url, detector, slack=slack,activation_frames=10)
#     recorder.record(outputdir, fps)


#record.add_command(motion)
record.add_command(continuous)
cli.add_command(record)
cli.add_command(test)

if __name__ == '__main__':
    cli()