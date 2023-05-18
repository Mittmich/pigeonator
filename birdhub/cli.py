import click
from birdhub.recorder import ContinuousRecorder, MotionRecoder
from birdhub.motion_detection import SimpleMotionDetector


@click.group()
def cli():
    pass

@click.group()
def record():
    pass

@click.command()
@click.argument('url')
@click.argument('outputdir')
@click.option('--fps', type=int, default=10)
def continuous(url, outputdir, fps):
    """Record from video stream and save to file"""
    recorder = ContinuousRecorder(url)
    recorder.record(outputdir, fps)

@click.command()
@click.argument('url')
@click.argument('outputdir')
@click.option('--fps', type=int, default=10)
@click.option('--slack', type=int, default=100)
def motion(url, outputdir, fps, slack):
    """Record from video stream and save to file"""
    detector = SimpleMotionDetector(threshold_area=500)
    recorder = MotionRecoder(url, detector, slack=slack)
    recorder.record(outputdir, fps)


record.add_command(motion)
record.add_command(continuous)
cli.add_command(record)

if __name__ == '__main__':
    cli()