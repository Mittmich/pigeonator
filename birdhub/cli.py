import click
from birdhub.recorder import ContinuousRecorder


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

record.add_command(continuous)
cli.add_command(record)

if __name__ == '__main__':
    cli()