import click
from video import Stream, VideoWriter


@click.group()
def cli():
    pass


@click.command()
@click.argument('url')
@click.argument('outputfile')
@click.option('--fps', type=int, default=10)
def record(url, outputfile, fps):
    """Record from video stream and save to file"""
    with Stream(url) as stream:
        print(stream.frameSize)
        with VideoWriter(outputfile, fps, stream.frameSize) as writer:
            for frame in stream:
                writer.write(frame)


cli.add_command(record)

if __name__ == '__main__':
    cli()