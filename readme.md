# Pigeonator

Pigeonator is a Python based software project designed for recording and analyzing video streams, specifically focused on bird detection. The primary use case of Pigeonator is to record video streams either continuously or upon detecting motion and subsequently performing object detection on the captured frames.

The project uses various tools and libraries such as OpenCV for motion detection, PyTorch and the YOLOv5 model for object detection, and Click for providing a command-line interface. 

## Installation

Before installing Pigeonator, ensure that you have Python 3.6 or above installed on your system. 

You can install Pigeonator by cloning the repository and running the setup script:

1. Clone the repository:

    ```bash
    git clone https://github.com/mittmich/thepigeonator.git
    cd thepigeonator
    ```

2. Install the package:

    ```bash
    python setup.py install
    ```

    It is recommended to use a virtual environment for installation to avoid potential conflicts with other Python packages you may have installed.

## Usage

After installation, you can start using Pigeonator from the command line:

- To start recording continuously from a video stream:

    ```bash
    pgn record continuous <url> <outputdir> [--fps=<fps>]
    ```

- To start recording upon detecting motion in a video stream:

    ```bash
    pgn record motion <url> <outputdir> [--fps=<fps>] [--slack=<slack>]
    ```

In the commands above, replace `<url>` with the URL of the video stream, `<outputdir>` with the path to the directory where the recordings should be saved, `<fps>` with the desired frames per second (default is 10), and `<slack>` with the number of frames to continue recording after motion has stopped (default is 100).

For further details on each of the command options, use the `--help` option:

```bash
pgn --help
