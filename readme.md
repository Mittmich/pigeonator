# Pigeonator

Pigeonator is a Python based software project designed for deterring pigeons using friendly means such as playing a sound recording. To do so, pigeonator employs Yolov5 based object detection using a custom model that is specifically trained for detecting pigeons.

In the following video, you can see pigeonator in action, detecting a pigeon and playing a recorded window knocking sound that causes the pigeon to fly away:

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

- To deter pigeons, use the following command:

    ```bash
    Usage: pgn deter [OPTIONS] URL OUTPUTDIR

    Options:
    --target_class TEXT
    --fps INTEGER
    --slack INTEGER
    --model TEXT
    --effector TEXT
    --record BOOLEAN
    --sound_path TEXT
    --minimum_number_detections INTEGER
    ```
    Here, `URL` refers to the url of your IP camera and the `OUTPUTDIR` refers to the directory, where recordings should be stored. `target_class` refers to the target class in your model that should trigger an effect, `fps` refers to the frames per second that your stream is supplying frames in, `slack` refers to the number of frames that the system will keep recording after an event has been registered. `model`, refers to the path of the Yolov5 model in onnx format, `effector` refers to the effector to be used, where the only option currently is the `sound` effector. `record` refers to whether detection events should be recorded, `sound_path` refers to the location of the audio file for the sound effector and `minimum_number_detections` refers to the treshold number of detections that are needed trigger an effect.

    There are defaults provided for all of the parameters, except for your IP camera URL and the outputdirectory. This repository contains two default sounds (a crow sound and a sound of someone knocking on glass) as well as a model that can detect pigeons.
    So to get going, you only need to run:
    
    ```bash
    pgn deter --effector sound YOUR_URL "./"
    ```

## Detection logic

To save on compute resources, pigeonator currently employs a two-stage detection scheme. First, a simple motion detector is run on provided images. If the motion detector detects a motion, the images are passed to the Yolov5 bird detector that emits a detection if more then `minimum_number_detections` objects have been found on the passed images. The emitted detection then triggers video recording as well as the specified effector.

## Model

Pigeonator comes with a trained Yolov5 model at `weights/bh_v1.onnx` that has been trained to recognise the classes:
- Pigeon
- Crow
- Blue Tit

### Data set for model training
The data used for training the object detection model can be found on roboflow: https://universe.roboflow.com/michael-mitter-6ffsp/pigeonator
