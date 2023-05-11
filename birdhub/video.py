"""A module to handle video streams and video files."""
import cv2

class Stream:

    def __init__(self, streamurl):
        self.streamurl = streamurl
        self.cap = cv2.VideoCapture(self.streamurl)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
        cv2.destroyAllWindows()

    def __next__(self):
        return self.get_frame()
    
    @property
    def frameSize(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self):
        return self
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


class VideoWriter:
    """A class to write video frames to a file."""

    def __init__(self, filename, fps, frameSize):
        """Initialize the VideoWriter object."""
        self.filename = filename
        self.fps = fps
        self.frameSize = frameSize
        self.videoWriter = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'), fps, frameSize)

    def write(self, frame):
        """Write a frame to the video file."""
        self.videoWriter.write(frame)

    def __enter__(self):
        """Return the VideoWriter object."""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Release the VideoWriter object."""
        self.videoWriter.release()

    def __del__(self):
        """Destroy the VideoWriter object."""
        self.videoWriter.release()