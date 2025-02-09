#include <video.cpp>
#include "orchestration.hpp"
#include "detection.hpp"
#include "mm.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    // Create the camera capture object with the desired settings
    V4l2CameraCapture camera("/dev/video0", 640, 480, V4L2_PIX_FMT_MJPEG);
    // instantiate image store
    auto image_store = std::make_shared<ImageStore>(1000);
    auto stream = Stream(image_store, &camera);
    // instantiate motion detector
    auto motion_detector = std::make_shared<MotionDetector>(
        image_store,
        20,
        21,
        5,
        200,
        0,
        std::chrono::seconds(5)
    );
    // instantiate event manager
    auto manager = VideoEventManager(stream);
    manager.add_subscriber(motion_detector);
    // start the event manager
    manager.run();
    return 0;
}
