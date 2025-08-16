#include "video/video.hpp"
#include "orchestration/orchestration.hpp"
#include "detection/detection.hpp"
#include "memory_management/mm.hpp"
#include "recording/recorder.hpp"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    // Setup signal handlers for graceful shutdown
    VideoEventManager::setup_signal_handlers();

    // Create the camera capture object with the desired settings
#ifdef __linux__
    V4l2CameraCapture camera("/dev/video0", 640, 480, V4L2_PIX_FMT_MJPEG);
#else
    OpenCVCameraCapture camera(0); // Use default camera
#endif
    // instantiate image store
    auto image_store = std::make_shared<ImageStore>(5000);
    auto stream = Stream(image_store, &camera);
    // instantiate motion detector
    auto motion_detector = std::make_shared<MotionDetector>(
        image_store,
        20,
        21,
        5,
        1000,
        1,
        std::chrono::seconds(5)
    );
    // instantiate event recorder
    auto recorder = std::make_shared<EventRecorder>(
        std::set<EventType>({EventType::NEW_FRAME, EventType::DETECTION}),
        image_store,
        "recordings",
        150,
        30, 
        150
    );
/*     auto recorder = std::make_shared<ContinuousRecorder>(
        std::set<EventType>({EventType::NEW_FRAME}),
        image_store,
        "recordings"
    ); */
    // instantiate event manager
    auto manager = VideoEventManager(stream);
    manager.add_subscriber(motion_detector);
    manager.add_subscriber(recorder);
    // start the event manager
    manager.run();
    std::cout << "VideoEventManager stopped gracefully." << std::endl;
    return 0;
}
