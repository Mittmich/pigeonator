#include "test_utils.hpp"

cv::Mat create_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_64FC1);
    double low = -500.0;
    double high = +500.0;
    cv::randu(img, cv::Scalar(low), cv::Scalar(high));
    return img;
}

// ConstantMockCameraCapture class
ConstantMockCameraCapture::ConstantMockCameraCapture(const char* device, int width, int height, uint32_t pixel_format) {}
cv::Mat ConstantMockCameraCapture::getNextFrame() {
    return create_random_image(3, 3);
};
void ConstantMockCameraCapture::startStreaming() {};
void ConstantMockCameraCapture::stopStreaming() {};


// MockStream class
MockStream::MockStream(std::shared_ptr<ImageStore> image_store, CameraCapture* cam_capture) 
        : Stream(image_store, cam_capture) {};
void MockStream::register_frame_queue(std::shared_ptr<std::queue<FrameEvent>> queue) {
        frame_queue = queue;
        queue_registered = true;
    };
void MockStream::start() {
        running = true;
    };
void MockStream::stop() {
        running = false;
    };

void MockStream::simulate_frame(const FrameEvent& frame) {
        if (queue_registered && frame_queue) {
            frame_queue->push(frame);
        }
    };

// MockSubscriber class

void MockSubscriber::set_event_queue(std::shared_ptr<std::queue<Event>> queue) {
        event_queue = queue;
    };

void MockSubscriber::start() {
        is_running = true;
    };

void MockSubscriber::stop() {
        is_running = false;
    };

void MockSubscriber::notify(Event event) {
        received_events.push_back(event);
    };

std::set<EventType> MockSubscriber::listening_to() {
        return event_types;
    };

// MockCameraCapture class

MockCameraCapture::MockCameraCapture(const char* device, int width, int height, uint32_t pixel_format) {};

cv::Mat MockCameraCapture::getNextFrame() {
        if (frames.empty()) {
            // Return random image if no frames are set
            return create_random_image(3, 3);
        }
        if (frame_index >= frames.size()) {
            return cv::Mat();
        }
        cv::Mat frame = frames.at(frame_index);
        frame_index++;
        return frame;
    };

void MockCameraCapture::startStreaming() {};

void MockCameraCapture::stopStreaming() {};

void MockCameraCapture::setMockFrames(std::vector<cv::Mat> frames) {
        this->frames = frames;
        this->frame_index = 0;
    };

time_t get_random_time() {
    return rand();
}