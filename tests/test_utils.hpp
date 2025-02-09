#pragma once
#include "video.hpp"
#include <opencv2/opencv.hpp>

cv::Mat create_random_image(int rows, int cols);

class ConstantMockCameraCapture : public CameraCapture {
public:
    ConstantMockCameraCapture(const char* device, int width, int height, uint32_t pixel_format);
    cv::Mat getNextFrame() override;
    void startStreaming() override;
    void stopStreaming() override;
};

class MockStream : public Stream {
public:
    MockStream(std::shared_ptr<ImageStore> image_store, CameraCapture* cam_capture);

    void register_frame_queue(std::shared_ptr<std::queue<FrameEvent>> queue) override;

    void start() override;

    void stop() override;

    // Test helper methods
    void simulate_frame(const FrameEvent& frame);

    bool is_running();
    bool has_queue();
};


class MockSubscriber : public Subscriber {
public:
    std::set<EventType> event_types;
    std::vector<Event> received_events;
    std::vector<FrameEvent> received_frames;
    bool is_running = false;
    void set_event_queue(std::shared_ptr<std::queue<Event>> queue) override;
    void start() override;
    void stop() override;
    void notify(Event event) override;
    std::set<EventType> listening_to() override;
    std::shared_ptr<std::queue<Event>> event_queue;
};




class MockCameraCapture : public CameraCapture {
public:
    MockCameraCapture(const char* device, int width, int height, uint32_t pixel_format);
    cv::Mat getNextFrame() override;
    void startStreaming() override;
    void stopStreaming() override;
    void setMockFrames(std::vector<cv::Mat> frames);
private:
    std::vector<cv::Mat> frames;
    int frame_index;
};

time_t get_random_time();