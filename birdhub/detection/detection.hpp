#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <set>
#include <thread>
#include "mm.hpp"
#include <chrono>

#pragma once

// add detector base class, inheriting from Subscriber

class Detector : public Subscriber {
public:
    Detector(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store
    );
    ~Detector();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) override;
    void notify(std::shared_ptr<Event> event) override;
    virtual std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) = 0;

protected:
    std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_write_queue;
    std::set<EventType> listening_events;
    std::shared_ptr<ImageStore> image_store;
    void _start();
    std::thread queue_thread;
    bool running = false;
    bool queue_registered = false;
    void poll_read_queue();
    std::queue<std::shared_ptr<FrameEvent>> event_read_queue;
};

// create subclass of Detector for motion detection

class MotionDetector : public Detector {
public:
    MotionDetector(
        std::shared_ptr<ImageStore> image_store,
        int threshold,
        int blur,
        int dilate,
        int threshold_area,
        int activation_frames,
        std::chrono::seconds max_delay
    );
    ~MotionDetector();
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override;
private:
    int threshold;
    int blur;
    int dilate;
    int threshold_area;
    int activation_frames;
    std::chrono::seconds max_delay;
    std::optional<cv::Mat> previous_image;
    int motion_frames;
    std::vector<Detection> detections;
    cv::Mat preprocess_image(cv::Mat image);
};