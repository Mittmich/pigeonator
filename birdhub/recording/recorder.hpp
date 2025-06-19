#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <set>
#include <queue>
#include <deque>
#include <map>
#include <mutex>
#include <filesystem>
#include "mm.hpp"
#include <chrono>

#ifndef BIRDHUB_RECORDING_HPP
#define BIRDHUB_RECORDING_HPP

class Recorder : public Subscriber {
public:
    Recorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store,
        const std::string& output_directory = "."
    );
    ~Recorder();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) override;
    void notify(std::shared_ptr<Event> event) override;
protected:
    std::set<EventType> listening_events;
    // add video frame size
    cv::Size frame_size = cv::Size(1920, 1080); // default size
    int fps = 30;
    std::string output_directory;
    std::shared_ptr<ImageStore> image_store;
    std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue;
    std::thread recording_thread;
    bool running = false;
    bool queue_registered = false;
    void _start();
    // define pure virtual functions for handling events
    virtual void handle_new_frame(std::shared_ptr<FrameEvent> frame_event) = 0;
    virtual void handle_detection(std::shared_ptr<DetectionEvent> detection_event) = 0;
    virtual void handle_effector_action(std::shared_ptr<Event> effector_event) = 0;
    void poll_read_queue();
    // video writer for recording
    cv::VideoWriter video_writer;
    std::queue<std::shared_ptr<Event>> event_read_queue;
    std::mutex queue_mutex; // Protect access to event_read_queue
};

// Add continuous recorder class that records video continuously, ignores all events except new_frame and adds it to the video
class ContinuousRecorder : public Recorder {
public:
    ContinuousRecorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store,
        const std::string& output_directory = "."
    );
    ~ContinuousRecorder();
protected:
    void handle_new_frame(std::shared_ptr<FrameEvent> frame_event);
    void handle_detection(std::shared_ptr<DetectionEvent> detection_event);
    void handle_effector_action(std::shared_ptr<Event> effector_event);
};

class EventRecorder : public Recorder {
public:
    EventRecorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store,
        const std::string& output_directory = ".",
        int slack = 100,
        int fps = 30,
        int look_back_frames = 3,
        int detection_buffer_size = 200
    );
    ~EventRecorder();
protected:
    void handle_new_frame(std::shared_ptr<FrameEvent> frame_event);
    void handle_detection(std::shared_ptr<DetectionEvent> detection_event);
    void handle_effector_action(std::shared_ptr<Event> effector_event);
    void _update_buffers(std::shared_ptr<FrameEvent> frame_event);
    void _write_detections();
    void _clear_buffers();
    void _close_video_writers();
    void _update_detections(std::shared_ptr<DetectionEvent> detection_event);
    std::vector<FrameEvent> create_detection_frames(std::shared_ptr<DetectionEvent> detection_event);
    FrameEvent create_detection_frame(std::shared_ptr<DetectionEvent> detection_event, Timestamp frame_timestamp);
    void _add_activation_overlay(FrameEvent detection_frame, std::shared_ptr<Event> activation, const std::vector<Timestamp>& write_timestamps);
    // Additional parameters for event recording
    int slack; // Number of frames wait until stop recording
    int fps; // frames per second for the video
    int look_back_frames; // number of frames to look back for event recording
    int detection_buffer_size; // number of frames to buffer for detection events
    int _stop_recording_in = 0;
    bool recording = false; // Flag to indicate if recording is in progress
    // Video writer for recording
    cv::VideoWriter video_writer;
    cv::VideoWriter detection_writer;
    // Buffer for detection events
    std::deque<FrameEvent> detection_buffer;
    // Buffer for effector actions
    std::deque<std::shared_ptr<Event>> effector_buffer;
    // Buffer for video frames
    std::deque<Timestamp> video_buffer;
    std::deque<Timestamp> detection_video_buffer;
    // recording start time
    std::chrono::time_point<std::chrono::steady_clock> recording_start_time;
};


#endif