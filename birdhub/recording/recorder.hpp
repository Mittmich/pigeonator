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
#include <fstream>
#include <vector>
#include <string>

#ifndef BIRDHUB_RECORDING_HPP
#define BIRDHUB_RECORDING_HPP

class Recorder : public Subscriber {
public:
    Recorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store,
    const std::string& output_directory = ".",
    cv::Size frame_size = cv::Size(1280, 720)
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
    cv::Size frame_size = cv::Size(1280, 720); // default size
    int fps = 10;
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
    const std::string& output_directory = ".",
    cv::Size frame_size = cv::Size(1280, 720)
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
    cv::Size frame_size = cv::Size(1280, 720)
    );
    ~EventRecorder();
    void stop() override;
    void start() override;
protected:
    void handle_new_frame(std::shared_ptr<FrameEvent> frame_event) override;
    void handle_detection(std::shared_ptr<DetectionEvent> detection_event) override;
    void handle_effector_action(std::shared_ptr<Event> effector_event) override;
    void _update_buffers(std::shared_ptr<FrameEvent> frame_event);
    void _write_frame_to_filebuffer(std::shared_ptr<FrameEvent> frame_event);
    void _write_detections_to_filebuffer(std::shared_ptr<DetectionEvent> detection_event);
    void _write_effectorevent_to_filebuffer(std::shared_ptr<Event> effector_event);
    void _create_filebuffers();
    void _create_outputs_from_filebuffers();
    void _write_detections();
    void _clear_buffers();
    FrameEvent create_detection_frame(std::shared_ptr<DetectionEvent> detection_event, Timestamp frame_timestamp);
    void _add_activation_overlay(FrameEvent detection_frame, std::shared_ptr<Event> activation, const std::vector<Timestamp>& write_timestamps);
    // Additional parameters for event recording
    int slack; // Number of frames wait until stop recording
    int fps; // frames per second for the video
    int look_back_frames; // number of frames to look back for event recording
    int _stop_recording_in = 0;
    bool recording = false; // Flag to indicate if recording is in progress
    // Video writer for recording
    cv::VideoWriter videobuffer_writer;
    std::filesystem::path video_buffer_full_path;
    std::filesystem::path video_timestamp_buffer_full_path;
    std::filesystem::path detection_buffer_full_path;
    std::filesystem::path effector_event_full_path;
    cv::VideoWriter detection_writer;
    std::ofstream detection_buffer_file;
    std::ofstream video_timestamp_buffer_file;
    std::ofstream effector_buffer_file;
    // Buffer for video frames
    std::deque<std::shared_ptr<FrameEvent>> video_buffer;
    // recording start time
    std::chrono::time_point<std::chrono::steady_clock> recording_start_time;
};


#endif