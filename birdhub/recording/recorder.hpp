#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <set>
#include <queue>
#include <deque>
#include <map>
#include <mutex>
#include "mm.hpp"
#include <chrono>

#ifndef BIRDHUB_RECORDING_HPP
#define BIRDHUB_RECORDING_HPP

class Recorder : public Subscriber {
public:
    Recorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store
    );
    ~Recorder();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) override;
    void notify(Event event) override;
protected:
    std::set<EventType> listening_events;
    // add video frame size
    cv::Size frame_size = cv::Size(1080, 1920); // default size
    int fps = 30;
    std::shared_ptr<ImageStore> image_store;
    std::shared_ptr<std::queue<Event>> event_queue;
    std::thread recording_thread;
    bool running = false;
    bool queue_registered = false;
    void _start();
    // define pure virtual functions for handling events
    virtual void handle_new_frame(Event event) = 0;
    virtual void handle_detection(Event event) = 0;
    virtual void handle_effector_action(Event event) = 0;
    void poll_read_queue();
    // video writer for recording
    cv::VideoWriter video_writer;
    std::queue<Event> event_read_queue;
    std::mutex queue_mutex; // Protect access to event_read_queue
};

// Add continuous recorder class that records video continuously, ignores all events except new_frame and adds it to the video
class ContinuousRecorder : public Recorder {
public:
    ContinuousRecorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store
    );
    ~ContinuousRecorder();
protected:
    void handle_new_frame(Event event) override;
    void handle_detection(Event event) override;
    void handle_effector_action(Event event) override;
};




#endif