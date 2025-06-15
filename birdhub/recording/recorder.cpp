#include "recorder.hpp"
#include <string>
#include <optional>
#include <iostream>


Recorder::Recorder(
    std::set<EventType> listening_events,
    std::shared_ptr<ImageStore> image_store,
    const std::string& output_directory
) : listening_events(listening_events), image_store(image_store), output_directory(output_directory) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_directory);
};

Recorder::~Recorder() {
    if (running) {
        stop();
    }
    if (recording_thread.joinable()) {
        recording_thread.join();
    }
}

std::set<EventType> Recorder::listening_to() {
    return listening_events;
}

void Recorder::set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) {
    this->event_queue = event_queue;
    this->queue_registered = true;
}

void Recorder::notify(Event event) {
    // check if event is in listening events
    if (listening_events.find(event.type) == listening_events.end()) {
        return;
    }
    // thread-safe push the event to the read queue
    std::lock_guard<std::mutex> lock(queue_mutex);
    this->event_read_queue.push(event);
}

void Recorder::start() {
    // check if event queue is registered
    if (!this->queue_registered) {
        throw std::runtime_error("Event queue not registered.");
    }
    // set running flag that is used to stop the thread
    this->running = true;
    // open video writer if not already opened
    if (!video_writer.isOpened()) {
        // create a video writer with a filename in the specified directory
        std::string filename = "recording_" + std::to_string(std::time(nullptr)) + ".mp4";
        std::filesystem::path full_path = std::filesystem::path(output_directory) / filename;
        video_writer.open(full_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, this->frame_size, true);
        if (!video_writer.isOpened()) {
            throw std::runtime_error("Could not open video writer.");
        }
    }
    // start thread
    this->recording_thread = std::thread(&Recorder::_start, this);
}

void Recorder::stop() {
    // set running flag to false to stop the thread
    this->running = false;
    // join the thread
    if (this->recording_thread.joinable()){
        this->recording_thread.join();
    }
    // release video writer
    if (video_writer.isOpened()) {
        video_writer.release();
    }
}

void Recorder::_start() {
    while (this->running) {
        poll_read_queue();
    }
}

void Recorder::poll_read_queue() {
    std::optional<Event> event_opt;
    
    // Thread-safe check and pop from queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (!this->event_read_queue.empty()) {
            event_opt = this->event_read_queue.front();
            this->event_read_queue.pop();
        }
    }
    
    // If no event was available, return early
    if (!event_opt.has_value()) {
        return;
    }
    
    Event event = event_opt.value();
    
    // Process event outside the lock
    switch (event.type) {
        case EventType::NEW_FRAME:
            handle_new_frame(event);
            break;
        case EventType::DETECTION:
            handle_detection(event);
            break;
        case EventType::EFFECTOR_ACTION:
            handle_effector_action(event);
            break;
        default:
            // Unknown event type, do nothing
            break;
    }
}

// ContinuousRecorder class implementation
ContinuousRecorder::ContinuousRecorder(
    std::set<EventType> listening_events,
    std::shared_ptr<ImageStore> image_store,
    const std::string& output_directory
) : Recorder(listening_events, image_store, output_directory) {
}
ContinuousRecorder::~ContinuousRecorder() {
    if (running) {
        stop();
    }
}

void ContinuousRecorder::handle_new_frame(Event event) {
    // Check if this is actually a FrameEvent
    if (event.type != EventType::NEW_FRAME) {
        return;
    }
    
    // Since we confirmed it's a NEW_FRAME event type, we can safely cast
    // In a production system, you might want to use dynamic_cast for safety
    FrameEvent& frame_event = static_cast<FrameEvent&>(event);
    
    // log that we got here
    std::cout << "Handling new frame event at timestamp: " << frame_event.get_timestamp() << std::endl;
    // Check if the image exists in the image store
    if (!image_store->get(frame_event.get_timestamp()).has_value()) {
        return; // No image available for this timestamp
    }
    std::cout << "Image found for timestamp: " << frame_event.get_timestamp() << std::endl;
    cv::Mat frame = image_store->get(frame_event.get_timestamp()).value();
    // Write the frame to the video file
    video_writer.write(frame);
}
void ContinuousRecorder::handle_detection(Event event) {
    // For continuous recording, we ignore detection events
}
void ContinuousRecorder::handle_effector_action(Event event) {
    // For continuous recording, we ignore effector action events
}