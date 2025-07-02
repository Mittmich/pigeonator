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

void Recorder::set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) {
    this->event_queue = event_queue;
    this->queue_registered = true;
}

void Recorder::notify(std::shared_ptr<Event> event) {
    // check if event is in listening events
    if (listening_events.find(event->type) == listening_events.end()) {
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
    std::optional<std::shared_ptr<Event>> event_opt;
    
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
    
    std::shared_ptr<Event> event = event_opt.value();
    
    // Process event outside the lock with proper casting
    switch (event->type) {
        case EventType::NEW_FRAME: {
            std::shared_ptr<FrameEvent> frame_event = std::static_pointer_cast<FrameEvent>(event);
            handle_new_frame(frame_event);
            break;
        }
        case EventType::DETECTION: {
            std::shared_ptr<DetectionEvent> detection_event = std::static_pointer_cast<DetectionEvent>(event);
            handle_detection(detection_event);
            break;
        }
        case EventType::EFFECTOR_ACTION: {
            handle_effector_action(event);
            break;
        }
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

void ContinuousRecorder::handle_new_frame(std::shared_ptr<FrameEvent> frame_event) {
    // Check if the image exists in the image store
    if (!image_store->get(frame_event->get_timestamp()).has_value()) {
        return; // No image available for this timestamp
    }
    cv::Mat frame = image_store->get(frame_event->get_timestamp()).value();
    // Write the frame to the video file
    video_writer.write(frame);
}
void ContinuousRecorder::handle_detection(std::shared_ptr<DetectionEvent> detection_event) {
    // For continuous recording, we ignore detection events
}
void ContinuousRecorder::handle_effector_action(std::shared_ptr<Event> effector_event) {
    // For continuous recording, we ignore effector action events
};

// EventRecorder class implementation

EventRecorder::EventRecorder(
    std::set<EventType> listening_events,
    std::shared_ptr<ImageStore> image_store,
    const std::string& output_directory,
    int slack,
    int fps,
    int look_back_frames,
    int detection_buffer_size
) : Recorder(listening_events, image_store, output_directory), slack(slack), fps(fps), look_back_frames(look_back_frames), detection_buffer_size(detection_buffer_size){
    
}

EventRecorder::~EventRecorder() {
    if (running) {
        stop();
    }
};

void EventRecorder::_update_buffers(std::shared_ptr<FrameEvent> frame_event) {
    // Add the frame event to the video buffer
    video_buffer.push_back(frame_event->get_timestamp());
    // Add the frame event to the detection buffer if it is a detection event
    detection_video_buffer.push_back(frame_event->get_timestamp());
    // If we have more frames than look_back_frames, remove the oldest one
    if (video_buffer.size() > look_back_frames) {
        video_buffer.pop_front();
    }
}

void EventRecorder::_clear_buffers() {
    // Clear all buffers
    video_buffer.clear();
    detection_video_buffer.clear();
    detection_buffer.clear();
    effector_buffer.clear();
}

void EventRecorder::_close_video_writers() {
    // Close the video writer if it is opened
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    if (detection_writer.isOpened()) {
        detection_writer.release();
    }
}

std::vector<FrameEvent> EventRecorder::create_detection_frames(std::shared_ptr<DetectionEvent> detection_event) {
    std::vector<FrameEvent> detection_frames;
    if (this->recording) {
        for (const auto& timestamp : detection_video_buffer) {
            // Create a detection frame for each timestamp in the detection video buffer
            FrameEvent detection_frame = this->create_detection_frame(detection_event, timestamp);
            detection_frames.push_back(detection_frame);
        }
    } else {
        for (const auto& timestamp : video_buffer) {
            // Create a detection frame for each timestamp in the video buffer
            FrameEvent detection_frame = this->create_detection_frame(detection_event, timestamp);
            detection_frames.push_back(detection_frame);
        }
    }
    return detection_frames;
}

FrameEvent EventRecorder::create_detection_frame(std::shared_ptr<DetectionEvent>  detection_event, Timestamp frame_timestamp) {
    // Retrive the image from the image store
    if (image_store->get(frame_timestamp).has_value()) {
        cv::Mat frame = image_store->get(frame_timestamp).value();
        // Search for the detection in the detection event that matches the frame timestamp
        auto detections = detection_event->get_detections();
        for (auto& detection : detections) {
            if (detection.get_frame_event()->get_timestamp() == frame_timestamp) {
                // Draw bounding boxes on the frame if available
                auto bounding_boxes_opt = detection.get_bounding_boxes();
                auto labels_opt = detection.get_labels(); // Assuming this method exists to get labels

                if (bounding_boxes_opt.has_value()) {
                    const auto& boxes_vec = bounding_boxes_opt.value();
                    std::vector<std::string> labels_vec;
                    if (labels_opt.has_value()) {
                        labels_vec = labels_opt.value();
                    }

                    for (size_t i = 0; i < boxes_vec.size(); ++i) {
                        const auto& box = boxes_vec[i];
                        // Draw the bounding box with red lines (BGR format), thickness 5
                        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 5);

                        if (i < labels_vec.size() && !labels_vec[i].empty()) {
                            const std::string& label_text = labels_vec[i];
                            // Position text near the top-left of the box, or adjust as needed
                            // Python: (x1, y2 + 15) -> (box.tl().x, box.br().y + 15)
                            cv::Point text_origin(box.tl().x, box.br().y + 15);
                            cv::putText(frame, label_text, text_origin, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1); // White text, thickness 1
                        }
                    }
                }
            }
        }
        // Put the modified image back to the store so _write_detections uses it
        image_store->put(frame_timestamp, frame);
    }
    return FrameEvent(
        frame_timestamp,
        std::make_optional<std::map<std::string, std::string>>({
            {"type", "detection_frame"},
            {"detection_count", std::to_string(detection_event->get_detections().size())}
        })
    );
}


void EventRecorder::_write_detections() {
    // Add all activations that are recorded to the detections
    for (auto& activation : effector_buffer) {
        // Find detections within 2 seconds of the activation
        std::vector<Timestamp> write_timestamps;
        for (auto& detection : detection_buffer) {
            auto time_diff = std::abs(std::chrono::duration_cast<std::chrono::seconds>(detection.get_timestamp() - activation->get_timestamp()).count());
            if (time_diff < 2) { // Within 2 seconds
                write_timestamps.push_back(detection.get_timestamp());
            }
        }
        
        // Add activation overlay to frames
        for (auto& detection_frame : detection_buffer) {
            _add_activation_overlay(detection_frame, activation, write_timestamps);
        }
    }
    // sort the detection buffer by timestamp
    std::sort(detection_buffer.begin(), detection_buffer.end(),
        [](FrameEvent& a, FrameEvent& b) {
            return a.get_timestamp() < b.get_timestamp();
        });
    
    // Create detection video writer if not already created
    if (!detection_writer.isOpened()) {
        std::string filename = "detection_" + std::to_string(std::time(nullptr)) + ".mp4";
        std::filesystem::path full_path = std::filesystem::path(output_directory) / filename;
        detection_writer.open(full_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, this->frame_size, true);
        if (!detection_writer.isOpened()) {
            throw std::runtime_error("Could not open detection video writer.");
        }
    }
    
    // Write all detection frames to the detection video
    for (auto& detection_frame : detection_buffer) {
        if (image_store->get(detection_frame.get_timestamp()).has_value()) {
            cv::Mat frame = image_store->get(detection_frame.get_timestamp()).value();
            detection_writer.write(frame);
        }
    }
    
    // Clear buffers after writing
    detection_buffer.clear();
    effector_buffer.clear();
}

void EventRecorder::_update_detections(std::shared_ptr<DetectionEvent> detection_event) {
    std::vector<FrameEvent> detection_frames = this->create_detection_frames(detection_event);
    // Add the detection frames to the detection buffer
    for (auto& detection : detection_frames) {
        // add frame to the detection buffer, if it is not already present
        auto it = std::find_if(detection_buffer.begin(), detection_buffer.end(),
            [&detection](FrameEvent& existing_frame) {
                return existing_frame.get_timestamp() == detection.get_timestamp();
            });
        if (it == detection_buffer.end()) {
            detection_buffer.push_back(detection);
        }
    }
}

void EventRecorder::handle_new_frame(std::shared_ptr<FrameEvent> frame_event) {
    // Check if the image exists in the image store
    if (!image_store->get(frame_event->get_timestamp()).has_value()) {
        // log the missing image
        std::cerr << "No image available " << std::endl;
        // Return early if no image is available
        return; // No image available for this timestamp
    }
    cv::Mat frame = image_store->get(frame_event->get_timestamp()).value();
    this->_update_buffers(frame_event);
    // Check  whether we should write detections to video
    if (this->detection_buffer.size() > this->detection_buffer_size) {
        // create empty detection event to populate buffers
        //DetectionEvent empty_detection_event(frame_event->get_timestamp(), {}, std::nullopt);
        //this->_update_detections(std::make_shared<DetectionEvent>(empty_detection_event));
        // log
        std::cout << "Writing detections to video." << std::endl;
        this->_write_detections();
        this->detection_video_buffer.clear();
    }
    if (this->_stop_recording_in > 0) {
        this->video_writer.write(frame);
        this->_stop_recording_in--;
    } else if (this->video_writer.isOpened()) {
        // call update detections with empty detection event
        // Log the end of recording
        std::cout << "Stopping recording."  << std::endl;
        // Create an empty detection event to signal the end of recording
        DetectionEvent empty_detection_event(frame_event->get_timestamp(), {}, std::nullopt);
        this->_update_detections(std::make_shared<DetectionEvent>(empty_detection_event));
        this->_write_detections();
        this->_clear_buffers();
        this->_close_video_writers();
        this->recording = false;
    }
};

void EventRecorder::handle_detection(std::shared_ptr<DetectionEvent> detection_event) {
    if (this->recording){
        this->_stop_recording_in = this->slack;
        this->_update_detections(detection_event);
        // clear look back frames buffer
        video_buffer.clear();
    } else {
        // Start recording if not already recording
        this->_stop_recording_in = this->slack;
        this->recording_start_time = std::chrono::steady_clock::now();
        // Create a video writer if not already created
        if (!video_writer.isOpened()) {
            // log the start of recording
            std::cout << "Starting recording." << std::endl;
            // Create a video writer with a filename in the specified
            std::string filename = "recording_" + std::to_string(std::time(nullptr)) + ".mp4";
            std::filesystem::path full_path = std::filesystem::path(output_directory) / filename;
            video_writer.open(full_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, this->frame_size, true);
            if (!video_writer.isOpened()) {
                throw std::runtime_error("Could not open video writer.");
            }
        }
        // Write look back frames to the video writer
        for (const auto& timestamp : video_buffer) {
            if (image_store->get(timestamp).has_value()) {
                cv::Mat frame = image_store->get(timestamp).value();
                video_writer.write(frame);
            }
        }
        // clear look back frames buffer
        this->_update_detections(detection_event);
        video_buffer.clear();
        this->recording = true;
    }
}

void EventRecorder::handle_effector_action(std::shared_ptr<Event> effector_event) {
    // Add the effector action event to the buffer for later processing
    effector_buffer.push_back(effector_event);
}

void EventRecorder::_add_activation_overlay(FrameEvent detection_frame, std::shared_ptr<Event> activation, const std::vector<Timestamp>& write_timestamps) {
    // Check if this frame's timestamp is in the write_timestamps list
    bool should_write = false;
    for (Timestamp timestamp : write_timestamps) {
        if (detection_frame.get_timestamp() == timestamp) {
            should_write = true;
            break;
        }
    }
    
    if (!should_write) {
        return;
    }
    
    // Get the image from the image store
    if (!image_store->get(detection_frame.get_timestamp()).has_value()) {
        return;
    }
    
    cv::Mat frame = image_store->get(detection_frame.get_timestamp()).value();
    
    // Get activation type from metadata (default to "ACTIVATION" if not available)
    std::string activation_type = "ACTIVATION";
    if (activation->get_meta_data().count("type") > 0) {
        activation_type = activation->get_meta_data().at("type");
    }
    
    // Calculate text size and position
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 7.0;
    int thickness = 2;
    cv::Size textSize = cv::getTextSize(activation_type, fontFace, fontScale, thickness, nullptr);
    
    // Center the text on the frame
    int textX = (frame.cols - textSize.width) / 2;
    int textY = (frame.rows + textSize.height) / 2;
    
    // Add the text overlay in red color
    cv::putText(frame, activation_type, cv::Point(textX, textY), fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
    
    // Put the modified image back to the store
    image_store->put(detection_frame.get_timestamp(), frame);
}

// Override stop method to ensure all resources are cleaned up
void EventRecorder::stop() {
    if (running) {
        if (this->recording) {
            DetectionEvent empty_detection_event(now(), {}, std::nullopt);
            this->_update_detections(std::make_shared<DetectionEvent>(empty_detection_event));
            this->_write_detections();
        }
        this->running = false;
    }
    if (recording_thread.joinable()) {
        recording_thread.join();
    }
    this->_close_video_writers();
    this->_clear_buffers();
    // log
    std::cout << "EventRecorder stopped." << std::endl;
}