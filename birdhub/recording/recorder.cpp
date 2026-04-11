#include "recorder.hpp"
#include <string>
#include <optional>
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>


Recorder::Recorder(
    std::set<EventType> listening_events,
    std::shared_ptr<ImageStore> image_store,
    const std::string& output_directory,
    cv::Size frame_size
) : listening_events(listening_events), image_store(image_store), output_directory(output_directory) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_directory);
    // Set desired frame size
    this->frame_size = frame_size;
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
    const std::string& output_directory,
    cv::Size frame_size
) : Recorder(listening_events, image_store, output_directory, frame_size) {
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
    cv::Size frame_size
) : Recorder(listening_events, image_store, output_directory, frame_size), slack(slack), fps(fps), look_back_frames(look_back_frames) {

}


// Override start to not open a video file

void EventRecorder::start() {
    // check if event queue is registered
    if (!this->queue_registered) {
        throw std::runtime_error("Event queue not registered.");
    }
    // set running flag that is used to stop the thread
    this->running = true;
    // start thread
    this->recording_thread = std::thread(&EventRecorder::_start, this);
}

EventRecorder::~EventRecorder() {
    if (running) {
        stop();
    }
};

void EventRecorder::_update_buffers(std::shared_ptr<FrameEvent> frame_event) {
    // Add the frame event to the video buffer
    video_buffer.push_back(frame_event);
    // Add the frame event to the detection buffer if it is a detection event
    if (video_buffer.size() > look_back_frames) {
        video_buffer.pop_front();
    }
}

void EventRecorder::_clear_buffers() {
    // Clear all buffers
    video_buffer.clear();
}



void EventRecorder::_write_frame_to_filebuffer(std::shared_ptr<FrameEvent> frame_event) {
    if (videobuffer_writer.isOpened()) {
        cv::Mat frame = image_store->get(frame_event->get_timestamp()).value();
        videobuffer_writer.write(frame);
    }
    // update video timestamp buffer
    if (video_timestamp_buffer_file.is_open()) {
        video_timestamp_buffer_file << frame_event->get_timestamp().time_since_epoch().count() << std::endl;
    }
}

void EventRecorder::_write_detections_to_filebuffer(std::shared_ptr<DetectionEvent> detection_event) {
    // Write detections to the detection buffer file
    // get detections from detection_event
    if (detection_buffer_file.is_open()) {
        for (const auto& detection : detection_event->get_detections()) {
            // Iterate over all labels and bounding boxes and write them to the file with the same timestamp
            auto labels = detection.get_labels();
            auto boxes = detection.get_bounding_boxes();
            auto track_uuids = detection.get_track_uuids();
            if (boxes.has_value() && labels.has_value()) {
                auto boxes_vec = boxes.value();
                auto labels_vec = labels.value();
                std::vector<std::string> uuid_vec;
                if (track_uuids.has_value()) {
                    uuid_vec = track_uuids.value();
                }
                for (size_t i = 0; i < boxes_vec.size(); ++i) {
                    const auto& box = boxes_vec[i];
                    const auto& label = labels_vec[i];
                    detection_buffer_file << detection.get_frame_event()->get_timestamp().time_since_epoch().count() << ","
                                          << box.x << "," << box.y << "," << box.width << "," << box.height << ","
                                          << label;
                    if (!uuid_vec.empty() && i < uuid_vec.size()) {
                        detection_buffer_file << "," << uuid_vec[i];
                    }
                    detection_buffer_file << std::endl;
                }
            }
        }
    }
}

void EventRecorder::_write_effectorevent_to_filebuffer(std::shared_ptr<Event> effector_event) {
    // Write effector action to the effector buffer file
    if (effector_buffer_file.is_open()) {
        // get effector type
        try {
            // copy value out of potentially temporary metadata map      avoid dangling reference
            std::string effector_type = effector_event->get_meta_data().at("type");
            effector_buffer_file << effector_event->get_timestamp().time_since_epoch().count() << ","
                                 << effector_type << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Effector type not found: " << e.what() << std::endl;
        }
    }
}

void EventRecorder::_create_outputs_from_filebuffers() {
    // Close all file buffers
    if (videobuffer_writer.isOpened()) {
        videobuffer_writer.release();
    }
    if (video_timestamp_buffer_file.is_open()) {
        video_timestamp_buffer_file.close();
    }
    if (detection_buffer_file.is_open()) {
        detection_buffer_file.close();
    }
    if (effector_buffer_file.is_open()) {
        effector_buffer_file.close();
    }
    // If no frames were written (empty file), skip output creation gracefully
    if (!std::filesystem::exists(video_buffer_full_path) ||
        std::filesystem::file_size(video_buffer_full_path) == 0) {
        std::cerr << "No frames in video buffer; skipping output creation." << std::endl;
        return;
    }

    // Load video buffer into memory -> this is an mp4 file
    auto video_reader = cv::VideoCapture(video_buffer_full_path.string());
    if (!video_reader.isOpened()) {
        std::cerr << "Warning: could not open video buffer at '" << video_buffer_full_path << "'; skipping output creation." << std::endl;
        return;
    }
    // Load video frames into memory
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (video_reader.read(frame)) {
        // clone to avoid all entries referencing the same underlying buffer
        frames.push_back(frame.clone());
    }
    video_reader.release();
    // load frame timestamp from text file
    std::ifstream video_timestamp_buffer_in(video_timestamp_buffer_full_path.string());
    if (!video_timestamp_buffer_in.is_open()) {
        std::cerr << "Warning: could not open video timestamp buffer file at '" << video_timestamp_buffer_full_path << "'. Skipping output creation." << std::endl;
        return;
    }
    std::string line;
    std::vector<Timestamp> frame_timestamps;
    while (std::getline(video_timestamp_buffer_in, line)) {
        // Parse the timestamp and associate it with the corresponding frame
        auto timestamp = std::stoll(line);
        // cast timestamp to Timestamp type
        Timestamp frame_timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(std::chrono::milliseconds(timestamp));
        frame_timestamps.push_back(frame_timestamp);
    }
    // load detections from text file
    std::vector<Timestamp> detection_timestamps;
    std::vector<std::vector<int>> detection_bounding_boxes;
    std::vector<std::string> detection_labels;
    std::vector<std::string> detection_track_uuids; // optional parallel vector (empty string if none)
    std::ifstream detection_buffer_in(detection_buffer_full_path.string());
    if (!detection_buffer_in.is_open()) {
        std::cerr << "Warning: could not open detection buffer file at '" << detection_buffer_full_path << "'. Proceeding without detections." << std::endl;
    }
    while (detection_buffer_in.is_open() && std::getline(detection_buffer_in, line)) {
        // Parse the detection information: the structure is timestamp, bbox1, bbox2, bbox3, bbox4, label
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() == 6 || tokens.size() == 7) {
            // Extract the detection information
            Timestamp detection_timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(std::chrono::milliseconds(std::stoll(tokens[0])));
            std::vector<int> detection_bounding_box = {
                std::stoi(tokens[1]), // x
                std::stoi(tokens[2]), // y
                std::stoi(tokens[3]), // width
                std::stoi(tokens[4])  // height
            };
            std::string detection_label = tokens[5];
            std::string track_uuid = (tokens.size() == 7) ? tokens[6] : std::string("");
            // Store the detection information
            detection_timestamps.push_back(detection_timestamp);
            detection_bounding_boxes.push_back(detection_bounding_box);
            detection_labels.push_back(detection_label);
            detection_track_uuids.push_back(track_uuid);
        } else {
            std::cerr << "Invalid detection line (expected 6 or 7 tokens): " << line << std::endl;
            continue;
        }
    }
    // close the files
    video_timestamp_buffer_in.close();
    detection_buffer_in.close();
    // effector events currently unused in rendering
    // create new writer
    cv::VideoWriter detection_writer;
    std::string detection_output_filename = "detections_" + std::to_string(std::time(nullptr)) + ".mp4";
    detection_output_path = std::filesystem::path(output_directory) / detection_output_filename;
    detection_writer.open(detection_output_path.string(),
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, this->frame_size, true);
    if (!detection_writer.isOpened()) {
        throw std::runtime_error("Could not open detection video writer.");
    }
    // iterate over all frames and add all detection bounding boxes and labels to it
    const size_t n = std::min(frames.size(), frame_timestamps.size());
    if (frames.size() != frame_timestamps.size()) {
        std::cerr << "Warning: frames count (" << frames.size() << ") and timestamps count (" << frame_timestamps.size() << ") differ; using min = " << n << std::endl;
    }
    // Prepare color map based on track UUIDs
    std::unordered_map<std::string, cv::Scalar> uuid_color_map;
    auto color_for_uuid = [&uuid_color_map](const std::string& uuid) -> cv::Scalar {
        if (uuid.empty()) {
            return cv::Scalar(0, 255, 0); // default green
        }
        auto it = uuid_color_map.find(uuid);
        if (it != uuid_color_map.end()) {
            return it->second;
        }
        // Hash uuid to hue 0-179
        size_t h = std::hash<std::string>{}(uuid);
        int hue = static_cast<int>(h % 180);
        int sat = 200 + (h % 55); // 200-254
        int val = 200 + ((h / 180) % 55);
        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(hue, sat, val));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Vec3b bgr_pix = bgr.at<cv::Vec3b>(0,0);
        cv::Scalar color(bgr_pix[0], bgr_pix[1], bgr_pix[2]);
        uuid_color_map[uuid] = color;
        return color;
    };

    for (size_t i = 0; i < n; ++i) {
        // Get the current frame timestamp
        Timestamp frame_timestamp = frame_timestamps[i];
        // get the current frame
        cv::Mat current_frame = frames[i].clone();
        // Find all detections for the current frame
        for (size_t j = 0; j < detection_timestamps.size(); ++j) {
            if (detection_timestamps[j] == frame_timestamp) {
                // If the timestamps match, draw the detection bounding box and label
                const auto& bbox = detection_bounding_boxes[j];
                const auto& label = detection_labels[j];
                std::string uuid = (j < detection_track_uuids.size()) ? detection_track_uuids[j] : std::string("");
                // bbox = [x, y, width, height]
                const int x = bbox[0];
                const int y = bbox[1];
                const int w = bbox[2];
                const int h = bbox[3];
                cv::Scalar color = color_for_uuid(uuid);
                std::string display_label = label;
                if (!uuid.empty()) {
                    // append short uuid (first 8 chars) for disambiguation
                    display_label += " [" + uuid.substr(0,8) + "]";
                }
                cv::rectangle(current_frame, cv::Rect(x, y, w, h), color, 2);
                cv::putText(current_frame, display_label, cv::Point(x, std::max(0, y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
            }
        }
        // Write the frame with detections to the video writer
        detection_writer.write(current_frame);
    }
    // close the video writer
    detection_writer.release();

    // Emit RECORDING_STOPPED event so subscribers (e.g. EventDispatcher) can upload the file
    if (event_queue && std::filesystem::exists(detection_output_path)) {
        Timestamp recording_end = now();
        event_queue->push(std::make_shared<RecordingStoppedEvent>(
            recording_start_system,
            recording_end,
            detection_output_path.string()
        ));
    }
}

void EventRecorder::_create_filebuffers() {
    // create image buffer -> this will be a mp4 video
    std::string video_buffer_filename = "recording_" + std::to_string(std::time(nullptr)) + ".mp4";
    video_buffer_full_path = std::filesystem::path(output_directory) / video_buffer_filename;
    videobuffer_writer.open(video_buffer_full_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, this->frame_size, true);
    if (!videobuffer_writer.isOpened()) {
        throw std::runtime_error("Could not open video writer.");
    }
    // create video timestamp buffer as a text file that stores alle timestamps of each frame in the video buffer
    std::string timestamp_filename = "videobuffer_timestamps_" + std::to_string(std::time(nullptr)) + ".txt";
    video_timestamp_buffer_full_path = std::filesystem::path(output_directory) / timestamp_filename;
    video_timestamp_buffer_file.open(video_timestamp_buffer_full_path.string(), std::ios::out);
    if (!video_timestamp_buffer_file.is_open()) {
        throw std::runtime_error("Could not open video timestamp buffer file.");
    }
    // create detection buffer
    std::string detection_buffer_filename = "detection_buffer_" + std::to_string(std::time(nullptr)) + ".txt";
    detection_buffer_full_path = std::filesystem::path(output_directory) / detection_buffer_filename;
    detection_buffer_file.open(detection_buffer_full_path.string(), std::ios::out);
    if (!detection_buffer_file.is_open()) {
        throw std::runtime_error("Could not open detection buffer file.");
    }
    // create effectorevent text file
    std::string effector_event_filename = "effectorevent_buffer_" + std::to_string(std::time(nullptr)) + ".txt";
    effector_event_full_path = std::filesystem::path(output_directory) / effector_event_filename;
    effector_buffer_file.open(effector_event_full_path.string(), std::ios::out);
    if (!effector_buffer_file.is_open()) {
        throw std::runtime_error("Could not open effector buffer file.");
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
    this->_update_buffers(frame_event);
    if (this->_stop_recording_in > 0) {
        this->_write_frame_to_filebuffer(frame_event);
        this->_stop_recording_in--;
        // If countdown just reached zero, finalize outputs immediately
        if (this->_stop_recording_in <= 0 && this->videobuffer_writer.isOpened()) {
            std::cout << "Stopping recording."  << std::endl;
            this->_create_outputs_from_filebuffers();
            this->_clear_buffers();
            this->recording = false;
        }
    } else if (this->videobuffer_writer.isOpened()) {
        // Log the end of recording
        std::cout << "Stopping recording."  << std::endl;
        this->_create_outputs_from_filebuffers();
        this->_clear_buffers();
        this->recording = false;
    }
};

void EventRecorder::handle_detection(std::shared_ptr<DetectionEvent> detection_event) {
    if (this->recording){
        this->_stop_recording_in = this->slack;
        this->_write_detections_to_filebuffer(detection_event);
    } else {
        // Start recording if not already recording
        this->_stop_recording_in = this->slack;
        this->recording_start_time = std::chrono::steady_clock::now();
        this->recording_start_system = now();
        this->_create_filebuffers();
        // Write everything in the video buffer to the file
        for (const auto& frame_event : video_buffer) {
            this->_write_frame_to_filebuffer(frame_event);
        }
        // write detections to file
        this->_write_detections_to_filebuffer(detection_event);
        // clear video buffer
        video_buffer.clear();
        // log the start of recording
        std::cout << "Starting recording." << std::endl;
        this->recording = true;
    }
}

void EventRecorder::handle_effector_action(std::shared_ptr<Event> effector_event) {
    // Add the effector action event to the buffer for later processing
    this->_write_effectorevent_to_filebuffer(effector_event);
}

// Override stop method to ensure all resources are cleaned up
void EventRecorder::stop() {
    if (running) {
        if (this->recording) {
            this->_create_outputs_from_filebuffers();
            this->_clear_buffers();
            this->recording = false;
        }
        this->running = false;
    }
    if (recording_thread.joinable()) {
        recording_thread.join();
    }
    this->_clear_buffers();
    // log
    std::cout << "EventRecorder stopped." << std::endl;
}