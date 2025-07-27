#include "detection.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <memory>
#include <deque>

Detector::Detector(
    std::set<EventType> listening_events,
    std::shared_ptr<ImageStore> image_store
) : listening_events(listening_events),
    image_store(image_store) {};

Detector::~Detector() {
    if (running) {
        stop();
    }
    if (queue_thread.joinable()) {
        queue_thread.join();
    }
}

std::set<EventType> Detector::listening_to() {
    return listening_events;
}

void Detector::set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) {
    this->event_write_queue = event_queue;
    this->queue_registered = true;
}

void Detector::notify(std::shared_ptr<Event> event) {
    // check if event is of type FrameEvent
    if (event->type != EventType::NEW_FRAME) {
        return;
    }
    // add event to read queue
    this->event_read_queue.push(std::static_pointer_cast<FrameEvent>(event));
}

void Detector::start() {
    // check if event queue is registered
    if (!this->queue_registered) {
        throw std::runtime_error("Event queue not registered.");
    }
    // set running flag that is used to stop the thread
    this->running = true;
    // start thread
    this->queue_thread = std::thread(&Detector::_start, this);
}

void Detector::stop() {
    // set running flag to false to stop the thread
    this->running = false;
    // join the thread
    if (this->queue_thread.joinable()) {
        this->queue_thread.join();
    }
}

void Detector::_start() {
    while (this->running) {
        poll_read_queue();
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Add sleep
    }
}

void Detector::poll_read_queue() {
    // check if event queue is empty
    if (this->event_read_queue.empty()) {
        return;
    }
    // get event from queue
    std::shared_ptr<FrameEvent> event = this->event_read_queue.front();
    std::optional<DetectionEvent> detections = detect(event);
    // check if detections are present
    if (detections.has_value()) {
        // print detections with a nicely formatted timestamp
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(detections.value().get_timestamp().time_since_epoch()).count();
        std::cout << "Detections at " << timestamp_ms << "ms since epoch" << std::endl;
        this->event_write_queue->push(std::make_shared<DetectionEvent>(detections.value()));
    }
    // remove event from read queue
    this->event_read_queue.pop();
}

MotionDetector::MotionDetector(
    std::shared_ptr<ImageStore> image_store,
    int threshold=20,
    int blur=21,
    int dilate=5,
    int threshold_area=50,
    int activation_frames=5,
    std::chrono::seconds max_delay=std::chrono::seconds(5)
) : Detector({EventType::NEW_FRAME}, image_store),
    threshold(threshold),
    blur(blur),
    dilate(dilate),
    threshold_area(threshold_area),
    activation_frames(activation_frames),
    max_delay(max_delay),
    previous_image(std::nullopt),
    detections({}),
    motion_frames(0) {};

MotionDetector::~MotionDetector() = default;


cv::Mat MotionDetector::preprocess_image(cv::Mat image) {
    // convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // apply gaussian blur
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(blur, blur), 0);
    return blurred;
}

std::optional<DetectionEvent> MotionDetector::detect(std::shared_ptr<FrameEvent> frame_event) {
    // check if frame is delayed
    if (frame_event->get_timestamp() < std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()) - max_delay) {
        return std::nullopt;
    }
    // check whether image exists
    if (!image_store->get(frame_event->get_timestamp()).has_value()) {
        return std::nullopt;
    }
    cv::Mat current_image = image_store->get(frame_event->get_timestamp()).value();
    // preprocess image
    cv::Mat processed_image = preprocess_image(current_image);
    // check if previous image is empty
    if (previous_image.has_value() == false) {
        previous_image = processed_image;
        return std::nullopt;
    }
    // calculate difference between current and previous image
    cv::Mat diff;
    cv::absdiff(previous_image.value(), processed_image, diff);
    // threshold the difference image
    cv::Mat thresholded;
    cv::threshold(diff, thresholded, threshold, 255, cv::THRESH_BINARY);
    // dilate the thresholded image
    cv::Mat dilated;
    cv::Mat kernel = cv::Mat::ones(dilate, dilate, CV_8U);
    cv::dilate(thresholded, dilated, kernel);
    // find contours in the dilated image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // check if any contours are found
    if (contours.empty()) {
        previous_image = processed_image;
        return std::nullopt;
    }
    // check if any contours are above the threshold area
    std::vector<cv::Rect> bounding_rects;
    for (std::vector<cv::Point> contour : contours) {
        double area = cv::contourArea(contour);
        if (area < threshold_area) {
            continue;
        }
        // get bounding rectangles
        cv::Rect bounding_rect = cv::boundingRect(contour);
        bounding_rects.push_back(bounding_rect);
    }
    // update detections vector if any bounding rectangles are found
    if (!bounding_rects.empty()) {
        //create labels as long as the number of bounding rectangles
        std::vector<std::string> labels(bounding_rects.size(), "motion");
        std::vector<float> confidences(bounding_rects.size(), 1.0);
        // calculate area of bounding rectangles
        std::vector<int> detection_areas;
        for (cv::Rect bounding_rect : bounding_rects) {
            detection_areas.push_back(bounding_rect.area());
        }
        // create metadata
        std::map<std::string, std::string> meta_data;
        meta_data["type"] = std::string("motion");
        Detection detection(
            now(),
            frame_event,
            labels,
            confidences,
            bounding_rects,
            detection_areas,
            meta_data
        );
        detections.push_back(detection);
        // check if motion is detected for activation frames
        if (motion_frames < activation_frames) {
            previous_image = processed_image;
            motion_frames++;
            return std::nullopt;
        }else{
            // added detections to event queue
            DetectionEvent detection_event(
                now(),
                detections
            );
            detections.clear();
            motion_frames = 0;
            previous_image = processed_image;
            return detection_event;
        }
    }else{
        motion_frames = 0;
        detections.clear();
        previous_image = processed_image;
        return std::nullopt;
    }
}

// BirdDetectorYolov5 implementation

BirdDetectorYolov5::BirdDetectorYolov5(
    std::shared_ptr<ImageStore> image_store,
    const std::string& model_path,
    cv::Size image_size,
    float confidence_threshold,
    float iou_threshold,
    std::chrono::seconds max_delay,
    int threshold_area
) : Detector({EventType::NEW_FRAME}, image_store),
    model_path(model_path),
    image_size(image_size),
    confidence_threshold(confidence_threshold),
    iou_threshold(iou_threshold),
    max_delay(max_delay),
    threshold_area(threshold_area),
    model_loaded(false) {
    
    load_model();
}

BirdDetectorYolov5::~BirdDetectorYolov5() = default;

void BirdDetectorYolov5::load_model() {
    try {
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            throw std::runtime_error("Failed to load ONNX model from: " + model_path);
        }
        
        // Use GPU if available
        if (cv::dnn::DNN_BACKEND_CUDA) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        model_loaded = true;
        std::cout << "YOLOv5 model loaded successfully from: " << model_path << std::endl;
        
        // Try to load class names from model metadata
        load_class_names_from_model();
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLOv5 model: " << e.what() << std::endl;
        model_loaded = false;
    }
}

std::vector<std::string> BirdDetectorYolov5::get_default_class_names() {
    // Default class names for bird detection
    return {
        "bird", "pigeon", "crow", "sparrow", "robin", "eagle", "hawk", "seagull",
        "duck", "goose", "swan", "heron", "owl", "woodpecker", "cardinal", "bluejay"
    };
}

void BirdDetectorYolov5::load_class_names_from_model() {
    // First, set default class names as fallback
    class_names = get_default_class_names();
    
    try {
        // Try to load class names from a text file alongside the model
        std::string txt_path = model_path;
        
        // Replace .onnx extension with _classes.txt
        size_t dot_pos = txt_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            txt_path = txt_path.substr(0, dot_pos) + "_classes.txt";
        } else {
            txt_path += "_classes.txt";
        }
        
        std::ifstream txt_file(txt_path);
        if (!txt_file.is_open()) {
            std::cout << "No class names file found at: " << txt_path << std::endl;
            std::cout << "Using default class names." << std::endl;
            return;
        }
        
        // Read class names line by line
        std::vector<std::string> extracted_names;
        std::string line;
        
        while (std::getline(txt_file, line)) {
            // Remove trailing whitespace and carriage returns
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            
            if (!line.empty()) {
                extracted_names.push_back(line);
            }
        }
        
        txt_file.close();
        
        if (!extracted_names.empty()) {
            class_names = extracted_names;
            for (size_t i = 0; i < class_names.size(); ++i) {
                std::cout << "  " << i << ": " << class_names[i] << std::endl;
            }
            std::cout.flush(); // Ensure output is visible
        } else {
            std::cout << "Class names file is empty, using default class names." << std::endl;
            std::cout.flush();
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error loading class names from text file: " << e.what() << std::endl;
        std::cout << "Using default class names." << std::endl;
    }
}

cv::Mat BirdDetectorYolov5::preprocess_image(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, image_size);
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImage(rgb, blob, 1.0/255.0, image_size, cv::Scalar(0,0,0), true, false, CV_32F);
    
    return blob;
}

std::vector<cv::Rect> BirdDetectorYolov5::extract_boxes(const std::vector<cv::Mat>& outputs, 
                                                        const cv::Size& original_size,
                                                        std::vector<float>& confidences, 
                                                        std::vector<int>& class_ids) {
    std::vector<cv::Rect> boxes;
    confidences.clear();
    class_ids.clear();
    
    for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
        const auto& output = outputs[out_idx];
        
        const float* data = (float*)output.data;
        
        // Handle 3D tensor: [batch_size, num_detections, features]
        int batch_size = 1;
        int num_detections = output.size[1];
        int num_features = output.size[2]; 
        
        for (int i = 0; i < num_detections; ++i) {
            // Data layout: [x_center, y_center, width, height, objectness, class0, class1, class2, class3]
            int base_idx = i * num_features;
            
            float center_x = data[base_idx + 0];
            float center_y = data[base_idx + 1]; 
            float width = data[base_idx + 2];
            float height = data[base_idx + 3];
            float objectness = data[base_idx + 4];
            
            if (objectness >= confidence_threshold) {
                // Find the class with maximum confidence
                float max_class_score = 0.0f;
                int best_class_id = 0;
                for (int c = 0; c < num_features - 5; ++c) {  // Skip first 5 elements (bbox + objectness)
                    float class_score = data[base_idx + 5 + c];
                    if (class_score > max_class_score) {
                        max_class_score = class_score;
                        best_class_id = c;
                    }
                }
                
                // Final confidence is objectness * class_confidence
                float final_confidence = objectness * max_class_score;
                
                if (final_confidence >= confidence_threshold) {
                    // Convert from center coordinates to corner coordinates
                    // Note: YOLOv5 outputs are typically normalized to image size already
                    int left = static_cast<int>((center_x - width / 2) * original_size.width / image_size.width);
                    int top = static_cast<int>((center_y - height / 2) * original_size.height / image_size.height);
                    int w = static_cast<int>(width * original_size.width / image_size.width);
                    int h = static_cast<int>(height * original_size.height / image_size.height);
                    
                    cv::Rect box(left, top, w, h);
                    int area = box.area();
                    
                    if (area > threshold_area) {
                        boxes.push_back(box);
                        confidences.push_back(final_confidence);
                        class_ids.push_back(best_class_id);
                    }
                }
            }
        }
    }
    
    return boxes;
}

void BirdDetectorYolov5::apply_nms(std::vector<cv::Rect>& boxes, 
                                  std::vector<float>& confidences, 
                                  std::vector<int>& class_ids) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold, indices);
    
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_confidences;
    std::vector<int> nms_class_ids;
    
    for (int idx : indices) {
        nms_boxes.push_back(boxes[idx]);
        nms_confidences.push_back(confidences[idx]);
        nms_class_ids.push_back(class_ids[idx]);
    }
    
    boxes = nms_boxes;
    confidences = nms_confidences;
    class_ids = nms_class_ids;
}

std::optional<DetectionEvent> BirdDetectorYolov5::detect(std::shared_ptr<FrameEvent> frame_event) {
    if (!model_loaded) {
        std::cerr << "YOLOv5 model not loaded" << std::endl;
        return std::nullopt;
    }
    
    // Check if frame is delayed
    if (frame_event->get_timestamp() < std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()) - max_delay) {
        return std::nullopt;
    }
    
    // Check whether image exists
    if (!image_store->get(frame_event->get_timestamp()).has_value()) {
        return std::nullopt;
    }
    
    cv::Mat image = image_store->get(frame_event->get_timestamp()).value();
    cv::Size original_size = image.size();
    
    // Preprocess image
    cv::Mat blob = preprocess_image(image);
    
    // Set input to the network
    net.setInput(blob);
    
    // Run inference
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    // Extract detections
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes = extract_boxes(outputs, original_size, confidences, class_ids);
    
    if (boxes.empty()) {
        return std::nullopt;
    }
    
    // Apply Non-Maximum Suppression
    apply_nms(boxes, confidences, class_ids);
    
    if (boxes.empty()) {
        return std::nullopt;
    }
    
    // Create labels from class IDs
    std::vector<std::string> labels;
    for (int class_id : class_ids) {
        if (class_id < static_cast<int>(class_names.size())) {
            labels.push_back(class_names[class_id]);
        } else {
            labels.push_back("bird"); // fallback to generic "bird" label
        }
    }
    
    // Calculate detection areas
    std::vector<int> detection_areas;
    for (const cv::Rect& box : boxes) {
        detection_areas.push_back(box.area());
    }
    
    // Create metadata
    std::map<std::string, std::string> meta_data;
    meta_data["type"] = "bird_detected";
    
    // Create detection
    Detection detection(
        now(),
        frame_event,
        labels,
        confidences,
        boxes,
        detection_areas,
        meta_data
    );
    
    // Create detection event
    DetectionEvent detection_event(
        now(),
        {detection}
    );
    
    return detection_event;
}

// ClassStatistics implementation
ClassStatistics::ClassStatistics() : total_confidence(0.0f), detection_count(0) {}

float ClassStatistics::get_average_confidence() const {
    return detection_count > 0 ? total_confidence / detection_count : 0.0f;
}

float ClassStatistics::get_weighted_score() const {
    return total_confidence; // Sum of all confidences for this class
}

// Track implementation
Track::Track(int id, const cv::Rect& bbox, Timestamp timestamp) 
    : track_id(id), last_bbox(bbox), frames_since_last_detection(0), 
      total_detections_in_track(0), last_detection_time(timestamp) {
    last_center = cv::Point2f(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
    trajectory.push_back(last_center);
}

std::string Track::get_most_likely_class() const {
    if (class_votes.empty()) return "unknown";
    
    std::string best_class;
    float best_score = -1.0f;
    for (const auto& [class_name, stats] : class_votes) {
        if (stats.get_weighted_score() > best_score) {
            best_score = stats.get_weighted_score();
            best_class = class_name;
        }
    }
    return best_class;
}

bool Track::has_reached_consensus(int minimum_detections) const {
    return total_detections_in_track >= minimum_detections;
}

float Track::get_mean_confidence_for_consensus_class() const {
    std::string consensus_class = get_most_likely_class();
    auto it = class_votes.find(consensus_class);
    if (it != class_votes.end()) {
        return it->second.get_average_confidence();
    }
    return 0.0f;
}

// ObjectTracker implementation
ObjectTracker::ObjectTracker(
    float iou_threshold,
    int max_frames_without_detection,
    float max_path_length_threshold
) : iou_threshold(iou_threshold),
    max_frames_without_detection(max_frames_without_detection),
    max_path_length_threshold(max_path_length_threshold),
    next_track_id(0) {}

void ObjectTracker::update_tracks(const std::vector<Detection>& detections, Timestamp current_time) {
    // Increment frame count for all tracks
    increment_frames_without_detection();
    
    // Associate detections with existing tracks
    associate_detections_to_tracks(detections, current_time);
    
    // Prune old or invalid tracks
    prune_tracks();
}

void ObjectTracker::associate_detections_to_tracks(
    const std::vector<Detection>& detections, 
    Timestamp current_time
) {
    std::vector<bool> detection_associated(detections.size(), false);
    
    // Extract bounding boxes from detections
    std::vector<cv::Rect> detection_boxes;
    std::vector<std::string> detection_labels;
    std::vector<float> detection_confidences;
    std::vector<size_t> detection_indices;
    
    for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
        const auto& detection = detections[det_idx];
        auto bboxes = detection.get_bounding_boxes();
        auto labels = detection.get_labels();
        auto confidences = detection.get_confidences();
        
        if (bboxes.has_value() && labels.has_value() && confidences.has_value()) {
            const auto& bbox_vec = bboxes.value();
            const auto& label_vec = labels.value();
            const auto& conf_vec = confidences.value();
            
            for (size_t i = 0; i < bbox_vec.size(); ++i) {
                detection_boxes.push_back(bbox_vec[i]);
                detection_labels.push_back(i < label_vec.size() ? label_vec[i] : "unknown");
                detection_confidences.push_back(i < conf_vec.size() ? conf_vec[i] : 0.0f);
                detection_indices.push_back(det_idx);
            }
        }
    }
    
    // Associate each detection box with tracks
    for (size_t box_idx = 0; box_idx < detection_boxes.size(); ++box_idx) {
        const cv::Rect& det_box = detection_boxes[box_idx];
        cv::Point2f det_center(det_box.x + det_box.width/2.0f, det_box.y + det_box.height/2.0f);
        
        float best_iou = 0.0f;
        int best_track_idx = -1;
        
        // Find best matching track
        for (size_t track_idx = 0; track_idx < active_tracks.size(); ++track_idx) {
            Track& track = active_tracks[track_idx];
            float iou = calculate_iou(track.last_bbox, det_box);
            
            if (iou > iou_threshold && iou > best_iou) {
                // Check if path length would exceed threshold
                if (!should_drop_track_for_path_length(track, det_center)) {
                    best_iou = iou;
                    best_track_idx = track_idx;
                }
            }
        }
        
        // Update best matching track
        if (best_track_idx >= 0) {
            Track& track = active_tracks[best_track_idx];
            
            // Update track properties
            track.last_bbox = det_box;
            track.last_center = det_center;
            track.trajectory.push_back(det_center);
            track.frames_since_last_detection = 0;
            track.total_detections_in_track++;
            track.last_detection_time = current_time;
            
            // Update class statistics
            std::string class_name = detection_labels[box_idx];
            float confidence = detection_confidences[box_idx];
            
            if (track.class_votes.find(class_name) == track.class_votes.end()) {
                track.class_votes[class_name] = ClassStatistics();
            }
            track.class_votes[class_name].total_confidence += confidence;
            track.class_votes[class_name].detection_count++;
            
            // Mark detection as associated
            size_t orig_det_idx = detection_indices[box_idx];
            detection_associated[orig_det_idx] = true;
        }
    }
    
    // Create new tracks for unassociated detections
    std::vector<Detection> unassociated_detections;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_associated[i]) {
            unassociated_detections.push_back(detections[i]);
        }
    }
    
    create_new_tracks(unassociated_detections, current_time);
}

void ObjectTracker::create_new_tracks(
    const std::vector<Detection>& unassociated_detections,
    Timestamp current_time
) {
    for (const auto& detection : unassociated_detections) {
        auto bboxes = detection.get_bounding_boxes();
        auto labels = detection.get_labels();
        auto confidences = detection.get_confidences();
        
        if (bboxes.has_value() && labels.has_value() && confidences.has_value()) {
            const auto& bbox_vec = bboxes.value();
            const auto& label_vec = labels.value();
            const auto& conf_vec = confidences.value();
            
            for (size_t i = 0; i < bbox_vec.size(); ++i) {
                // Create new track
                Track new_track(next_track_id++, bbox_vec[i], current_time);
                new_track.total_detections_in_track = 1;
                
                // Add class vote
                std::string class_name = i < label_vec.size() ? label_vec[i] : "unknown";
                float confidence = i < conf_vec.size() ? conf_vec[i] : 0.0f;
                
                new_track.class_votes[class_name] = ClassStatistics();
                new_track.class_votes[class_name].total_confidence = confidence;
                new_track.class_votes[class_name].detection_count = 1;
                
                active_tracks.push_back(new_track);
            }
        }
    }
}

bool ObjectTracker::should_drop_track_for_path_length(Track& track, const cv::Point2f& new_center) {
    float distance = calculate_distance(track.last_center, new_center);
    
    if (distance > max_path_length_threshold) {
        // Reset track statistics but keep the track with new position
        track.class_votes.clear();
        track.total_detections_in_track = 0;
        track.trajectory.clear();
        track.trajectory.push_back(new_center);
        return false; // Don't drop, but reset
    }
    
    return false;
}

void ObjectTracker::increment_frames_without_detection() {
    for (auto& track : active_tracks) {
        track.frames_since_last_detection++;
    }
}

void ObjectTracker::prune_tracks() {
    active_tracks.erase(
        std::remove_if(active_tracks.begin(), active_tracks.end(),
            [this](const Track& track) {
                return track.frames_since_last_detection > max_frames_without_detection;
            }),
        active_tracks.end()
    );
}

std::vector<Track> ObjectTracker::get_tracks_with_consensus(int minimum_detections) const {
    std::vector<Track> consensus_tracks;
    for (const auto& track : active_tracks) {
        if (track.has_reached_consensus(minimum_detections)) {
            consensus_tracks.push_back(track);
        }
    }
    return consensus_tracks;
}

std::vector<Track> ObjectTracker::get_all_active_tracks() const {
    return active_tracks;
}

void ObjectTracker::remove_track(int track_id) {
    active_tracks.erase(
        std::remove_if(active_tracks.begin(), active_tracks.end(),
            [track_id](const Track& track) {
                return track.track_id == track_id;
            }),
        active_tracks.end()
    );
}

float ObjectTracker::calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

float ObjectTracker::calculate_iou(const cv::Rect& box1, const cv::Rect& box2) const {
    cv::Rect intersection = box1 & box2;
    float intersection_area = intersection.area();
    float union_area = box1.area() + box2.area() - intersection_area;
    
    return union_area > 0 ? intersection_area / union_area : 0.0f;
}

// SingleClassSequenceDetector implementation
SingleClassSequenceDetector::SingleClassSequenceDetector(
    std::shared_ptr<Detector> base_detector,
    std::shared_ptr<ImageStore> image_store,
    int minimum_number_detections,
    float iou_threshold,
    int max_frames_without_detection,
    float max_path_length_threshold
) : Detector({EventType::NEW_FRAME}, image_store),
    base_detector(base_detector),
    minimum_number_detections(minimum_number_detections) {
    
    tracker = std::make_unique<ObjectTracker>(
        iou_threshold, 
        max_frames_without_detection, 
        max_path_length_threshold
    );
}

SingleClassSequenceDetector::~SingleClassSequenceDetector() = default;

std::optional<DetectionEvent> SingleClassSequenceDetector::detect(
    std::shared_ptr<FrameEvent> frame_event
) {
    // Delegate detection to base detector
    std::optional<DetectionEvent> base_result = base_detector->detect(frame_event);
    
    if (base_result.has_value()) {
        std::vector<Detection> detections = base_result.value().get_detections();
        
        // Update tracker with new detections
        tracker->update_tracks(detections, frame_event->get_timestamp());
        
        // Store detections by track ID for consensus processing
        for (const auto& detection : detections) {
            // In a more sophisticated implementation, you would map detections back to track IDs
            // For now, we'll process consensus tracks independently
        }
    }
    
    // Check for tracks that have reached consensus
    std::vector<Track> consensus_tracks = tracker->get_tracks_with_consensus(minimum_number_detections);
    
    if (!consensus_tracks.empty()) {
        // Process consensus tracks and create detection events
        std::vector<Detection> all_consensus_detections;
        
        for (const auto& track : consensus_tracks) {
            // Create consensus detection for this track
            auto meta_data = create_consensus_metadata(track);
            
            Detection consensus_detection(
                frame_event->get_timestamp(),
                frame_event,
                std::vector<std::string>{track.get_most_likely_class()},
                std::vector<float>{track.get_mean_confidence_for_consensus_class()},
                std::vector<cv::Rect>{track.last_bbox},
                std::vector<int>{track.last_bbox.area()},
                meta_data
            );
            
            all_consensus_detections.push_back(consensus_detection);
        }
        
        if (!all_consensus_detections.empty()) {
            // Clean up processed tracks (remove them from active tracking)
            cleanup_completed_tracks(consensus_tracks);
            
            return DetectionEvent(
                frame_event->get_timestamp(),
                all_consensus_detections
            );
        }
    }
    
    return std::nullopt;
}

void SingleClassSequenceDetector::cleanup_completed_tracks(const std::vector<Track>& consensus_tracks) {
    // Remove tracks that have reached consensus from active tracking
    for (const auto& consensus_track : consensus_tracks) {
        tracker->remove_track(consensus_track.track_id);
    }
}

std::optional<std::map<std::string, std::string>> 
SingleClassSequenceDetector::create_consensus_metadata(const Track& track) const {
    std::map<std::string, std::string> meta;
    meta["detector_type"] = "SingleClassSequenceDetector";
    meta["track_id"] = std::to_string(track.track_id);
    meta["most_likely_object"] = track.get_most_likely_class();
    meta["mean_confidence"] = std::to_string(track.get_mean_confidence_for_consensus_class());
    meta["total_detections_in_track"] = std::to_string(track.total_detections_in_track);
    meta["detection_type"] = "track_consensus";
    
    // Add class distribution information
    for (const auto& [class_name, stats] : track.class_votes) {
        meta["class_" + class_name + "_count"] = std::to_string(stats.detection_count);
        meta["class_" + class_name + "_total_conf"] = std::to_string(stats.total_confidence);
    }
    
    return meta;
}

// MotionActivatedDetector implementation
MotionActivatedDetector::MotionActivatedDetector(
    std::shared_ptr<Detector> motion_detector,
    std::shared_ptr<Detector> secondary_detector,
    std::shared_ptr<ImageStore> image_store,
    int slack_frames,
    int max_frame_history
) : Detector({EventType::NEW_FRAME}, image_store),
    motion_detector(motion_detector),
    secondary_detector(secondary_detector),
    slack_frames_remaining(0),
    slack_frames(slack_frames),
    max_frame_history(max_frame_history),
    motion_detected_recently(false) {}

MotionActivatedDetector::~MotionActivatedDetector() = default;

std::optional<DetectionEvent> MotionActivatedDetector::detect(
    std::shared_ptr<FrameEvent> frame_event
) {
    // Always update frame history to keep recent frames available
    update_frame_history(frame_event);
    
    // Check for motion detection
    std::optional<DetectionEvent> motion_result = motion_detector->detect(frame_event);
    
    if (motion_result.has_value()) {
        // Motion detected! Set up for secondary detection
        motion_detected_recently = true;
        slack_frames_remaining = slack_frames;
        
        // Process all accumulated frames through secondary detector
        process_accumulated_frames();
        
        // Now process the current frame through secondary detector
        std::optional<DetectionEvent> secondary_result = secondary_detector->detect(frame_event);
        
        if (secondary_result.has_value()) {
            // Enhance metadata to indicate this was motion-activated
            std::vector<Detection> enhanced_detections;
            
            for (const auto& detection : secondary_result.value().get_detections()) {
                auto meta_data = detection.get_meta_data();
                std::map<std::string, std::string> enhanced_meta;
                
                if (meta_data.has_value()) {
                    enhanced_meta = meta_data.value();
                }
                
                enhanced_meta["activation_type"] = "motion_triggered";
                enhanced_meta["motion_detector_triggered"] = "true";
                
                Detection enhanced_detection(
                    detection.get_timestamp(),
                    detection.get_frame_event(),
                    detection.get_labels(),
                    detection.get_confidences(),
                    detection.get_bounding_boxes(),
                    detection.get_detection_areas(),
                    enhanced_meta
                );
                
                enhanced_detections.push_back(enhanced_detection);
            }
            
            return DetectionEvent(
                frame_event->get_timestamp(),
                enhanced_detections
            );
        }
    } else if (motion_detected_recently && slack_frames_remaining > 0) {
        // No motion this frame, but we're still in the slack period
        slack_frames_remaining--;
        
        // Continue processing frames through secondary detector
        std::optional<DetectionEvent> secondary_result = secondary_detector->detect(frame_event);
        
        if (secondary_result.has_value()) {
            // Enhance metadata to indicate this was motion-activated (slack period)
            std::vector<Detection> enhanced_detections;
            
            for (const auto& detection : secondary_result.value().get_detections()) {
                auto meta_data = detection.get_meta_data();
                std::map<std::string, std::string> enhanced_meta;
                
                if (meta_data.has_value()) {
                    enhanced_meta = meta_data.value();
                }
                
                enhanced_meta["activation_type"] = "motion_triggered_slack";
                enhanced_meta["slack_frames_remaining"] = std::to_string(slack_frames_remaining);
                
                Detection enhanced_detection(
                    detection.get_timestamp(),
                    detection.get_frame_event(),
                    detection.get_labels(),
                    detection.get_confidences(),
                    detection.get_bounding_boxes(),
                    detection.get_detection_areas(),
                    enhanced_meta
                );
                
                enhanced_detections.push_back(enhanced_detection);
            }
            
            return DetectionEvent(
                frame_event->get_timestamp(),
                enhanced_detections
            );
        }
        
        // Reset motion state if slack period expired
        if (slack_frames_remaining == 0) {
            reset_motion_state();
        }
    }
    
    return std::nullopt;
}

void MotionActivatedDetector::update_frame_history(std::shared_ptr<FrameEvent> frame_event) {
    frame_history.push_back(frame_event);
    
    // Maintain maximum history size
    while (frame_history.size() > static_cast<size_t>(max_frame_history)) {
        frame_history.pop_front();
    }
}

void MotionActivatedDetector::process_accumulated_frames() {
    // Process all frames in history through secondary detector
    // This ensures the secondary detector has context from before motion was detected
    for (const auto& historical_frame : frame_history) {
        // Process but don't emit results from historical frames
        // This builds up internal state in the secondary detector
        secondary_detector->detect(historical_frame);
    }
}

void MotionActivatedDetector::reset_motion_state() {
    motion_detected_recently = false;
    slack_frames_remaining = 0;
}