#include "detection.hpp"
#include <string>
#include <iostream>

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
    
    // Initialize class names for bird detection
    // These would typically be loaded from the model metadata or a separate file
    class_names = {
        "bird", "pigeon", "crow", "sparrow", "robin", "eagle", "hawk", "seagull",
        "duck", "goose", "swan", "heron", "owl", "woodpecker", "cardinal", "bluejay"
    };
    
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
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLOv5 model: " << e.what() << std::endl;
        model_loaded = false;
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
    
    for (const auto& output : outputs) {
        const float* data = (float*)output.data;
        
        for (int i = 0; i < output.rows; ++i) {
            float confidence = data[4];
            if (confidence >= confidence_threshold) {
                // Extract class scores
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point class_id_point;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
                
                if (max_class_score > confidence_threshold) {
                    // Extract bounding box
                    float center_x = data[0];
                    float center_y = data[1];
                    float width = data[2];
                    float height = data[3];
                    
                    int left = static_cast<int>((center_x - width / 2) * original_size.width / image_size.width);
                    int top = static_cast<int>((center_y - height / 2) * original_size.height / image_size.height);
                    int w = static_cast<int>(width * original_size.width / image_size.width);
                    int h = static_cast<int>(height * original_size.height / image_size.height);
                    
                    cv::Rect box(left, top, w, h);
                    int area = box.area();
                    
                    if (area > threshold_area) {
                        boxes.push_back(box);
                        confidences.push_back(static_cast<float>(max_class_score));
                        class_ids.push_back(class_id_point.x);
                    }
                }
            }
            data += output.cols;
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