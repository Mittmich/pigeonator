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

void Detector::set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) {
    this->event_write_queue = event_queue;
    this->queue_registered = true;
}

void Detector::notify(Event event) {
    // check if event is of type FrameEvent
    if (event.type != EventType::NEW_FRAME) {
        return;
    }
    // add event to read queue
    this->event_read_queue.push(static_cast<FrameEvent&>(event));
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
    this->queue_thread.join();
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
    FrameEvent event = this->event_read_queue.front();
    std::optional<DetectionEvent> detections = detect(event);
    // check if detections are present
    if (detections.has_value()) {
        // print detections with a nicely formatted timestamp
        std::cout << "Detections at " << detections.value().get_timestamp() << std::endl;
        this->event_write_queue->push(detections.value());
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

std::optional<DetectionEvent> MotionDetector::detect(FrameEvent &frame_event) {
    // check if frame is delayed
    if (std::chrono::system_clock::from_time_t(frame_event.get_timestamp()) < std::chrono::system_clock::now() - max_delay) {
        return std::nullopt;
    }
    // check whether image exists
    if (!image_store->get(frame_event.get_timestamp()).has_value()) {
        return std::nullopt;
    }
    cv::Mat current_image = image_store->get(frame_event.get_timestamp()).value();
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
            time(nullptr),
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
                time(nullptr),
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