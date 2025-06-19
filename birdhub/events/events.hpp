/*
    Libarary for orchestrating events
*/

#include <set>
#include <map>
#include <queue>
#include <optional>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#ifndef BIRDHUB_EVENTS_EVENTS_HPP
#define BIRDHUB_EVENTS_EVENTS_HPP

// Define timestamp type with millisecond precision
using Timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

// Helper function to get current timestamp
inline Timestamp now() {
    return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
}

// create enum for event types

enum class EventType {
    NEW_FRAME,
    DETECTION,
    EFFECTOR_ACTION
};

// event class that holds metadata

class Event {
public:
    Event(EventType type,
         Timestamp event_timestamp,
         std::optional<std::map<std::string,
         std::string>> meta_data);
    ~Event();
    EventType type;
    Timestamp get_timestamp();
    std::map<std::string, std::string> get_meta_data();
protected:
    Timestamp event_timestamp;
    std::optional<std::map<std::string, std::string>> meta_data;  
};

// create a subclass of event for frame events

class FrameEvent : public Event {
public:
    FrameEvent(
         Timestamp event_timestamp,
         std::optional<std::map<std::string,
         std::string>> meta_data);
    ~FrameEvent();
    EventType type = EventType::NEW_FRAME;
};


// create base class for subscribers

class Subscriber {
public:
    virtual ~Subscriber() = default;
    virtual std::set<EventType> listening_to() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) = 0;
    virtual void notify(std::shared_ptr<Event> event) = 0;
};


class Detection {
    public:
    Detection(
         Timestamp timestamp,
         std::shared_ptr<FrameEvent> frame_event,
         std::optional<std::vector<std::string>> labels = std::nullopt,
         std::optional<std::vector<float>> confidences = std::nullopt,
         std::optional<std::vector<cv::Rect>> bounding_boxes = std::nullopt,
         std::optional<std::vector<int>> detection_areas = std::nullopt,
         std::optional<std::map<std::string, std::string>> meta_data = std::nullopt);
    ~Detection();
    Timestamp get_timestamp();
    std::shared_ptr<FrameEvent> get_frame_event();
    std::optional<std::vector<std::string>> get_labels();
    std::optional<std::vector<float>> get_confidences();
    std::optional<std::vector<cv::Rect>> get_bounding_boxes();
    std::optional<std::vector<int>> get_detection_areas();
    std::optional<std::map<std::string, std::string>> get_meta_data();
private:
    Timestamp timestamp;
    std::shared_ptr<FrameEvent> frame_event;
    std::optional<std::vector<std::string>> labels;
    std::optional<std::vector<float>> confidences;
    std::optional<std::vector<cv::Rect>> bounding_boxes;
    std::optional<std::vector<int>> detection_areas;
    std::optional<std::map<std::string, std::string>> meta_data;
};

// create a subclass of event for detection events

class DetectionEvent : public Event {
public:
    DetectionEvent(
         Timestamp event_timestamp,
         std::vector<Detection> detections,
         std::optional<std::map<std::string, std::string>> meta_data = std::nullopt);
    ~DetectionEvent();
    EventType type = EventType::DETECTION;
    std::vector<Detection> get_detections();
private:
    std::vector<Detection> detections;
};

#endif