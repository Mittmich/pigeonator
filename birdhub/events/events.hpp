/*
    Libarary for orchestrating events
*/

#include <set>
#include <map>
#include <queue>
#include <optional>
#include <ctime>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#ifndef BIRDHUB_EVENTS_EVENTS_HPP
#define BIRDHUB_EVENTS_EVENTS_HPP

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
         time_t event_timestamp,
         std::optional<std::map<std::string,
         std::string>> meta_data);
    ~Event();
    EventType type;
    time_t get_timestamp();
    std::map<std::string, std::string> get_meta_data();
protected:
    time_t event_timestamp;
    std::optional<std::map<std::string, std::string>> meta_data;  
};

// create a subclass of event for frame events

class FrameEvent : public Event {
public:
    FrameEvent(
         time_t event_timestamp,
         std::optional<std::map<std::string,
         std::string>> meta_data);
    ~FrameEvent();
    EventType type = EventType::NEW_FRAME;
};

// create a subclass of event for detection events

class DetectionEvent : public Event {
public:
    DetectionEvent(
         time_t event_timestamp,
         FrameEvent frame_event,
         std::optional<std::vector<std::string>> labels = std::nullopt,
         std::optional<std::vector<float>> confidences = std::nullopt,
         std::optional<std::vector<cv::Rect>> bounding_boxes = std::nullopt,
         std::optional<std::vector<int>> detection_areas = std::nullopt,
         std::optional<std::map<std::string, std::string>> meta_data = std::nullopt);
    ~DetectionEvent();
    EventType type = EventType::DETECTION;
    FrameEvent get_frame_event();
    std::optional<std::vector<std::string>> get_labels();
    std::optional<std::vector<float>> get_confidences();
    std::optional<std::vector<cv::Rect>> get_bounding_boxes();
    std::optional<std::vector<int>> get_detection_areas();
private:
    FrameEvent frame_event;
    std::optional<std::vector<std::string>> labels;
    std::optional<std::vector<float>> confidences;
    std::optional<std::vector<cv::Rect>> bounding_boxes;
    std::optional<std::vector<int>> detection_areas;
};

// create base class for subscribers

class Subscriber {
public:
    virtual ~Subscriber() = default;
    virtual std::set<EventType> listening_to() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) = 0;
    virtual void notify(Event event) = 0;
};


#endif