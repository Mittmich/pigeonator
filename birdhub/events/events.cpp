/*
    Libarary for handling events
*/

#include "events.hpp"

Event::Event(
    EventType type,
    time_t event_timestamp,
    std::optional<std::map<std::string, std::string>> meta_data)
    : type(type), event_timestamp(event_timestamp), meta_data(meta_data) {}

Event::~Event() = default;

time_t Event::get_timestamp()
{
    return event_timestamp;
}

std::map<std::string, std::string> Event::get_meta_data()
{
    if (meta_data.has_value())
    {
        return meta_data.value();
    }
    return {};
}

FrameEvent::FrameEvent(
    time_t event_timestamp,
    std::optional<std::map<std::string, std::string>> meta_data)
    : Event(EventType::NEW_FRAME, event_timestamp, meta_data) {}

FrameEvent::~FrameEvent() = default;

DetectionEvent::DetectionEvent(
    time_t event_timestamp,
    FrameEvent frame_event,
    std::optional<std::vector<std::string>> labels,
    std::optional<std::vector<float>> confidences,
    std::optional<std::vector<cv::Rect>> bounding_boxes,
    std::optional<std::vector<int>> detection_areas,
    std::optional<std::map<std::string, std::string>> meta_data)
    : Event(EventType::DETECTION, event_timestamp, meta_data),
      frame_event(frame_event),
      labels(labels),
      confidences(confidences),
      bounding_boxes(bounding_boxes),
      detection_areas(detection_areas) {};

DetectionEvent::~DetectionEvent() = default;

FrameEvent DetectionEvent::get_frame_event()
{
    return frame_event;
}

std::optional<std::vector<std::string>> DetectionEvent::get_labels()
{
    return labels;
}

std::optional<std::vector<float>> DetectionEvent::get_confidences()
{
    return confidences;
}

std::optional<std::vector<cv::Rect>> DetectionEvent::get_bounding_boxes()
{
    return bounding_boxes;
}

std::optional<std::vector<int>> DetectionEvent::get_detection_areas()
{
    return detection_areas;
}

