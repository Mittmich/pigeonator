/*
    Libarary for handling events
*/

#include "events.hpp"

Event::Event(
    EventType type,
    Timestamp event_timestamp,
    std::optional<std::map<std::string, std::string>> meta_data)
    : type(type), event_timestamp(event_timestamp), meta_data(meta_data) {}

Event::~Event() = default;

Timestamp Event::get_timestamp()
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
    Timestamp event_timestamp,
    std::optional<std::map<std::string, std::string>> meta_data)
    : Event(EventType::NEW_FRAME, event_timestamp, meta_data) {}

FrameEvent::~FrameEvent() = default;

DetectionEvent::DetectionEvent(
    Timestamp event_timestamp,
    std::vector<Detection> detections,
    std::optional<std::map<std::string, std::string>> meta_data)
    : Event(EventType::DETECTION, event_timestamp, meta_data),
      detections(detections) {};

DetectionEvent::~DetectionEvent() = default;

std::vector<Detection> DetectionEvent::get_detections() const
{
    return detections;
}

Detection::Detection(
    Timestamp timestamp,
    std::shared_ptr<FrameEvent> frame_event,
    std::optional<std::vector<std::string>> labels,
    std::optional<std::vector<float>> confidences,
    std::optional<std::vector<cv::Rect>> bounding_boxes,
    std::optional<std::vector<int>> detection_areas,
    std::optional<std::map<std::string, std::string>> meta_data)
    : timestamp(timestamp),
      frame_event(frame_event),
      labels(labels),
      confidences(confidences),
      bounding_boxes(bounding_boxes),
      detection_areas(detection_areas),
      meta_data(meta_data) {};

Detection::~Detection() = default;


std::shared_ptr<FrameEvent> Detection::get_frame_event() const
{
    return frame_event;
}

std::optional<std::vector<std::string>> Detection::get_labels() const
{
    return labels;
}

std::optional<std::vector<float>> Detection::get_confidences() const
{
    return confidences;
}

std::optional<std::vector<cv::Rect>> Detection::get_bounding_boxes() const
{
    return bounding_boxes;
}

std::optional<std::vector<int>> Detection::get_detection_areas() const
{
    return detection_areas;
}

Timestamp Detection::get_timestamp() const
{
    return timestamp;
}

std::optional<std::map<std::string, std::string>> Detection::get_meta_data() const
{
    return meta_data;
}

