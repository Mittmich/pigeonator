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

time_t Event::get_timestamp() {
    return event_timestamp;
}

std::map<std::string, std::string> Event::get_meta_data() {
    if (meta_data.has_value()) {
        return meta_data.value();
    }
    return {};
}

FrameEvent::FrameEvent(
        time_t event_timestamp,
        std::optional<std::map<std::string, std::string>> meta_data)
    : Event(EventType::NEW_FRAME, event_timestamp, meta_data) {}

FrameEvent::~FrameEvent() = default;

std::map<std::string, std::string> FrameEvent::get_meta_data() {
    return {};
}