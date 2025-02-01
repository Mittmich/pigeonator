/*
    Libarary for orchestrating events
*/

#include "video.hpp"
#include <set>
#include <map>
#include <queue>
#include <optional>

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
private:
    time_t event_timestamp;
    std::optional<std::map<std::string, std::string>> meta_data;  
};

#endif