/*
    Libarary for orchestrating events
*/

#include <set>
#include <map>
#include <queue>
#include <optional>
#include <ctime>
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
    std::map<std::string, std::string> get_meta_data();
    EventType type = EventType::NEW_FRAME;
};

#endif