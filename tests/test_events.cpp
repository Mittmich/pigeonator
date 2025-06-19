#include "doctest/doctest.h"
#include "video.hpp"
#include "events.hpp"
#include "timestamp_utils.hpp"

// test reading timestamp from event

TEST_CASE("Test reading timestamp from event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    Event event(EventType::NEW_FRAME, timestamp, std::nullopt);
    CHECK(event.get_timestamp() == timestamp);
}

// test reading metadata from event

TEST_CASE("Test reading metadata from event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    std::map<std::string, std::string> meta_data = {{"key1", "value1"}, {"key2", "value2"}};
    Event event(EventType::NEW_FRAME, timestamp, meta_data);
    CHECK(event.get_meta_data() == meta_data);
}

// test reading type from event

TEST_CASE("Test reading type from event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    Event event(EventType::NEW_FRAME, timestamp, std::nullopt);
    CHECK(event.type == EventType::NEW_FRAME);
}

// test reading timestamp from frame event

TEST_CASE("Test reading timestamp from frame event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    FrameEvent frame_event(timestamp, std::nullopt);
    CHECK(frame_event.get_timestamp() == timestamp);
}

// test reading metadata from frame event

TEST_CASE("Test reading metadata from frame event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    std::map<std::string, std::string> meta_data = {{"key1", "value1"}, {"key2", "value2"}};
    FrameEvent frame_event(timestamp, meta_data);
    CHECK(frame_event.get_meta_data() == meta_data);
}

// test reading type from frame event

TEST_CASE("Test reading type from frame event") {
    Timestamp timestamp = test_timestamp_offset_ms(123456);
    FrameEvent frame_event(timestamp, std::nullopt);
    CHECK(frame_event.type == EventType::NEW_FRAME);
}