#pragma once
#include "events.hpp"
#include <chrono>

// Helper functions for creating timestamps in tests
inline Timestamp test_now() {
    return now();
}

// Helper to create a timestamp from seconds offset from now
inline Timestamp test_timestamp_offset(int seconds_offset) {
    return now() + std::chrono::seconds(seconds_offset);
}

// Helper to create a timestamp from milliseconds offset from now  
inline Timestamp test_timestamp_offset_ms(int ms_offset) {
    return now() + std::chrono::milliseconds(ms_offset);
}
