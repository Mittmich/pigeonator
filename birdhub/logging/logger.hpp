#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <filesystem>

// EventLogger: structured log to stdout + rotating log file.
// Format: ISO8601timestamp.ms\tLEVEL\tevent_type\tevent_information
// Log file: ${PGN_LOGDIR:-logs}/birdhub.log, max 2 GB, 10 backups.
class EventLogger {
public:
    static EventLogger& instance();

    void log_event(const std::string& level,
                   const std::string& event_type,
                   const std::string& info = "");

private:
    EventLogger();
    ~EventLogger();
    EventLogger(const EventLogger&) = delete;
    EventLogger& operator=(const EventLogger&) = delete;

    void open_log_file();
    void rotate_if_needed();
    std::string current_timestamp() const;

    std::filesystem::path log_path_;
    std::ofstream        file_;
    mutable std::mutex   mutex_;

    static constexpr std::uintmax_t MAX_BYTES   = 2'000'000'000ULL;
    static constexpr int            MAX_BACKUPS = 10;
};

// Convenience free function – mirrors Python's logger.log_event()
void log_event(const std::string& level,
               const std::string& event_type,
               const std::string& info = "");
