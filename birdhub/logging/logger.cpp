#include "logger.hpp"

#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cstdlib>

// ────────────────────────────────────────────────────────────────────────────

EventLogger& EventLogger::instance() {
    static EventLogger inst;
    return inst;
}

EventLogger::EventLogger() {
    const char* env = std::getenv("PGN_LOGDIR");
    std::string log_dir = env ? env : "logs";
    std::filesystem::create_directories(log_dir);
    log_path_ = std::filesystem::path(log_dir) / "birdhub.log";
    open_log_file();
}

EventLogger::~EventLogger() {
    if (file_.is_open()) {
        file_.close();
    }
}

void EventLogger::open_log_file() {
    file_.open(log_path_, std::ios::app);
}

void EventLogger::rotate_if_needed() {
    if (!std::filesystem::exists(log_path_)) return;
    if (std::filesystem::file_size(log_path_) < MAX_BYTES) return;

    file_.close();

    // Shift backups: .10 removed, .9→.10, ..., .1→.2, .log→.log.1
    for (int i = MAX_BACKUPS; i >= 1; --i) {
        auto dst = std::filesystem::path(log_path_.string() + "." + std::to_string(i));
        auto src = (i == 1)
                   ? log_path_
                   : std::filesystem::path(log_path_.string() + "." + std::to_string(i - 1));
        if (std::filesystem::exists(dst)) std::filesystem::remove(dst);
        if (std::filesystem::exists(src)) std::filesystem::rename(src, dst);
    }

    open_log_file();
}

std::string EventLogger::current_timestamp() const {
    auto now_tp = std::chrono::system_clock::now();
    auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now_tp.time_since_epoch()) % 1000;
    std::time_t t = std::chrono::system_clock::to_time_t(now_tp);
    std::tm   tm  = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

void EventLogger::log_event(const std::string& level,
                             const std::string& event_type,
                             const std::string& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    rotate_if_needed();

    std::string line = current_timestamp() + "\t" + level + "\t" + event_type
                     + "\t" + info + "\n";

    std::cout << line;

    if (file_.is_open()) {
        file_ << line;
        file_.flush();
    }
}

// ── free function ─────────────────────────────────────────────────────────────

void log_event(const std::string& level,
               const std::string& event_type,
               const std::string& info) {
    EventLogger::instance().log_event(level, event_type, info);
}
