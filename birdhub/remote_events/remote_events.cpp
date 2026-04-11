#include "remote_events.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

using json = nlohmann::json;

// ─── helpers ─────────────────────────────────────────────────────────────────

static std::string timestamp_to_iso8601(const Timestamp& ts) {
    std::time_t t = std::chrono::system_clock::to_time_t(ts);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

// Parse "https://host:port" or "http://host:port" into usable parts — kept for
// future use extracting path prefixes if needed.

// Execute fn(client) using the scheme_host_port form of httplib::Client.
static bool with_client(const std::string& server_address,
                         const std::string& user,
                         const std::string& password,
                         std::function<bool(httplib::Client&)> fn)
{
    httplib::Client cli(server_address);

    if (!user.empty()) {
        cli.set_basic_auth(user, password);
    }
    cli.set_connection_timeout(5);
    cli.set_read_timeout(10);

    return fn(cli);
}

// ─── EventDispatcher ─────────────────────────────────────────────────────────

EventDispatcher::EventDispatcher(
    std::string server_address,
    std::string user,
    std::string password)
    : server_address(std::move(server_address)),
      user(std::move(user)),
      password(std::move(password)) {}

EventDispatcher::~EventDispatcher() {
    stop();
}

std::set<EventType> EventDispatcher::listening_to() {
    return {EventType::DETECTION, EventType::EFFECTOR_ACTION};
}

void EventDispatcher::set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> eq) {
    event_queue = eq;
}

void EventDispatcher::start() {
    if (running.exchange(true)) return;
    worker = std::thread(&EventDispatcher::worker_loop, this);
}

void EventDispatcher::stop() {
    if (!running.exchange(false)) return;
    pending_cv.notify_all();
    if (worker.joinable()) worker.join();
}

void EventDispatcher::notify(std::shared_ptr<Event> event) {
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        pending.push(event);
    }
    pending_cv.notify_one();
}

void EventDispatcher::worker_loop() {
    while (running) {
        std::unique_lock<std::mutex> lock(pending_mutex);
        pending_cv.wait(lock, [this] { return !pending.empty() || !running; });

        while (!pending.empty()) {
            auto event = pending.front();
            pending.pop();
            lock.unlock();
            dispatch(event);
            lock.lock();
        }
    }
}

void EventDispatcher::dispatch(std::shared_ptr<Event> event) {
    switch (event->type) {
        case EventType::DETECTION:
            send_detection(std::static_pointer_cast<DetectionEvent>(event));
            break;
        case EventType::EFFECTOR_ACTION:
            send_effector_action(std::static_pointer_cast<EffectorActionEvent>(event));
            break;
        default:
            break;
    }
}

void EventDispatcher::send_detection(std::shared_ptr<DetectionEvent> event) {
    auto detections = event->get_detections();
    if (detections.empty()) return;

    // Use the last detection for the summary (mirrors Python behaviour)
    const auto& last = detections.back();
    auto meta = last.get_meta_data(); // returns std::optional<map>

    std::string detected_class = "unknown";
    std::string detection_ts   = timestamp_to_iso8601(last.get_timestamp());
    float       mean_confidence = 0.0f;

    if (meta.has_value()) {
        auto& m = meta.value();
        if (m.count("most_likely_object"))  detected_class  = m.at("most_likely_object");
        if (m.count("mean_confidence"))     mean_confidence = std::stof(m.at("mean_confidence"));
    }

    json body = {
        {"detections", json::array({
            {
                {"detected_class",       detected_class},
                {"detection_timestamp",  detection_ts},
                {"confidence",          mean_confidence},
            }
        })}
    };

    std::string path = "/detections/";
    bool ok = with_client(server_address, user, password,
        [&](httplib::Client& cli) {
            auto res = cli.Post(path, body.dump(), "application/json");
            if (!res || res->status >= 400) {
                std::cerr << "[EventDispatcher] POST /detections/ failed: "
                          << (res ? std::to_string(res->status) : "no response") << "\n";
                return false;
            }
            return true;
        });
    (void)ok;
}

void EventDispatcher::send_effector_action(std::shared_ptr<EffectorActionEvent> event) {
    auto meta = event->get_meta_data(); // returns std::map<string,string> (not optional)

    json metadata_json = json::object();
    for (const auto& [k, v] : meta) {
        metadata_json[k] = v;
    }

    json body = {
        {"action",              event->get_action()},
        {"action_metadata",     metadata_json},
        {"action_timestamp",    timestamp_to_iso8601(event->get_timestamp())},
    };

    std::string path = "/effectorAction/";
    bool ok = with_client(server_address, user, password,
        [&](httplib::Client& cli) {
            auto res = cli.Post(path, body.dump(), "application/json");
            if (!res || res->status >= 400) {
                std::cerr << "[EventDispatcher] POST /effectorAction/ failed: "
                          << (res ? std::to_string(res->status) : "no response") << "\n";
                return false;
            }
            return true;
        });
    (void)ok;
}
