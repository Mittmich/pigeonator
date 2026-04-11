#pragma once

#include "events.hpp"
#include <string>
#include <set>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <vector>
#include <atomic>

/**
 * EventDispatcher - sends detection and effector events to a remote HTTP server.
 *
 * Implements Subscriber so it can be wired directly into VideoEventManager.
 * All HTTP calls happen in a dedicated background thread (non-blocking for callers).
 *
 * Endpoints mirrored from the Python version:
 *   POST {server}/detections/       — on DETECTION
 *   POST {server}/effectorAction/   — on EFFECTOR_ACTION
 *
 * Optional HTTP Basic Auth and SSL verification control.
 */
class EventDispatcher : public Subscriber {
public:
    EventDispatcher(
        std::string server_address,
        std::string user     = "",
        std::string password = ""
    );
    ~EventDispatcher() override;

    // Subscriber interface
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) override;
    void notify(std::shared_ptr<Event> event) override;

private:
    void worker_loop();
    void dispatch(std::shared_ptr<Event> event);
    void send_detection(std::shared_ptr<DetectionEvent> event);
    void send_effector_action(std::shared_ptr<EffectorActionEvent> event);

    std::string server_address;
    std::string user;
    std::string password;

    // Dispatch queue fed by notify(), drained by the worker thread
    std::queue<std::shared_ptr<Event>> pending;
    std::mutex pending_mutex;
    std::condition_variable pending_cv;

    std::thread worker;
    std::atomic<bool> running{false};

    // Not used for dispatching, kept to satisfy Subscriber interface
    std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue;
};
