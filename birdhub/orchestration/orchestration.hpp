#include "video.hpp"
#include "events.hpp"
#include <set>
#include <map>
#include <queue>
#include <optional>
#include <memory>
#include <signal.h>
#include <atomic>

#ifndef BIRDHUB_ORCHESTRATION_HPP
#define BIRDHUB_ORCHESTRATION_HPP

// Global signal flag for graceful shutdown
extern std::atomic<bool> g_shutdown_requested;

class Mediator {
public:
    virtual ~Mediator() = default;
    virtual void add_subscriber(std::shared_ptr<Subscriber> subscriber) = 0;
    virtual void run() = 0;
    virtual void stop() = 0;
};

class VideoEventManager : public Mediator {
public:
    // stream is added to the constructor
    // since the event manager needs to subscribe to the stream
    VideoEventManager(Stream &stream);
    ~VideoEventManager();
    void add_subscriber(std::shared_ptr<Subscriber> subscriber) override;
    void run() override;
    void stop() override;
    
    // Signal handling
    static void setup_signal_handlers();
    static void signal_handler(int signal);
    
private:
    Stream &stream;
    std::vector<std::shared_ptr<Subscriber>> subscribers;
    std::shared_ptr<std::queue<std::shared_ptr<Event>>>  event_queue;
    std::shared_ptr<std::queue<std::shared_ptr<FrameEvent>>> frame_queue;
    bool running = false;
};

#endif