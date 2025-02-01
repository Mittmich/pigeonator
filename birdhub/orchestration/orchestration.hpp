
#include "video.hpp"
#include "events.hpp"
#include <set>
#include <map>
#include <queue>
#include <optional>

#ifndef BIRDHUB_ORCHESTRATION_HPP
#define BIRDHUB_ORCHESTRATION_HPP

// create base class for subscribers

class Subscriber {
public:
    virtual ~Subscriber() = default;
    virtual std::set<EventType> listening_to() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void set_event_queue(std::queue<Event> *event_queue) = 0;
    virtual void notify(Event event) = 0;
};

class Mediator {
public:
    virtual ~Mediator() = default;

    virtual void add_subscriber(std::shared_ptr<Subscriber> subscriber) = 0;
    virtual void run();
};

class VideoEventManager : public Mediator {
public:
    // stream is added to the constructor
    // since the event manager needs to subscribe to the stream
    VideoEventManager(Stream &stream);
    ~VideoEventManager();
    void add_subscriber(std::shared_ptr<Subscriber> subscriber) override;
    void run() override;
private:
    Stream &stream;
    std::vector<std::shared_ptr<Subscriber>> subscribers;
    std::queue<Event> event_queue;
    // add frame queue. TODO: Fefactor to use EVENT
    std::queue<FrameToken> frame_queue;
    void notify();
};

#endif