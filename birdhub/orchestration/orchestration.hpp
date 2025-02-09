
#include "video.hpp"
#include "events.hpp"
#include <set>
#include <map>
#include <queue>
#include <optional>
#include <memory>

#ifndef BIRDHUB_ORCHESTRATION_HPP
#define BIRDHUB_ORCHESTRATION_HPP

class Mediator {
public:
    virtual ~Mediator() = default;
    virtual void add_subscriber(std::shared_ptr<Subscriber> subscriber) = 0;
    virtual void run() = 0;
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
    std::shared_ptr<std::queue<Event>> event_queue;
    std::shared_ptr<std::queue<FrameEvent>> frame_queue;
};

#endif