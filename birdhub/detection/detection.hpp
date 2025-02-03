#include "events.hpp"
#include <set>

// add detector base class, inheriting from Subscriber

class Detector : public Subscriber {
public:
    Detector();
    ~Detector();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::queue<Event> *event_queue) override;
    void notify(Event event) override;
    virtual void detect(cv::Mat frame) = 0;

protected:
    std::queue<Event> *event_queue;
};