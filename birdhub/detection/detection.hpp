#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <set>

// add detector base class, inheriting from Subscriber

class Detector : public Subscriber {
public:
    Detector();
    ~Detector();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) override;
    void notify(Event event) override;
    virtual void detect(cv::Mat frame) = 0;

protected:
    std::shared_ptr<std::queue<Event>> event_queue;
};

// create subclass of Detector for motion detection

class MotionDetector : public Detector {
public:
    MotionDetector();
    ~MotionDetector();
    void detect(cv::Mat frame) override;
};