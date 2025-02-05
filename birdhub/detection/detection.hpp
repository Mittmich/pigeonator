#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <set>
#include <thread>
#include "mm.hpp"

// add detector base class, inheriting from Subscriber

class Detector : public Subscriber {
public:
    Detector(
        std::set<EventType> listening_events,
        ImageStore image_store
    );
    ~Detector();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) override;
    void notify(Event event) override;
    // TODO: think about why this needs to be a vector
    virtual std::optional<std::vector<DetectionEvent>> detect(FrameEvent &frame_event) = 0;

protected:
    std::shared_ptr<std::queue<Event>> event_write_queue;
    std::set<EventType> listening_events;
    ImageStore image_store;
    void _start();
    std::thread queue_thread;
    bool running = false;
    bool queue_registered = false;
    void poll_read_queue();
    std::queue<FrameEvent> event_read_queue;
};

// create subclass of Detector for motion detection

class MotionDetector : public Detector {
public:
    MotionDetector(
        int threshold,
        int blur,
        int dilate,
        int threshold_area,
        int activation_frames,
        time_t max_delay
    );
    ~MotionDetector();
    std::optional<std::vector<DetectionEvent>> detect(FrameEvent &frame_event) override;
private:
    int threshold;
    int blur;
    int dilate;
    int threshold_area;
    int activation_frames;
    time_t max_delay;
    cv::Mat previous_image;
    int motion_frames;
    std::vector<DetectionEvent> detections;
    cv::Mat preprocess_image(cv::Mat image);
};