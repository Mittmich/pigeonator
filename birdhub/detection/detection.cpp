#include "detection.hpp"

Detector::Detector(
    std::set<EventType> listening_events,
    ImageStore image_store
) : listening_events(listening_events),
    image_store(image_store) {};

Detector::~Detector() = default;

std::set<EventType> Detector::listening_to() {
    return listening_events;
}

void Detector::set_event_queue(std::shared_ptr<std::queue<Event>> event_queue) {
    this->event_write_queue = event_queue;
    this->queue_registered = true;
}

void Detector::notify(Event event) {
    // check if event is of type FrameEvent
    if (event.type != EventType::NEW_FRAME) {
        return;
    }
    // add event to read queue
    this->event_read_queue.push(static_cast<FrameEvent&>(event));
}

void Detector::start() {
    // check if event queue is registered
    if (!this->queue_registered) {
        throw std::runtime_error("Event queue not registered.");
    }
    // set running flag that is used to stop the thread
    this->running = true;
    // start thread
    this->queue_thread = std::thread(&Detector::_start, this);
}

void Detector::stop() {
    // set running flag to false to stop the thread
    this->running = false;
    // join the thread
    this->queue_thread.join();
}

void Detector::_start() {
    while (this->running) {
        poll_read_queue();
    }
}

void Detector::poll_read_queue() {
    // check if event queue is empty
    if (this->event_read_queue.empty()) {
        return;
    }
    // get event from queue
    FrameEvent event = this->event_read_queue.front();
    std::optional<std::vector<DetectionEvent>> detections = detect(event);
    // check if detections are present
    if (detections.has_value()) {
        // add detections to event queue
        for (DetectionEvent detection : detections.value()) {
            this->event_write_queue->push(detection);
        }
    }
    // remove event from read queue
    this->event_read_queue.pop();
}