#include "orchestration.hpp"
#include "video.hpp"

void VideoEventManager::add_subscriber(std::shared_ptr<Subscriber> subscriber) {
    // add subscriber to the  vector of subscribers
    this->subscribers.push_back(subscriber);
    // set the event queue for the subscriber
    subscriber->set_event_queue(&this->event_queue);
}

void VideoEventManager::run() {
    // call the start method for all subscribers
    for (auto &subscriber : this->subscribers) {
        subscriber->start();
    }
    // run the event manager
    while (true) {
        // check frame queue
        if (!this->frame_queue.empty()) {
            // get the frame token from the queue
            FrameToken frame_token = this->frame_queue.front();
            // notify all subscribers
            for (auto &subscriber : this->subscribers) {
                // check if subscriber is listening to the frame event
                if (subscriber->listening_to().find(EventType::NEW_FRAME) != subscriber->listening_to().end()) {
                    // create a frame event
                    Event event(EventType::NEW_FRAME, time(nullptr), std::nullopt);
                    // update the subscriber
                    subscriber->notify(event);
                }
            }
            // remove the frame token from the queue
            this->frame_queue.pop();
        }
        // check if event queue is empty
        if (this->event_queue.empty()) {
            continue;
        }
        // get the event from the queue
        Event event = this->event_queue.front();
        // notify all subscribers
        for (auto &subscriber : this->subscribers) {
            // check if subscriber is listening to the event type
            if (subscriber->listening_to().find(event.type) != subscriber->listening_to().end()) {
                // update the subscriber
                subscriber->notify(event);
            }
        }
        // remove the event from the queue
        this->event_queue.pop();
    }
}