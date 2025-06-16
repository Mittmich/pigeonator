#include "orchestration.hpp"
#include <iostream>
#include <signal.h>
#include <atomic>


// Global signal flag for graceful shutdown
std::atomic<bool> g_shutdown_requested{false};


void VideoEventManager::add_subscriber(std::shared_ptr<Subscriber> subscriber) {
    // add subscriber to the  vector of subscribers
    this->subscribers.push_back(subscriber);
    // set the event queue for the subscriber
    subscriber->set_event_queue(this->event_queue);
}

void VideoEventManager::run() {
    // register the frame queue with the stream
    this->stream.register_frame_queue(this->frame_queue);
    // start the stream
    this->stream.start();
    // call the start method for all subscribers
    for (auto &subscriber : this->subscribers) {
        subscriber->start();
    }
    // set running flag to true
    this->running = true;
    // run the event manager
    while (true) {
        // check whether event manager should continue to run
        if (!this->running || g_shutdown_requested) {
            break;
        }
        // check frame queue
        if (!this->frame_queue->empty()) {
            std::shared_ptr<FrameEvent> frame_event = this->frame_queue->front();
            for (auto &subscriber : this->subscribers) {
                // check if subscriber is listening to the frame event
                if (subscriber->listening_to().find(EventType::NEW_FRAME) != subscriber->listening_to().end()) {
                    subscriber->notify(frame_event);
                }
            }
            // remove the frame token from the queue
            this->frame_queue->pop();
        }
        // check if event queue is empty
        if (this->event_queue->empty()) {
            continue;
        }
        // get the event from the queue
        auto event = this->event_queue->front();
        // notify all subscribers
        for (auto &subscriber : this->subscribers) {
            // check if subscriber is listening to the event type
            if (subscriber->listening_to().find(event->type) != subscriber->listening_to().end()) {
                // update the subscriber
                subscriber->notify(event);
            }
        }
        // remove the event from the queue
        this->event_queue->pop();
    }
    
    // If shutdown was requested by signal, call stop to clean up properly
    if (g_shutdown_requested) {
        this->stop();
    }
}

void VideoEventManager::stop() {
    // stop the stream
    this->stream.stop();
    // stop all subscribers
    for (auto &subscriber : this->subscribers) {
        subscriber->stop();
    }
    // set running flag to false
    this->running = false;
}

VideoEventManager::VideoEventManager(Stream &stream) : stream(stream) {
    // create event queue
    this->event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    // create frame queue
    this->frame_queue = std::make_shared<std::queue<std::shared_ptr<FrameEvent>>>();
}

VideoEventManager::~VideoEventManager() {
    // stop the event manager
    this->stop();
}

// Signal handling methods
void VideoEventManager::setup_signal_handlers() {
    struct sigaction sa;
    sa.sa_handler = VideoEventManager::signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    // Register handler for SIGTERM and SIGINT
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
}

void VideoEventManager::signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", initiating graceful shutdown..." << std::endl;
    g_shutdown_requested = true;
}