#include "doctest/doctest.h"
#include "orchestration.hpp"
#include <memory>
#include <queue>
#include <set>
#include <thread>
#include "test_utils.hpp"

TEST_CASE("VideoEventManager - Add subscriber") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    // add subscriber
    auto subscriber = std::make_shared<MockSubscriber>();
    
    manager.add_subscriber(subscriber);
    CHECK(subscriber->event_queue != nullptr);
}

// Test case whether frame queue is registered with stream

TEST_CASE("VideoEventManager - Register frame queue") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    
    auto subscriber = std::make_shared<MockSubscriber>();
    subscriber->event_types.insert(EventType::NEW_FRAME);
    manager.add_subscriber(subscriber);
    
    // Start manager in separate thread
    std::thread manager_thread([&manager]() {
        manager.run();
    });
    
    // Allow time for startup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(stream.has_queue());
    CHECK(stream.is_running());
    // stop manager
    manager.stop();
    manager_thread.join();
}

TEST_CASE("VideoEventManager - Frame event propagation") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    
    auto subscriber = std::make_shared<MockSubscriber>();
    subscriber->event_types.insert(EventType::NEW_FRAME);
    manager.add_subscriber(subscriber);
    
    // Start manager in separate thread
    std::thread manager_thread([&manager]() {
        manager.run();
    });
    
    // Allow time for startup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Simulate frame
    auto frame = std::make_shared<FrameEvent>(std::time(nullptr), std::nullopt);
    stream.simulate_frame(frame);
    
    // Allow time for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(subscriber->received_events.size() == 1);
    // stop manager
    manager.stop();
    manager_thread.join();
}


TEST_CASE("VideoEventManager - Event propagation") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    
    // add reader subscriber
    auto subscriber = std::make_shared<MockSubscriber>();
    subscriber->event_types.insert(EventType::DETECTION);
    manager.add_subscriber(subscriber);

    // add writer subscriber
    auto writer = std::make_shared<MockSubscriber>();
    manager.add_subscriber(writer);
    
    std::thread manager_thread([&manager]() {
        manager.run();
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Push event to queue
    auto frame_event = std::make_shared<FrameEvent>(std::time(nullptr), std::nullopt);
    Detection detection(std::time(nullptr), frame_event, std::nullopt);
    auto event = std::make_shared<DetectionEvent>(std::time(nullptr), std::vector<Detection>{detection}, std::nullopt);
    writer->simulate_event(event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(subscriber->received_events.size() == 1);
    CHECK(subscriber->received_events[0]->type == EventType::DETECTION);
    
    manager.stop();
    manager_thread.join();
}

TEST_CASE("VideoEventManager - Multiple subscribers") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    
    auto subscriber1 = std::make_shared<MockSubscriber>();
    auto subscriber2 = std::make_shared<MockSubscriber>();
    subscriber1->event_types.insert(EventType::DETECTION);
    subscriber2->event_types.insert(EventType::DETECTION);
    
    manager.add_subscriber(subscriber1);
    manager.add_subscriber(subscriber2);

    // add writer subscriber
    auto writer = std::make_shared<MockSubscriber>();
    manager.add_subscriber(writer);

    
    std::thread manager_thread([&manager]() {
        manager.run();
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Push event to queue
    auto frame_event = std::make_shared<FrameEvent>(std::time(nullptr), std::nullopt);
    Detection detection(std::time(nullptr), frame_event, std::nullopt);
    auto event = std::make_shared<DetectionEvent>(std::time(nullptr), std::vector<Detection>{detection}, std::nullopt);
    writer->simulate_event(event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(subscriber1->received_events.size() == 1);
    CHECK(subscriber2->received_events.size() == 1);
    
    manager.stop();
    manager_thread.join();
}

TEST_CASE("VideoEventManager - Startup sequence") {
    // instantiate mock camera capture
    auto camera_capture = ConstantMockCameraCapture("", 0, 0, 0);
    // instantate mock stream
    auto stream = MockStream(nullptr, &camera_capture);
    VideoEventManager manager(stream);
    
    auto subscriber = std::make_shared<MockSubscriber>();
    manager.add_subscriber(subscriber);
    
    std::thread manager_thread([&manager]() {
        manager.run();
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(stream.is_running());
    CHECK(subscriber->is_running);
    
    manager.stop();
    manager_thread.join();
}