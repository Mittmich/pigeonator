#include "doctest/doctest.h"
#include "detection.hpp"
#include "events.hpp"
#include <opencv2/opencv.hpp>


cv::Mat create_test_image(int rows, int cols, uchar value) {
    // Create 3-channel image (BGR)
    cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(value, value, value));
    return img;
}
std::shared_ptr<ImageStore> setup_image_store() {
    auto store = std::make_shared<ImageStore>(10);
    return store;
}

TEST_CASE("MotionDetector - No motion detected for single fame.") {
    auto store = setup_image_store();
    MotionDetector detector(store, 20, 3, 3, 50, 0, std::chrono::seconds(5));
    
    cv::Mat img = create_test_image(100, 100, 128);
    time_t t1 = std::time(nullptr);
    store->put(t1, img);
    
    auto event = std::make_shared<FrameEvent>(t1, std::nullopt);
    auto result = detector.detect(event);
    CHECK(!result.has_value());
}

TEST_CASE("MotionDetector - No motion detected for identical frames.") {
    auto store = setup_image_store();
    MotionDetector detector(store, 20, 3, 3, 50, 0, std::chrono::seconds(5));
    
    cv::Mat img1 = create_test_image(100, 100, 128);
    time_t t1 = std::time(nullptr);
    store->put(t1, img1);

    cv::Mat img2 = create_test_image(100,100,128);
    time_t t2 = t1 + 2;
    store->put(t2, img2);

    auto event1 = std::make_shared<FrameEvent>(t1, std::nullopt);
    auto event2 = std::make_shared<FrameEvent>(t2, std::nullopt);
    auto result1 = detector.detect(event1);
    auto result2 = detector.detect(event2);
    CHECK(!result1.has_value());
    CHECK(!result2.has_value());
}

TEST_CASE("MotionDetector - Motion detected between different frames") {
    auto store = setup_image_store();
    MotionDetector detector(store, 20, 3, 3, 50, 0, std::chrono::seconds(5));
    
    // First frame
    cv::Mat img1 = create_test_image(100, 100, 0);
    time_t t1 = std::time(nullptr);
    store->put(t1, img1);
    auto event1 = std::make_shared<FrameEvent>(t1, std::nullopt);
    detector.detect(event1);
    
    // Second frame with motion
    cv::Mat img2 = create_test_image(100, 100, 255);
    time_t t2 = t1 + 1;
    store->put(t2, img2);
    auto event2 = std::make_shared<FrameEvent>(t2, std::nullopt);
    auto result = detector.detect(event2);
    
    CHECK(result.has_value());
}

TEST_CASE("MotionDetector - Respects activation frames threshold") {
    auto store = setup_image_store();
    MotionDetector detector(store, 20, 3, 3, 50, 1, std::chrono::seconds(5));
    
    cv::Mat img1 = create_test_image(100, 100, 0);
    cv::Mat img2 = create_test_image(100, 100, 255);
    cv::Mat img3 = create_test_image(100,100,50);
    
    // First frame
    time_t t1 = std::time(nullptr);
    store->put(t1, img1);
    auto event1 = std::make_shared<FrameEvent>(t1, std::nullopt);
    detector.detect(event1);

    // second frame
    time_t t2 = t1 + 1;
    store->put(t2, img2);
    auto event2 = std::make_shared<FrameEvent>(t2, std::nullopt);
    auto result = detector.detect(event2);
    CHECK(!result.has_value());

    // third frame -> should trigger
    time_t t3 = t1 + 2;
    store->put(t3, img3);
    auto event3 = std::make_shared<FrameEvent>(t3, std::nullopt);
    auto result2 = detector.detect(event3);
    CHECK(result2.has_value());
}

TEST_CASE("MotionDetector - Respects max delay threshold") {
    auto store = setup_image_store();
    std::chrono::seconds max_delay(5);
    MotionDetector detector(store, 20, 3, 3, 50, 0, max_delay);
    
    cv::Mat img1 = create_test_image(100, 100, 0);
    cv::Mat img2 = create_test_image(100, 100, 255);
    cv::Mat img3 = create_test_image(100,100,50);
    
    // Set up initial frame
    time_t current_time = std::time(nullptr);
    time_t t1 = current_time - 2; // Within threshold
    store->put(t1, img1);
    auto event1 = std::make_shared<FrameEvent>(t1, std::nullopt);
    detector.detect(event1);

    // Test frame within delay threshold
    time_t t2 = current_time - 1;
    store->put(t2, img2);
    auto event2 = std::make_shared<FrameEvent>(t2, std::nullopt);
    auto result1 = detector.detect(event2);
    CHECK(result1.has_value());

    // Test frame outside delay threshold
    time_t t3 = current_time - 10; // Outside threshold
    store->put(t3, img3);
    auto event3 = std::make_shared<FrameEvent>(t3, std::nullopt);
    auto result2 = detector.detect(event3);
    CHECK(!result2.has_value());
}

TEST_CASE("MotionDetector - Integration test with event queue") {
    auto store = setup_image_store();
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    
    MotionDetector detector(store, 20, 3, 3, 50, 0, std::chrono::seconds(5));
    detector.set_event_queue(event_queue);
    detector.start();
    
    // Create frames with motion
    cv::Mat img1 = create_test_image(100, 100, 0);
    cv::Mat img2 = create_test_image(100, 100, 255);
    
    // Push first frame
    time_t t1 = std::time(nullptr);
    store->put(t1, img1);
    auto event1 = std::make_shared<FrameEvent>(t1, std::nullopt);
    detector.notify(event1);
    
    // Push second frame with motion
    time_t t2 = t1 + 1;
    store->put(t2, img2);
    auto event2 = std::make_shared<FrameEvent>(t2, std::nullopt);
    detector.notify(event2);
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify detection event in queue
    CHECK(event_queue->size() == 1);
    auto detection_event = event_queue->front();
    CHECK(detection_event->type == EventType::DETECTION);
    
    detector.stop();
}