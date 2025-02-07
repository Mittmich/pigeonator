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
    
    FrameEvent event(t1, std::nullopt);
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

    FrameEvent event(t1, std::nullopt);
    FrameEvent event2(t2, std::nullopt);
    auto result1 = detector.detect(event);
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
    FrameEvent event1(t1, std::nullopt);
    detector.detect(event1);
    
    // Second frame with motion
    cv::Mat img2 = create_test_image(100, 100, 255);
    time_t t2 = t1 + 1;
    store->put(t2, img2);
    FrameEvent event2(t2, std::nullopt);
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
    FrameEvent event1(t1, std::nullopt);
    detector.detect(event1);

    // second frame
    time_t t2 = t1 + 1;
    store->put(t2, img2);
    FrameEvent event2(t2, std::nullopt);
    auto result = detector.detect(event2);
    CHECK(!result.has_value());

    // third frame -> should trigger
    time_t t3 = t1 + 2;
    store->put(t3, img3);
    FrameEvent event3(t3, std::nullopt);
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
    FrameEvent event1(t1, std::nullopt);
    detector.detect(event1);

    // Test frame within delay threshold
    time_t t2 = current_time - 1;
    store->put(t2, img2);
    FrameEvent event2(t2, std::nullopt);
    auto result1 = detector.detect(event2);
    CHECK(result1.has_value());

    // Test frame outside delay threshold
    time_t t3 = current_time - 10; // Outside threshold
    store->put(t3, img3);
    FrameEvent event3(t3, std::nullopt);
    auto result2 = detector.detect(event3);
    CHECK(!result2.has_value());
}