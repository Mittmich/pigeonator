#include "doctest/doctest.h"
#include "video.hpp"
#include "events.hpp"
#include "timestamp_utils.hpp"
#include <ctime> 
#include <vector>
#include <opencv2/opencv.hpp>
#include "test_utils.hpp"
#include <unistd.h>

#ifdef __linux__
#include <linux/videodev2.h>
#else
// Mock V4L2 constants for non-Linux platforms
#define V4L2_PIX_FMT_YUYV 0x56595559
#endif


// Test Imagestore throws exception when size is negative

TEST_CASE("ImageStore throws exception when size is negative") {
    CHECK_THROWS_AS(ImageStore store(-1), std::invalid_argument);
}

// test Imagestore throws exception when size is too large

TEST_CASE("ImageStore throws exception when size is too large") {
    CHECK_THROWS_AS(ImageStore store(MAX_IMAGE_STORE_SIZE + 1), std::invalid_argument);
}

// test Imagestore throws exception when image is empty

TEST_CASE("ImageStore throws exception when image is empty") {
    auto store = std::make_shared<ImageStore>(1);
    cv::Mat img;
    CHECK_THROWS_AS(store->put(test_now(), img), std::invalid_argument);
}

// test Imagestore drops oldest frames when size is exceeded

TEST_CASE("ImageStore drops oldest frames when size is exceeded") {
    auto store = std::make_shared<ImageStore>(2);
    cv::Mat img1 = create_random_image(3, 3);
    cv::Mat img2 = create_random_image(3, 3);
    cv::Mat img3 = create_random_image(3, 3);
    Timestamp t1 = test_timestamp_offset_ms(-100);
    Timestamp t2 = test_timestamp_offset_ms(-50);
    Timestamp t3 = test_now();
    store->put(t1, img1);
    store->put(t2, img2);
    store->put(t3, img3);
    CHECK(store->get(t1).has_value() == false);
    CHECK(store->get(t2).has_value() == true);
    CHECK(store->get(t3).has_value() == true);
    cv::Mat img2_retrieved = store->get(t2).value();
    cv::Mat img3_retrieved = store->get(t3).value();
    CHECK(cv::countNonZero(img2_retrieved != img2) == 0);
    CHECK(cv::countNonZero(img3_retrieved != img3) == 0);
}


// Test single frame is enqued correctly by stream

TEST_CASE("Single frame is enqued correctly by stream") {
    auto store = std::make_shared<ImageStore>(1);
    MockCameraCapture cam_capture("test", 3, 3, V4L2_PIX_FMT_YUYV);
    cv::Mat img = create_random_image(3, 3);
    cam_capture.setMockFrames({img});
    Stream stream(store, &cam_capture);
    auto frame_queue = std::make_shared<std::queue<std::shared_ptr<FrameEvent>>>();
    stream.register_frame_queue(frame_queue);
    stream.start();
    // Wait for 500 ms to allow the stream to process the frame
    usleep(10 * 1000);
    // stop the stream
    stream.stop();
    CHECK(frame_queue->size() == 1);
    std::shared_ptr<FrameEvent> frame_event = frame_queue->front();
    cv::Mat image_retrieved = store->get(frame_event->get_timestamp()).value();
    CHECK(cv::countNonZero(image_retrieved != img) == 0);
}

// Test stream without frame queue cannot be started

TEST_CASE("Stream without frame queue cannot be started") {
    auto store = std::make_shared<ImageStore>(1);
    MockCameraCapture cam_capture("test", 3, 3, V4L2_PIX_FMT_YUYV);
    Stream stream(store, &cam_capture);
    CHECK_THROWS_AS(stream.start(), std::runtime_error);
}