#include "doctest/doctest.h"
#include "video.hpp"
#include "events.hpp"
#include <ctime> 
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat create_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_64FC1);
    double low = -500.0;
    double high = +500.0;
    cv::randu(img, cv::Scalar(low), cv::Scalar(high));
    return img;
}


// mock camera capture class that returns random images and inherits from CameraCapture


class MockCameraCapture : public CameraCapture {
public:
    MockCameraCapture(const char* device, int width, int height, uint32_t pixel_format) {}
    cv::Mat getNextFrame() override {
        if (frames.empty()) {
            // Return random image if no frames are set
            return create_random_image(3, 3);
        }
        if (frame_index >= frames.size()) {
            return cv::Mat();
        }
        cv::Mat frame = frames.at(frame_index);
        frame_index++;
        return frame;
    }
    void startStreaming() override {}
    void stopStreaming() override {}
    void setMockFrames(std::vector<cv::Mat> frames) {
        this->frames = frames;
        this->frame_index = 0;
    }
private:
    std::vector<cv::Mat> frames;
    int frame_index;
};

time_t get_random_time() {
    return rand();
}

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
    CHECK_THROWS_AS(store->put(1, img), std::invalid_argument);
}

// test Imagestore drops oldest frames when size is exceeded

TEST_CASE("ImageStore drops oldest frames when size is exceeded") {
    auto store = std::make_shared<ImageStore>(2);
    cv::Mat img1 = create_random_image(3, 3);
    cv::Mat img2 = create_random_image(3, 3);
    cv::Mat img3 = create_random_image(3, 3);
    time_t t1 = get_random_time();
    time_t t2 = get_random_time();
    time_t t3 = get_random_time();
    store->put(t1, img1);
    store->put(t2, img2);
    store->put(t3, img3);
    CHECK_THROWS_AS(store->get(1), std::invalid_argument);
    CHECK(cv::countNonZero(store->get(t2) != img2) == 0);
    CHECK(cv::countNonZero(store->get(t3) != img3) == 0);
}


// Test single frame is enqued correctly by stream

TEST_CASE("Single frame is enqued correctly by stream") {
    auto store = std::make_shared<ImageStore>(1);
    MockCameraCapture cam_capture("test", 3, 3, V4L2_PIX_FMT_YUYV);
    cv::Mat img = create_random_image(3, 3);
    cam_capture.setMockFrames({img});
    Stream stream(store, &cam_capture);
    auto frame_queue = std::make_shared<std::queue<FrameEvent>>();
    stream.register_frame_queue(frame_queue);
    stream.start();
    // Wait for 500 ms to allow the stream to process the frame
    usleep(10 * 1000);
    // stop the stream
    stream.stop();
    CHECK(frame_queue->size() == 1);
    FrameEvent frame_event = frame_queue->front();
    CHECK(cv::countNonZero(store->get(frame_event.get_timestamp()) != img) == 0);
}

// Test stream without frame queue cannot be started

TEST_CASE("Stream without frame queue cannot be started") {
    auto store = std::make_shared<ImageStore>(1);
    MockCameraCapture cam_capture("test", 3, 3, V4L2_PIX_FMT_YUYV);
    Stream stream(store, &cam_capture);
    CHECK_THROWS_AS(stream.start(), std::runtime_error);
}