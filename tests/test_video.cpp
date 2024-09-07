#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <video.cpp>
#include <ctime> 
#include <opencv2/opencv.hpp>

// helper function to create a random image

cv::Mat create_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_64FC1);
    double low = -500.0;
    double high = +500.0;
    cv::randu(img, cv::Scalar(low), cv::Scalar(high));
    return img;
}


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
    ImageStore store(1);
    cv::Mat img;
    CHECK_THROWS_AS(store.put(1, img), std::invalid_argument);
}

// test Imagestore drops oldest frames when size is exceeded

TEST_CASE("ImageStore drops oldest frames when size is exceeded") {
    ImageStore store(2);
    cv::Mat img1 = create_random_image(3, 3);
    cv::Mat img2 = create_random_image(3, 3);
    cv::Mat img3 = create_random_image(3, 3);
    time_t t1 = get_random_time();
    time_t t2 = get_random_time();
    time_t t3 = get_random_time();
    store.put(t1, img1);
    store.put(t2, img2);
    store.put(t3, img3);
    CHECK_THROWS_AS(store.get(1), std::invalid_argument);
    CHECK(cv::countNonZero(store.get(t2) != img2) == 0);
    CHECK(cv::countNonZero(store.get(t3) != img3) == 0);
}