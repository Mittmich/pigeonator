#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <video.cpp>
#include <opencv2/opencv.hpp>

// helper function to create a random image

cv::Mat create_random_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_64FC1);
    double low = -500.0;
    double high = +500.0;
    cv::randu(img, cv::Scalar(low), cv::Scalar(high));
    return img;
}

// Test Imagestore throws exception when size is negative

TEST_CASE("ImageStore throws exception when size is negative") {
    CHECK_THROWS_AS(ImageStore store(-1), std::invalid_argument);
}


