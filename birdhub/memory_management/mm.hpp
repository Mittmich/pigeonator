#pragma once
#include <opencv2/opencv.hpp>
#include <optional>
#include <chrono>

// Use the same timestamp type as events
using Timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

const int MAX_IMAGE_STORE_SIZE = 5000;

class ImageStore
/*
    A shared store for the frames from the camera.
    It is a map with fixed size that drops the oldest frames
    when the size is exceeded.
*/
{
public:
    ImageStore(int size);
    void put(Timestamp timestamp, cv::Mat &image);
    std::optional<cv::Mat> get(Timestamp timestamp);

private:
    int size;
    std::deque<Timestamp> timestamp_queue;
    std::map<Timestamp, cv::Mat> image_map;
};
