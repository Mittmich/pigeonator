#include <opencv2/opencv.hpp>
#include <optional>

const int MAX_IMAGE_STORE_SIZE = 1000;

class ImageStore
/*
    A shared store for the frames from the camera.
    It is a map with fixed size that drops the oldest frames
    when the size is exceeded.
*/
{
public:
    ImageStore(int size);
    void put(std::time_t timestamp, cv::Mat &image);
    std::optional<cv::Mat> get(std::time_t timestamp);

private:
    int size;
    std::deque<time_t> timestamp_queue;
    std::map<std::time_t, cv::Mat> image_map;
};
