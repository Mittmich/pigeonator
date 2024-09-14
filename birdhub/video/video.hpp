/*
    Functionality for streaming from camera.
*/

#ifndef BIRDHUB_VIDEO_HPP
#define BIRDHUB_VIDEO_HPP

#include <ctime>
#include <deque>
#include <queue>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

const int MAX_IMAGE_STORE_SIZE = 1000;

struct Frame
/*
    Represents a frame from the camera.
    Contains the frame data and the timestamp of the frame.
    The timestamp for each frame is unique and serves
    as id to retrive the data from the shared image store.
*/
{
    std::time_t timestamp;
    std::time_t capture_time;
};

class ImageStore
/*
    A shared store for the frames from the camera.
    It is a map with fixed size that drops the oldest frames
    when the size is exceeded.
    The store is thread-safe.
*/
{
public:
    ImageStore(int size);
    void put(std::time_t timestamp, cv::Mat &image);
    cv::Mat get(std::time_t timestamp);

private:
    int size;
    std::deque<time_t> timestamp_queue;
    std::map<std::time_t, cv::Mat> image_map;
};

enum class StreamBackend
{
    OPENCV,
    LIBCAMERA
};

class Stream
/*
    Represents a stream. Implements methods to
    start the stream, and get frames.
*/
{
public:
    Stream(
        std::string stream_source,
        ImageStore &image_store,
        bool write_timestamps = true,
        StreamBackend backend = StreamBackend::OPENCV);
    void start(std::queue<Frame> &frame_queue);

private:
    Frame get_frame();
    ImageStore &image_store;
    std::string stream_source;
    bool write_timestamps;
    StreamBackend backend;
};

#endif