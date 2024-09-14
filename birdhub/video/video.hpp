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
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <vector>
#include <thread>

const int MAX_IMAGE_STORE_SIZE = 1000;

struct FrameToken
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


class CameraCapture {
public:
    // method to start stream
    virtual void startStreaming() = 0;
    virtual void stopStreaming() = 0;
    // Method to get the next frame as an OpenCV Mat
    virtual cv::Mat getNextFrame()= 0;
};


// V4l2CameraCapture class

class V4l2CameraCapture : public CameraCapture {
public:
    V4l2CameraCapture(const char* device, int width, int height, uint32_t pixel_format, bool non_blocking);
    ~V4l2CameraCapture();
    void startStreaming() override;
    void stopStreaming() override;
    cv::Mat getNextFrame() override;

private:
    const char* device_;
    int width_;
    int height_;
    uint32_t pixel_format_;
    int fd_;
    bool non_blocking_;
    std::vector<void*> buffers_;
    bool started = false;
    bool openDevice();
    bool initDevice();
    void cleanup();
};




class Stream
/*
    Represents a stream. It reads frames from the camera
    and puts them in the shared image store and the frame queue.
*/
{
public:
    Stream(
        ImageStore &image_store,
        CameraCapture *cam_capture,
        bool write_timestamps = true);
    void register_frame_queue(std::queue<FrameToken> *frame_queue);
    void start();
    void stop();

private:
    void enque_frame_token(std::queue<FrameToken> *frame_queue);
    ImageStore &image_store;
    bool write_timestamps;
    CameraCapture *cam_capture;
    std::queue<FrameToken> *frame_queue;
    void _start();
    std::thread queue_thread;
    bool running = false;
};

#endif