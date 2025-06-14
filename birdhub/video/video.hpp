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
#include <vector>
#include <thread>
#include "events.hpp"
#include "mm.hpp"

#ifdef __linux__
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#endif

class CameraCapture {
public:
    // Destructor
    virtual ~CameraCapture() {}
    // method to start stream
    virtual void startStreaming() = 0;
    virtual void stopStreaming() = 0;
    // Method to get the next frame as an OpenCV Mat
    virtual cv::Mat getNextFrame()= 0;
};

#ifdef __linux__
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
#endif

// Cross-platform OpenCV-based camera capture
class OpenCVCameraCapture : public CameraCapture {
public:
    OpenCVCameraCapture(int camera_id = 0);
    ~OpenCVCameraCapture();
    void startStreaming() override;
    void stopStreaming() override;
    cv::Mat getNextFrame() override;

private:
    cv::VideoCapture cap;
    int camera_id_;
    bool started = false;
};




class Stream
/*
    Represents a stream. It reads frames from the camera
    and puts them in the shared image store and the frame queue.
*/
{
public:
    Stream(
        std::shared_ptr<ImageStore> image_store,
        CameraCapture *cam_capture,
        bool write_timestamps = true);
    // registering the frame queue needs to be separte from the constructor because
    // the evenemanager needs to attach it to the stream
    virtual void register_frame_queue(std::shared_ptr<std::queue<FrameEvent>> frame_queue);
    virtual void start();
    virtual void stop();

protected:
    void enque_frame_token();
    std::shared_ptr<ImageStore> image_store;
    bool write_timestamps;
    CameraCapture *cam_capture;
    std::shared_ptr<std::queue<FrameEvent>> frame_queue;
    void _start();
    std::thread queue_thread;
    bool running = false;
    bool queue_registered = false;
};

#endif