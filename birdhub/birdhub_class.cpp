#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdexcept>

class CameraCapture {
public:
    // Constructor
    CameraCapture(const char* device, int width, int height, uint32_t pixel_format, bool non_blocking = false)
        : device_(device), width_(width), height_(height), pixel_format_(pixel_format), fd_(-1), non_blocking_(non_blocking) {
        if (!openDevice()) {
            throw std::runtime_error("Failed to open camera device");
        }
        if (!initDevice()) {
            throw std::runtime_error("Failed to initialize camera device");
        }
    }

    // Destructor
    ~CameraCapture() {
        stopStreaming();
        cleanup();
    }

    // Method to get the next frame as an OpenCV Mat
    cv::Mat getNextFrame() {
        v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        bool valid_frame = false;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN && non_blocking_) {
                std::cerr << "No frame available (non-blocking), retrying..." << std::endl;
                return cv::Mat();  // Return empty Mat if no frame is available
            } else {
                std::cerr << "Failed to dequeue buffer: " << strerror(errno) << std::endl;
                throw std::runtime_error("Failed to dequeue buffer");
            }
        }


        // Process the frame depending on the pixel format
        cv::Mat frame;
        if (pixel_format_ == V4L2_PIX_FMT_MJPEG) {
            std::vector<uchar> jpegData((uchar*)buffers_[buf.index], (uchar*)buffers_[buf.index] + buf.bytesused);
            frame = cv::imdecode(jpegData, cv::IMREAD_COLOR);
        } else if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
            cv::Mat yuyv(height_, width_, CV_8UC2, buffers_[buf.index]);
            cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUYV);
        }

        // Requeue the buffer
        if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "Failed to requeue buffer" << std::endl;
            throw std::runtime_error("Failed to requeue buffer");
        }

        return frame;
    }

private:
    const char* device_;
    int width_;
    int height_;
    uint32_t pixel_format_;
    int fd_;
    bool non_blocking_;
    std::vector<void*> buffers_;

    bool openDevice() {
        // Open the camera device with or without non-blocking mode
        fd_ = open(device_, O_RDWR | (non_blocking_ ? O_NONBLOCK : 0));
        if (fd_ == -1) {
            std::cerr << "Failed to open device" << std::endl;
            return false;
        }
        return true;
    }

    bool initDevice() {
        // Query capabilities
        v4l2_capability cap;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == -1) {
            std::cerr << "Failed to query capabilities" << std::endl;
            return false;
        }

        // Set format
        v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = pixel_format_;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
            std::cerr << "Failed to set format. Details: " << strerror(errno) << std::endl;
            return false;
        }

        // Request buffers
        v4l2_requestbuffers req = {0};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "Failed to request buffers" << std::endl;
            return false;
        }

        // Map buffers
        buffers_.resize(req.count);
        for (unsigned int i = 0; i < req.count; ++i) {
            v4l2_buffer buf = {0};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1) {
                std::cerr << "Failed to query buffer" << std::endl;
                return false;
            }
            buffers_[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        }

        // Queue the buffers
        for (unsigned int i = 0; i < req.count; ++i) {
            v4l2_buffer buf = {0};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Failed to queue buffer" << std::endl;
                close(fd_);
                return 1;
            }
        }

        // Start streaming
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
            std::cerr << "Failed to start streaming" << std::endl;
            return false;
        }

        return true;
    }

    void stopStreaming() {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMOFF, &type) == -1) {
            std::cerr << "Failed to stop streaming" << std::endl;
        }
    }

    void cleanup() {
        // Unmap buffers
        for (auto& buffer : buffers_) {
            munmap(buffer, width_ * height_ * 2);  // 2 bytes per pixel for YUYV (adjust for other formats if necessary)
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }
};


int main() {
    try {
        // Create the camera capture object with the desired settings
        CameraCapture camera("/dev/video0", 640, 480, V4L2_PIX_FMT_MJPEG);

        // Define the codec and create a VideoWriter object to write the output video
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // Codec for MP4 (MPEG-4)
        cv::VideoWriter videoWriter("output_video.mp4", fourcc, 30.0, cv::Size(640, 480));

        // Check if the video writer is initialized correctly
        if (!videoWriter.isOpened()) {
            std::cerr << "Could not open the video file for writing" << std::endl;
            return -1;
        }

        // Capture and write frames to the video file
        for (int i = 0; i < 300; ++i) {  // Capture 300 frames (~10 seconds of video at 30 FPS)
            cv::Mat frame = camera.getNextFrame();
            if (!frame.empty()) {
                // Write the frame to the MP4 file
                videoWriter.write(frame);
            } else {
                std::cerr << "Empty frame captured" << std::endl;
            }
        }

        // Release the VideoWriter
        videoWriter.release();
        std::cout << "Video has been saved to output_video.mp4" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
