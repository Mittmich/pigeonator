#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    const char* device = "/dev/video0";
    int fd = open(device, O_RDWR);
    if (fd == -1) {
        std::cerr << "Failed to open device" << std::endl;
        return 1;
    }
    std::cout << "Device opened" << std::endl;

    // Query capabilities
    v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        std::cerr << "Failed to query capabilities" << std::endl;
        close(fd);
        return 1;
    }
    std::cout << "Driver: " << cap.driver << std::endl;

    // Set format to MJPEG
    v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;  // MJPEG format
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        std::cerr << "Failed to set format to MJPEG. Details: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }
    std::cout << "Format set to MJPEG" << std::endl;

    // Request buffers
    v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        std::cerr << "Failed to request buffers" << std::endl;
        close(fd);
        return 1;
    }
    std::cout << "Buffers requested" << std::endl;

    // Map buffers
    std::vector<void*> buffers(req.count);
    for (unsigned int i = 0; i < req.count; ++i) {
        v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            std::cerr << "Failed to query buffer" << std::endl;
            close(fd);
            return 1;
        }
        buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    }
    std::cout << "Buffers mapped" << std::endl;

    // Queue the buffers
    for (unsigned int i = 0; i < req.count; ++i) {
        v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "Failed to queue buffer" << std::endl;
            close(fd);
            return 1;
        }
    }

    // Start streaming
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        std::cerr << "Failed to start streaming" << std::endl;
        close(fd);
        return 1;
    }
    std::cout << "Streaming started" << std::endl;

    // wait for 10 seconds
    usleep(10000000);
    // debug output
    std::cout << "10 seconds passed" << std::endl;

    for (int attempt = 0; attempt < 10; ++attempt) {
        // log the attempt
        std::cout << "Attempt " << attempt << std::endl;
        // Capture frame and save to file
        v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            std::cerr << "Failed to dequeue buffer" << std::endl;
        } else {
            std::cout << "Buffer dequeued" << std::endl;

            // Convert MJPEG to BGR using OpenCV
            std::vector<uchar> jpegData((uchar*)buffers[buf.index], (uchar*)buffers[buf.index] + buf.bytesused);
            cv::Mat img = cv::imdecode(jpegData, cv::IMREAD_COLOR);

            // Save the image
            cv::imwrite("captured_frame_mjpeg.jpg", img);
            std::cout << "Frame saved to captured_frame_mjpeg.jpg" << std::endl;

            // Requeue buffer
            if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Failed to requeue buffer" << std::endl;
            }
            break;  // Save just one frame
        }

        usleep(100000);  // Wait 100ms
    }

    // Stop streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        std::cerr << "Failed to stop streaming" << std::endl;
    }

    // Cleanup
    for (auto& buffer : buffers) {
        munmap(buffer, fmt.fmt.pix.sizeimage);
    }
    close(fd);

    return 0;
}
