/*
    Implementation for video.hpp.
*/

#include "video.hpp"

ImageStore::ImageStore(int size)
{
    // check if size is negative
    if (size <= 0)
    {
        throw std::invalid_argument("Size must be greater than 0.");
    }
    // check if size is too large
    if (size > MAX_IMAGE_STORE_SIZE)
    {
        throw std::invalid_argument("Size must be less than " + std::to_string(MAX_IMAGE_STORE_SIZE) + ".");
    }
    this->size = size;
}

void ImageStore::put(std::time_t timestamp, cv::Mat &image)
{
    // check if image is empty
    if (image.empty())
    {
        throw std::invalid_argument("Image must not be empty.");
    }
    // check if store is full
    if (this->timestamp_queue.size() >= this->size)
    {
        // get front element from queue and delete it from map
        std::time_t front = this->timestamp_queue.front();
        this->image_map.erase(front);
        // delete front element from queue
        this->timestamp_queue.pop_front();
    }
    this->timestamp_queue.push_back(timestamp);
    this->image_map[timestamp] = image;
}

cv::Mat ImageStore::get(std::time_t timestamp)
{
    // check if timestamp is in store
    if (this->image_map.count(timestamp) == 0)
    {
        throw std::invalid_argument("Timestamp not found in store.");
    }
    return this->image_map[timestamp];
}


V4l2CameraCapture::V4l2CameraCapture(const char* device, int width, int height, uint32_t pixel_format, bool non_blocking = false)
    : device_(device), width_(width), height_(height), pixel_format_(pixel_format), fd_(-1), non_blocking_(non_blocking) {}

void V4l2CameraCapture::startStreaming() {
    if (!openDevice()) {
        throw std::runtime_error("Failed to open camera device");
    }
    if (!initDevice()) {
        throw std::runtime_error("Failed to initialize camera device");
    }
    started = true;
}

// Destructor
V4l2CameraCapture::~V4l2CameraCapture() {
    this->stopStreaming();
    this->cleanup();
}

// Method to get the next frame as an OpenCV Mat
cv::Mat V4l2CameraCapture::getNextFrame() {
    if (!started) {
        throw std::runtime_error("V4l2CameraCapture not started");
    }
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


bool V4l2CameraCapture::openDevice() {
    // Open the camera device with or without non-blocking mode
    fd_ = open(device_, O_RDWR | (non_blocking_ ? O_NONBLOCK : 0));
    if (fd_ == -1) {
        std::cerr << "Failed to open device" << std::endl;
        return false;
    }
    return true;
}

bool V4l2CameraCapture::initDevice() {
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

void V4l2CameraCapture::stopStreaming() {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type) == -1) {
        std::cerr << "Failed to stop streaming" << std::endl;
    }
    started = false;
}

void V4l2CameraCapture::cleanup() {
    // Unmap buffers
    for (auto& buffer : buffers_) {
        munmap(buffer, width_ * height_ * 2);  // 2 bytes per pixel for YUYV (adjust for other formats if necessary)
    }
    if (fd_ != -1) {
        close(fd_);
    }
    }


Stream::Stream(ImageStore &image_store, CameraCapture *cam_capture, bool write_timestamps)
    : image_store(image_store), cam_capture(cam_capture), write_timestamps(write_timestamps) {
}

void Stream::register_frame_queue(std::queue<FrameToken> *frame_queue) {
    this->frame_queue = frame_queue;
}

void Stream::start() {
    // set running flag that is used to stop the thread
    this->running = true;
    // start thread
    this->queue_thread = std::thread(&Stream::_start, this);
}

void Stream::stop() {
    // set running flag to false to stop the thread
    this->running = false;
    // join the thread
    this->queue_thread.join();
}

void Stream::_start() {
    while (this->running) {
        enque_frame_token(this->frame_queue);
    }
}

void Stream::enque_frame_token(std::queue<FrameToken> *frame_queue) {
    cv::Mat frame = cam_capture->getNextFrame();
    if (frame.empty()) {
        return;
    }
    std::time_t timestamp = std::time(0);
    FrameToken token = {timestamp, timestamp};
    image_store.put(timestamp, frame);
    frame_queue->push(token);
}