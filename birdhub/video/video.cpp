/*
    Implementation for video.hpp.
*/

#include "video.hpp"


ImageStore::ImageStore(int size) {
    // check if size is negative
    if (size <= 0) {
        throw std::invalid_argument("Size must be greater than 0.");
    }
    // check if size is too large
    if (size > MAX_IMAGE_STORE_SIZE) {
        throw std::invalid_argument("Size must be less than " + std::to_string(MAX_IMAGE_STORE_SIZE) + ".");
    }
    this->size = size;
}

void ImageStore::put(std::time_t timestamp, cv::Mat image) {
    // check if image is empty
    if (image.empty()) {
        throw std::invalid_argument("Image must not be empty.");
    }
    // check if store is full
    if (this->timestamp_queue.size() >= this->size) {
        this->timestamp_queue.pop_front();
    }
    this->timestamp_queue.push_back(timestamp);
    this->image_map[timestamp] = image;
}

cv::Mat ImageStore::get(std::time_t timestamp) {
    // check if timestamp is in store
    if (this->image_map.count(timestamp) == 0) {
        throw std::invalid_argument("Timestamp not found in store.");
    }
    return this->image_map[timestamp];
}