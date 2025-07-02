#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <set>
#include <thread>
#include "mm.hpp"
#include <chrono>

#pragma once

// add detector base class, inheriting from Subscriber

class Detector : public Subscriber {
public:
    Detector(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store
    );
    ~Detector();
    std::set<EventType> listening_to() override;
    void start() override;
    void stop() override;
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_queue) override;
    void notify(std::shared_ptr<Event> event) override;
    virtual std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) = 0;

protected:
    std::shared_ptr<std::queue<std::shared_ptr<Event>>> event_write_queue;
    std::set<EventType> listening_events;
    std::shared_ptr<ImageStore> image_store;
    void _start();
    std::thread queue_thread;
    bool running = false;
    bool queue_registered = false;
    void poll_read_queue();
    std::queue<std::shared_ptr<FrameEvent>> event_read_queue;
};

// create subclass of Detector for motion detection

class MotionDetector : public Detector {
public:
    MotionDetector(
        std::shared_ptr<ImageStore> image_store,
        int threshold,
        int blur,
        int dilate,
        int threshold_area,
        int activation_frames,
        std::chrono::seconds max_delay
    );
    ~MotionDetector();
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override;
private:
    int threshold;
    int blur;
    int dilate;
    int threshold_area;
    int activation_frames;
    std::chrono::seconds max_delay;
    std::optional<cv::Mat> previous_image;
    int motion_frames;
    std::vector<Detection> detections;
    cv::Mat preprocess_image(cv::Mat image);
};

// create subclass of Detector for bird detection using YOLOv5

class BirdDetectorYolov5 : public Detector {
public:
    BirdDetectorYolov5(
        std::shared_ptr<ImageStore> image_store,
        const std::string& model_path,
        cv::Size image_size = cv::Size(640, 640),
        float confidence_threshold = 0.25f,
        float iou_threshold = 0.45f,
        std::chrono::seconds max_delay = std::chrono::seconds(10),
        int threshold_area = 50
    );
    ~BirdDetectorYolov5();
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override;

private:
    std::string model_path;
    cv::dnn::Net net;
    cv::Size image_size;
    float confidence_threshold;
    float iou_threshold;
    std::chrono::seconds max_delay;
    int threshold_area;
    std::vector<std::string> class_names;
    bool model_loaded;
    
    void load_model();
    cv::Mat preprocess_image(const cv::Mat& image);
    std::vector<cv::Rect> extract_boxes(const std::vector<cv::Mat>& outputs, const cv::Size& original_size, 
                                       std::vector<float>& confidences, std::vector<int>& class_ids);
    cv::Rect convert_bbox_to_original_size(const cv::Rect& bbox, const cv::Size& original_size, const cv::Size& resized_size);
    void apply_nms(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<int>& class_ids);
};