#include "events.hpp"
#include <opencv2/opencv.hpp>
#include <set>
#include <thread>
#include "mm.hpp"
#include <chrono>
#include <map>
#include <vector>
#include <memory>

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
    void load_class_names_from_model();
    std::vector<std::string> get_default_class_names();
    cv::Mat preprocess_image(const cv::Mat& image);
    std::vector<cv::Rect> extract_boxes(const std::vector<cv::Mat>& outputs, const cv::Size& original_size, 
                                       std::vector<float>& confidences, std::vector<int>& class_ids);
    cv::Rect convert_bbox_to_original_size(const cv::Rect& bbox, const cv::Size& original_size, const cv::Size& resized_size);
    void apply_nms(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<int>& class_ids);
};

// Forward declarations for object tracking
struct ClassStatistics;
struct Track;
class ObjectTracker;

// Class statistics for tracking class votes
struct ClassStatistics {
    float total_confidence;
    int detection_count;
    
    ClassStatistics();
    float get_average_confidence() const;
    float get_weighted_score() const;
};

// Track structure for object tracking
struct Track {
    int track_id;
    cv::Rect last_bbox;
    cv::Point2f last_center;
    std::map<std::string, ClassStatistics> class_votes;
    int frames_since_last_detection;
    int total_detections_in_track;
    std::vector<cv::Point2f> trajectory;
    Timestamp last_detection_time;
    
    Track(int id, const cv::Rect& bbox, Timestamp timestamp);
    std::string get_most_likely_class() const;
    bool has_reached_consensus(int minimum_detections) const;
    float get_mean_confidence_for_consensus_class() const;
};

// Object tracker for managing multiple tracks
class ObjectTracker {
public:
    ObjectTracker(
        float iou_threshold = 0.3f, 
        int max_frames_without_detection = 5, 
        float max_path_length_threshold = 100.0f
    );
    
    void update_tracks(const std::vector<Detection>& detections, Timestamp current_time);
    std::vector<Track> get_tracks_with_consensus(int minimum_detections) const;
    std::vector<Track> get_all_active_tracks() const;
    void remove_track(int track_id);
    void prune_tracks();
    
private:
    std::vector<Track> active_tracks;
    int next_track_id;
    float iou_threshold;
    int max_frames_without_detection;
    float max_path_length_threshold;
    
    float calculate_iou(const cv::Rect& box1, const cv::Rect& box2) const;
    float calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    void associate_detections_to_tracks(const std::vector<Detection>& detections, Timestamp current_time);
    void create_new_tracks(const std::vector<Detection>& unassociated_detections, Timestamp current_time);
    bool should_drop_track_for_path_length(Track& track, const cv::Point2f& new_center);
    void increment_frames_without_detection();
};

// Single class sequence detector with object tracking
class SingleClassSequenceDetector : public Detector {
public:
    SingleClassSequenceDetector(
        std::shared_ptr<Detector> base_detector,
        std::shared_ptr<ImageStore> image_store,
        int minimum_number_detections = 5,
        float iou_threshold = 0.3f,
        int max_frames_without_detection = 5,
        float max_path_length_threshold = 100.0f
    );
    ~SingleClassSequenceDetector();
    
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override;

private:
    std::shared_ptr<Detector> base_detector;
    int minimum_number_detections;
    std::unique_ptr<ObjectTracker> tracker;
    
    // Track-based detection storage
    std::map<int, std::vector<Detection>> detections_by_track;
    
    void process_consensus_tracks(const std::vector<Track>& consensus_tracks, Timestamp timestamp);
    std::vector<Detection> rewrite_track_detections_to_consensus(
        const std::vector<Detection>& track_detections, 
        const Track& consensus_track,
        Timestamp timestamp
    );
    void cleanup_completed_tracks(const std::vector<Track>& consensus_tracks);
    std::optional<std::map<std::string, std::string>> create_consensus_metadata(const Track& track) const;
};