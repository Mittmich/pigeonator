#include "test_utils.hpp"
#include "detection.hpp"
#include "orchestration.hpp"
#include "recorder.hpp"
#include "video.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>

// Video file capture implementation for testing
class VideoFileCapture : public CameraCapture {
public:
    VideoFileCapture(const std::string& video_path) : video_path_(video_path), frame_index_(0) {
        cap_.open(video_path);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Could not open video file: " + video_path);
        }
    }
    
    ~VideoFileCapture() override {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }
    
    cv::Mat getNextFrame() override {
        cv::Mat frame;
        if (cap_.read(frame)) {
            frame_index_++;
            return frame;
        }
        return cv::Mat(); // Return empty frame when video ends
    }
    
    void startStreaming() override {
        // Video file is already "streaming"
    }
    
    void stopStreaming() override {
        // Nothing special needed for video file
    }

private:
    std::string video_path_;
    cv::VideoCapture cap_;
    int frame_index_;
};

// Helper functions for MockSubscriber testing
size_t get_event_count_by_type(const MockSubscriber& subscriber, EventType type) {
    return std::count_if(subscriber.received_events.begin(), subscriber.received_events.end(),
                       [type](const std::shared_ptr<Event>& event) {
                           return event->type == type;
                       });
}

std::vector<std::string> get_all_detection_labels(const MockSubscriber& subscriber) {
    std::vector<std::string> all_labels;
    for (const auto& event : subscriber.received_events) {
        if (event->type == EventType::DETECTION) {
            // Cast to DetectionEvent - we know it's safe because we checked the type
            auto detection_event = std::static_pointer_cast<DetectionEvent>(event);
            auto detections = detection_event->get_detections();
            for (const auto& detection : detections) {
                auto labels = detection.get_labels();
                if (labels.has_value() && !labels.value().empty()) {
                    all_labels.insert(all_labels.end(), 
                                    labels.value().begin(), labels.value().end());
                }
            }
        }
    }
    return all_labels;
}

bool verify_event_sequence(const MockSubscriber& subscriber, const std::vector<EventType>& expected_sequence) {
    if (subscriber.received_events.size() < expected_sequence.size()) {
        return false;
    }
    
    for (size_t i = 0; i < expected_sequence.size(); ++i) {
        if (subscriber.received_events[i]->type != expected_sequence[i]) {
            return false;
        }
    }
    return true;
}

void print_event_summary(const MockSubscriber& subscriber) {
    std::cout << "=== Event Summary ===" << std::endl;
    std::cout << "Total events: " << subscriber.received_events.size() << std::endl;
    std::cout << "Frame events: " << get_event_count_by_type(subscriber, EventType::NEW_FRAME) << std::endl;
    std::cout << "Detection events: " << get_event_count_by_type(subscriber, EventType::DETECTION) << std::endl;
    std::cout << "Effector events: " << get_event_count_by_type(subscriber, EventType::EFFECTOR_ACTION) << std::endl;
    
    auto labels = get_all_detection_labels(subscriber);
    if (!labels.empty()) {
        std::cout << "Detection labels: ";
        for (const auto& label : labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;
    }
}

// Helper function to create a test video file with motion and birds
std::string create_test_video_with_motion_and_birds(const std::string& filename, int frames = 60) {
    std::string full_path = "/tmp/" + filename;
    
    // Create a video writer
    cv::VideoWriter writer(full_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10.0, cv::Size(640, 480));
    
    if (!writer.isOpened()) {
        throw std::runtime_error("Could not create test video file: " + full_path);
    }
    
    for (int i = 0; i < frames; ++i) {
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(50, 100, 50)); // Green background
        
        // Add some motion and bird-like objects at different times
        if (i >= 10 && i <= 20) {
            // First motion sequence - moving object that's not a bird
            int x = 100 + (i - 10) * 20;
            cv::rectangle(frame, cv::Point(x, 100), cv::Point(x + 30, 130), cv::Scalar(200, 100, 100), -1);
        }
        
        if (i >= 25 && i <= 45) {
            // Second motion sequence - bird-like object
            int x = 200 + (i - 25) * 10;
            int y = 150 + sin((i - 25) * 0.3) * 20; // Slight vertical movement
            
            // Draw bird-like shape
            cv::ellipse(frame, cv::Point(x, y), cv::Size(25, 15), 0, 0, 360, cv::Scalar(80, 80, 120), -1);
            cv::circle(frame, cv::Point(x - 8, y - 3), 3, cv::Scalar(60, 60, 100), -1); // Eye
            cv::circle(frame, cv::Point(x + 12, y), 2, cv::Scalar(100, 90, 80), -1);    // Beak
        }
        
        if (i >= 50) {
            // Third motion sequence - another bird
            int x = 400;
            int y = 200 + (i - 50) * 5;
            cv::ellipse(frame, cv::Point(x, y), cv::Size(20, 12), 0, 0, 360, cv::Scalar(70, 70, 110), -1);
        }
        
        writer.write(frame);
    }
    
    writer.release();
    return full_path;
}

TEST_CASE("E2E Integration Test - Motion Activated Bird Detection Pipeline") {
    
    SUBCASE("Test bird detections emitted") {
        // This subcase will be used when the user provides actual video files
        std::string test_dir = "tests/e2e_video_output/";
        // Placeholder path - user will supply this later
        std::string user_video_path = "tests/test_videos/20250813_173458.mp4";
        
        INFO("Placeholder for user-supplied video: " << user_video_path);
        
        // Skip this test if user video doesn't exist yet
        if (!std::filesystem::exists(user_video_path)) {
            INFO("User-supplied video not found, skipping test");
            return;
        }
        
        // Set up same pipeline as above but with user video
        auto image_store = std::make_shared<ImageStore>(800);
        
        try {
            auto video_capture = std::make_unique<VideoFileCapture>(user_video_path);
            auto video_stream = std::make_shared<Stream>(image_store, video_capture.get());
            
            // Same setup as previous test...
            auto motion_detector = std::make_shared<MotionDetector>(
                image_store, 24, 21, 5, 100, 0, std::chrono::seconds(100)
            );
            
            auto bird_detector = std::make_shared<BirdDetectorYolov5>(
                image_store, "weights/bh_v3.onnx", cv::Size(640, 640), 
                0.25f, 0.45f, std::chrono::seconds(500), 50
            );
            
            auto video_recorder = std::make_shared<EventRecorder>(
                std::set<EventType>({EventType::NEW_FRAME, EventType::DETECTION}),
                image_store,
                test_dir,
                300,
                10, 
                300
            );

            auto mock_subscriber = std::make_shared<MockSubscriber>();
            
            VideoEventManager event_manager(*video_stream);
            event_manager.add_subscriber(bird_detector);
            event_manager.add_subscriber(video_recorder);
            event_manager.add_subscriber(mock_subscriber);
            
            // Run pipeline for user video
            std::thread pipeline_thread([&event_manager]() {
                event_manager.run();
            });
            
            // Let it run longer for user videos which might be longer
            std::this_thread::sleep_for(std::chrono::seconds(5));
            
            event_manager.stop();

            if (pipeline_thread.joinable()) {
                pipeline_thread.join();
            }
            // Let it run longer for user videos which might be longer
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // Verify results
            print_event_summary(*mock_subscriber);
            CHECK(mock_subscriber->received_events.size() > 0);
            // Check that bird detections were emitted
            CHECK(get_event_count_by_type(*mock_subscriber, EventType::DETECTION) > 0);
            // Check that there was a pigeon detected
            auto labels = get_all_detection_labels(*mock_subscriber);
            CHECK(std::find(labels.begin(), labels.end(), "Pigeon") != labels.end());

            INFO("User video processing completed successfully");
            
        } catch (const std::exception& e) {
            FAIL("Failed to process user video: " << e.what());
        }
    }
    
    // Clean up test directory
    //if (std::filesystem::exists(test_dir)) {
    //    std::filesystem::remove_all(test_dir);
    //}
}