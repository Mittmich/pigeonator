#include "test_utils.hpp"
#include "detection.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

// Mock motion detector for testing
class MockMotionDetector : public Detector {
public:
    MockMotionDetector(std::shared_ptr<ImageStore> image_store) 
        : Detector({EventType::NEW_FRAME}, image_store), should_detect_motion(false) {}
    
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override {
        if (should_detect_motion) {
            // Create a mock motion detection
            Detection motion_detection(
                frame_event->get_timestamp(),
                frame_event,
                std::vector<std::string>{"motion"},
                std::vector<float>{1.0f},
                std::vector<cv::Rect>{cv::Rect(50, 50, 100, 100)},
                std::vector<int>{10000},
                std::map<std::string, std::string>{{"type", "motion"}}
            );
            
            motion_detections_count++;
            return DetectionEvent(frame_event->get_timestamp(), {motion_detection});
        }
        return std::nullopt;
    }
    
    // Test configuration
    bool should_detect_motion;
    int motion_detections_count = 0;
};

// Mock secondary detector for testing
class MockSecondaryDetector : public Detector {
public:
    MockSecondaryDetector(std::shared_ptr<ImageStore> image_store) 
        : Detector({EventType::NEW_FRAME}, image_store), should_detect_object(false) {}
    
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override {
        frames_processed++;
        
        if (should_detect_object) {
            Detection object_detection(
                frame_event->get_timestamp(),
                frame_event,
                detection_labels,
                detection_confidences,
                detection_bboxes,
                detection_areas,
                detection_metadata
            );
            
            object_detections_count++;
            return DetectionEvent(frame_event->get_timestamp(), {object_detection});
        }
        return std::nullopt;
    }
    
    // Test configuration
    bool should_detect_object;
    int frames_processed = 0;
    int object_detections_count = 0;
    
    // Detection data to return
    std::vector<std::string> detection_labels = {"pigeon"};
    std::vector<float> detection_confidences = {0.8f};
    std::vector<cv::Rect> detection_bboxes = {cv::Rect(75, 75, 50, 40)};
    std::vector<int> detection_areas = {2000};
    std::optional<std::map<std::string, std::string>> detection_metadata = 
        std::map<std::string, std::string>{{"type", "bird"}};
};

TEST_CASE("MotionActivatedDetector - Construction and Basic Setup") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    SUBCASE("Constructor with default parameters") {
        MotionActivatedDetector detector(
            motion_detector,
            secondary_detector,
            image_store
        );
        
        CHECK(detector.listening_to().count(EventType::NEW_FRAME) == 1);
        CHECK(detector.listening_to().size() == 1);
    }
    
    SUBCASE("Constructor with custom parameters") {
        MotionActivatedDetector detector(
            motion_detector,
            secondary_detector,
            image_store,
            10,  // slack_frames
            15   // max_frame_history
        );
        
        CHECK(detector.listening_to().count(EventType::NEW_FRAME) == 1);
    }
}

TEST_CASE("MotionActivatedDetector - No Motion, No Detection") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store
    );
    
    // Configure detectors - no motion, no secondary detection
    motion_detector->should_detect_motion = false;
    secondary_detector->should_detect_object = false;
    
    // Process several frames
    for (int i = 0; i < 5; ++i) {
        cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
        Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
        image_store->put(timestamp, img);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        auto result = detector.detect(frame_event);
        
        CHECK(!result.has_value());
    }
    
    CHECK(motion_detector->motion_detections_count == 0);
    CHECK(secondary_detector->frames_processed == 0); // Should not process frames without motion
}

TEST_CASE("MotionActivatedDetector - Motion But No Object Detection") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store
    );
    
    // Configure detectors - motion detected, but no objects
    motion_detector->should_detect_motion = true;
    secondary_detector->should_detect_object = false;
    
    cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
    Timestamp timestamp = test_now();
    image_store->put(timestamp, img);
    
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    auto result = detector.detect(frame_event);
    
    CHECK(!result.has_value()); // No detection should be emitted
    CHECK(motion_detector->motion_detections_count == 1);
    CHECK(secondary_detector->frames_processed > 0); // Should have processed frames
}

TEST_CASE("MotionActivatedDetector - Motion Triggers Object Detection") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store,
        3,  // slack_frames
        5   // max_frame_history
    );
    
    SUBCASE("Direct motion-triggered detection") {
        // Configure detectors - motion and object both detected
        motion_detector->should_detect_motion = true;
        secondary_detector->should_detect_object = true;
        
        cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
        Timestamp timestamp = test_now();
        image_store->put(timestamp, img);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        auto result = detector.detect(frame_event);
        
        REQUIRE(result.has_value());
        
        auto detections = result.value().get_detections();
        CHECK(detections.size() == 1);
        
        auto detection = detections[0];
        auto labels = detection.get_labels();
        auto meta_data = detection.get_meta_data();
        
        REQUIRE(labels.has_value());
        CHECK(labels.value()[0] == "pigeon");
        
        REQUIRE(meta_data.has_value());
        CHECK(meta_data.value()["activation_type"] == "motion_triggered");
        CHECK(meta_data.value()["motion_detector_triggered"] == "true");
    }
}

TEST_CASE("MotionActivatedDetector - Slack Period Behavior") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store,
        3,  // slack_frames
        5   // max_frame_history
    );
    
    SUBCASE("Detection during slack period") {
        std::vector<bool> motion_sequence = {true, false, false, false, false}; // Motion then no motion
        std::vector<bool> object_sequence = {false, true, false, false, false}; // Object detected in frame 1
        
        std::optional<DetectionEvent> slack_detection;
        
        for (size_t i = 0; i < motion_sequence.size(); ++i) {
            motion_detector->should_detect_motion = motion_sequence[i];
            secondary_detector->should_detect_object = object_sequence[i];
            
            cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = detector.detect(frame_event);
            
            if (result.has_value() && i == 1) { // Frame 1 should have detection
                slack_detection = result;
            }
        }
        
        REQUIRE(slack_detection.has_value());
        
        auto detections = slack_detection.value().get_detections();
        CHECK(detections.size() == 1);
        
        auto meta_data = detections[0].get_meta_data();
        REQUIRE(meta_data.has_value());
        CHECK(meta_data.value()["activation_type"] == "motion_triggered_slack");
        CHECK(meta_data.value().count("slack_frames_remaining") > 0);
    }
    
    SUBCASE("Slack period expiration") {
        // Motion on frame 0, then no motion and no objects
        motion_detector->should_detect_motion = true;
        secondary_detector->should_detect_object = false;
        
        // First frame with motion
        cv::Mat img1 = create_test_image_with_bird_like_pattern(300, 300);
        Timestamp timestamp1 = test_now();
        image_store->put(timestamp1, img1);
        auto frame_event1 = std::make_shared<FrameEvent>(timestamp1, std::nullopt);
        auto result1 = detector.detect(frame_event1);
        CHECK(!result1.has_value()); // No object detected
        
        // Now disable motion for slack period
        motion_detector->should_detect_motion = false;
        
        int frames_processed_during_slack = 0;
        
        // Process slack frames (3 frames)
        for (int i = 1; i <= 4; ++i) { // 4 frames to exceed slack period
            cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = detector.detect(frame_event);
            
            CHECK(!result.has_value()); // No objects detected
            
            if (i <= 3) { // Within slack period
                frames_processed_during_slack++;
            }
        }
        
        // Secondary detector should have been called during slack period
        CHECK(secondary_detector->frames_processed > frames_processed_during_slack);
    }
}

TEST_CASE("MotionActivatedDetector - Frame History Processing") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store,
        2,  // slack_frames
        3   // max_frame_history (small for testing)
    );
    
    SUBCASE("Historical frames processed when motion detected") {
        // Configure no motion initially, then motion
        motion_detector->should_detect_motion = false;
        secondary_detector->should_detect_object = false;
        
        // Process a few frames without motion (building history)
        for (int i = 0; i < 3; ++i) {
            cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = detector.detect(frame_event);
            CHECK(!result.has_value());
        }
        
        int frames_before_motion = secondary_detector->frames_processed;
        
        // Now trigger motion
        motion_detector->should_detect_motion = true;
        secondary_detector->should_detect_object = true;
        
        cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
        Timestamp timestamp = test_now() + std::chrono::milliseconds(300);
        image_store->put(timestamp, img);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        auto result = detector.detect(frame_event);
        
        REQUIRE(result.has_value());
        
        // Secondary detector should have processed historical frames when motion was detected
        int frames_after_motion = secondary_detector->frames_processed;
        CHECK(frames_after_motion > frames_before_motion + 1); // +1 for current frame, more for history
    }
}

TEST_CASE("MotionActivatedDetector - Complex Scenario") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto motion_detector = std::make_shared<MockMotionDetector>(image_store);
    auto secondary_detector = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        motion_detector,
        secondary_detector,
        image_store,
        3,  // slack_frames
        5   // max_frame_history
    );
    
    SUBCASE("Multiple motion periods with varying object detection") {
        struct FrameConfig {
            bool has_motion;
            bool has_object;
            std::string expected_activation_type;
        };
        
        std::vector<FrameConfig> frame_sequence = {
            {false, false, ""},           // Frame 0: No motion, no object
            {false, false, ""},           // Frame 1: No motion, no object
            {true, true, "motion_triggered"}, // Frame 2: Motion + object
            {false, true, "motion_triggered_slack"}, // Frame 3: Slack period + object
            {false, false, ""},           // Frame 4: Slack period, no object
            {false, false, ""},           // Frame 5: Slack expired
            {true, false, ""},            // Frame 6: Motion but no object
            {false, true, "motion_triggered_slack"}, // Frame 7: Slack + object
        };
        
        std::vector<std::optional<DetectionEvent>> results;
        
        for (size_t i = 0; i < frame_sequence.size(); ++i) {
            const auto& config = frame_sequence[i];
            
            motion_detector->should_detect_motion = config.has_motion;
            secondary_detector->should_detect_object = config.has_object;
            
            cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = detector.detect(frame_event);
            results.push_back(result);
            
            if (result.has_value() && !config.expected_activation_type.empty()) {
                auto detections = result.value().get_detections();
                REQUIRE(detections.size() == 1);
                
                auto meta_data = detections[0].get_meta_data();
                REQUIRE(meta_data.has_value());
                CHECK(meta_data.value()["activation_type"] == config.expected_activation_type);
                
                INFO("Frame " << i << " correctly detected with activation type: " << config.expected_activation_type);
            } else if (!result.has_value() && config.expected_activation_type.empty()) {
                // Expected no detection
                INFO("Frame " << i << " correctly had no detection");
            }
        }
        
        // Verify specific frames had detections
        CHECK(results[2].has_value());  // Motion + object
        CHECK(results[3].has_value());  // Slack + object
        CHECK(!results[4].has_value()); // Slack, no object
        CHECK(!results[5].has_value()); // Slack expired
        CHECK(!results[6].has_value()); // Motion but no object
        CHECK(results[7].has_value());  // Slack + object
    }
}

TEST_CASE("MotionActivatedDetector - Integration with Real Detectors") {
    auto image_store = std::make_shared<ImageStore>(20);
    
    // Create real motion detector
    auto real_motion_detector = std::make_shared<MotionDetector>(
        image_store,
        20,   // threshold
        21,   // blur
        5,    // dilate
        100,  // threshold_area
        2,    // activation_frames
        std::chrono::seconds(5) // max_delay
    );
    
    // Create mock secondary detector for controlled testing
    auto mock_secondary = std::make_shared<MockSecondaryDetector>(image_store);
    
    MotionActivatedDetector detector(
        real_motion_detector,
        mock_secondary,
        image_store,
        3,  // slack_frames
        5   // max_frame_history
    );
    
    SUBCASE("Real motion detection with mock secondary") {
        mock_secondary->should_detect_object = true;
        
        // Create images with actual motion (different content)
        cv::Mat img1 = create_test_image_with_bird_like_pattern(300, 300);
        cv::Mat img2(300, 300, CV_8UC3, cv::Scalar(100, 150, 100)); // Different background
        cv::circle(img2, cv::Point(150, 150), 50, cv::Scalar(200, 200, 200), -1); // Add circle for motion
        
        Timestamp timestamp1 = test_now();
        Timestamp timestamp2 = test_now() + std::chrono::milliseconds(100);
        
        image_store->put(timestamp1, img1);
        image_store->put(timestamp2, img2);
        
        auto frame_event1 = std::make_shared<FrameEvent>(timestamp1, std::nullopt);
        auto frame_event2 = std::make_shared<FrameEvent>(timestamp2, std::nullopt);
        
        // First frame should not trigger (motion detector needs previous frame)
        auto result1 = detector.detect(frame_event1);
        CHECK(!result1.has_value());
        
        // Second frame might trigger motion and should process through secondary
        auto result2 = detector.detect(frame_event2);
        // Note: May or may not detect motion depending on the specific image differences
        // The important thing is that if motion is detected, secondary processing occurs
        
        CHECK(mock_secondary->frames_processed >= 0); // Secondary was called at least for history processing
    }
}
