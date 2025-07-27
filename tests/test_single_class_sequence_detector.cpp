#include "test_utils.hpp"
#include "detection.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>

// Mock detector for testing sequence detection logic
class MockDetector : public Detector {
public:
    MockDetector(std::shared_ptr<ImageStore> image_store) 
        : Detector({EventType::NEW_FRAME}, image_store), detection_counter(0) {}
    
    std::optional<DetectionEvent> detect(std::shared_ptr<FrameEvent> frame_event) override {
        if (should_return_detection) {
            std::vector<Detection> detections;
            
            for (const auto& detection_data : mock_detections) {
                Detection detection(
                    frame_event->get_timestamp(),
                    frame_event,
                    detection_data.labels,
                    detection_data.confidences,
                    detection_data.bboxes,
                    detection_data.areas,
                    detection_data.meta_data
                );
                detections.push_back(detection);
            }
            
            if (!detections.empty()) {
                detection_counter++;
                return DetectionEvent(frame_event->get_timestamp(), detections);
            }
        }
        return std::nullopt;
    }
    
    // Test configuration
    bool should_return_detection = false;
    int detection_counter = 0;
    
    struct MockDetectionData {
        std::vector<std::string> labels;
        std::vector<float> confidences;
        std::vector<cv::Rect> bboxes;
        std::vector<int> areas;
        std::optional<std::map<std::string, std::string>> meta_data;
    };
    
    std::vector<MockDetectionData> mock_detections;
    
    void add_mock_detection(const std::string& label, float confidence, const cv::Rect& bbox) {
        MockDetectionData data;
        data.labels = {label};
        data.confidences = {confidence};
        data.bboxes = {bbox};
        data.areas = {bbox.area()};
        data.meta_data = std::map<std::string, std::string>{{"type", "mock"}};
        mock_detections.push_back(data);
    }
    
    void clear_mock_detections() {
        mock_detections.clear();
    }
};

// Helper function to create test images from actual files
cv::Mat load_test_image(const std::string& filename) {
    std::string full_path = "tests/test_files/" + filename;
    cv::Mat img = cv::imread(full_path);
    if (img.empty()) {
        // Fallback to creating a synthetic image if file not found
        img = create_test_image_with_bird_like_pattern(300, 300);
    }
    return img;
}

// Create synthetic images that simulate different scenarios
cv::Mat create_pigeon_track_image(int x, int y, int frame_num) {
    cv::Mat img(480, 640, CV_8UC3, cv::Scalar(50, 100, 50)); // Green background
    
    // Draw a pigeon-like shape (ellipse) at position (x, y)
    cv::Point center(x, y);
    cv::Size axes(20 + frame_num % 5, 15 + frame_num % 3); // Slight size variation
    cv::Scalar color(80, 80, 120); // Grayish color for pigeon
    
    cv::ellipse(img, center, axes, 0, 0, 360, color, -1);
    
    // Add some texture
    cv::circle(img, cv::Point(x-5, y-3), 3, cv::Scalar(60, 60, 100), -1); // Eye
    cv::circle(img, cv::Point(x+8, y), 2, cv::Scalar(100, 90, 80), -1);   // Beak
    
    return img;
}

TEST_CASE("SingleClassSequenceDetector - Construction and Basic Setup") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SUBCASE("Constructor with default parameters") {
        SingleClassSequenceDetector sequence_detector(
            mock_detector,
            image_store
        );
        
        CHECK(sequence_detector.listening_to().count(EventType::NEW_FRAME) == 1);
        CHECK(sequence_detector.listening_to().size() == 1);
    }
    
    SUBCASE("Constructor with custom parameters") {
        SingleClassSequenceDetector sequence_detector(
            mock_detector,
            image_store,
            10,     // minimum_number_detections
            0.5f,   // iou_threshold
            3,      // max_frames_without_detection
            150.0f  // max_path_length_threshold
        );
        
        CHECK(sequence_detector.listening_to().count(EventType::NEW_FRAME) == 1);
    }
}

TEST_CASE("SingleClassSequenceDetector - No Detections from Base Detector") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        5  // minimum_number_detections
    );
    
    // Configure mock detector to return no detections
    mock_detector->should_return_detection = false;
    
    // Create test frame
    cv::Mat img = create_test_image_with_bird_like_pattern(300, 300);
    Timestamp timestamp = test_now();
    image_store->put(timestamp, img);
    
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    auto result = sequence_detector.detect(frame_event);
    
    CHECK(!result.has_value());
    CHECK(mock_detector->detection_counter == 0);
}

TEST_CASE("SingleClassSequenceDetector - Single Track Consensus") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        3  // minimum_number_detections for faster testing
    );
    
    mock_detector->should_return_detection = true;
    
    SUBCASE("Consistent pigeon detections should reach consensus") {
        // Simulate a pigeon moving across frames
        std::vector<cv::Point> track_positions = {
            {100, 100}, {110, 105}, {120, 110}, {130, 115}, {140, 120}
        };
        
        std::optional<DetectionEvent> consensus_result;
        
        for (size_t i = 0; i < track_positions.size(); ++i) {
            // Create frame with pigeon at specific position
            cv::Mat img = create_pigeon_track_image(track_positions[i].x, track_positions[i].y, i);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            // Configure mock detection for this frame
            mock_detector->clear_mock_detections();
            cv::Rect bbox(track_positions[i].x - 15, track_positions[i].y - 10, 30, 20);
            mock_detector->add_mock_detection("pigeon", 0.8f + (i * 0.02f), bbox);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            
            if (result.has_value()) {
                consensus_result = result;
                break;
            }
        }
        
        REQUIRE(consensus_result.has_value());
        
        auto detections = consensus_result.value().get_detections();
        CHECK(detections.size() == 1);
        
        auto detection = detections[0];
        auto labels = detection.get_labels();
        auto meta_data = detection.get_meta_data();
        
        REQUIRE(labels.has_value());
        CHECK(labels.value()[0] == "pigeon");
        
        REQUIRE(meta_data.has_value());
        CHECK(meta_data.value()["detector_type"] == "SingleClassSequenceDetector");
        CHECK(meta_data.value()["most_likely_object"] == "pigeon");
        CHECK(meta_data.value()["detection_type"] == "track_consensus");
    }
    
    SUBCASE("Mixed class detections should converge to highest confidence") {
        std::vector<std::pair<std::string, float>> class_sequence = {
            {"pigeon", 0.6f}, {"crow", 0.4f}, {"pigeon", 0.7f}, {"pigeon", 0.8f}, {"crow", 0.3f}
        };
        
        std::optional<DetectionEvent> consensus_result;
        
        for (size_t i = 0; i < class_sequence.size(); ++i) {
            cv::Mat img = create_pigeon_track_image(100 + i * 5, 100 + i * 2, i);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            cv::Rect bbox(95 + i * 5, 90 + i * 2, 30, 20);
            mock_detector->add_mock_detection(class_sequence[i].first, class_sequence[i].second, bbox);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            
            if (result.has_value()) {
                consensus_result = result;
                break;
            }
        }
        
        REQUIRE(consensus_result.has_value());
        
        auto detections = consensus_result.value().get_detections();
        auto detection = detections[0];
        auto labels = detection.get_labels();
        
        REQUIRE(labels.has_value());
        CHECK(labels.value()[0] == "pigeon"); // Should be pigeon due to higher cumulative confidence
    }
}

TEST_CASE("SingleClassSequenceDetector - Multi-Track Scenario") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        3,      // minimum_number_detections
        0.3f,   // iou_threshold
        5,      // max_frames_without_detection
        50.0f   // max_path_length_threshold
    );
    
    mock_detector->should_return_detection = true;
    
    SUBCASE("Two separate tracks should be tracked independently") {
        std::vector<std::tuple<cv::Point, cv::Point, std::string, std::string>> frame_data = {
            // Frame 0: Two birds far apart
            {{100, 100}, {300, 200}, "pigeon", "crow"},
            // Frame 1: Both move slightly
            {{105, 105}, {305, 205}, "pigeon", "crow"},
            // Frame 2: Continue movement
            {{110, 110}, {310, 210}, "pigeon", "crow"},
            // Frame 3: Should trigger consensus for both tracks
            {{115, 115}, {315, 215}, "pigeon", "crow"},
        };
        
        std::vector<std::optional<DetectionEvent>> results;
        
        for (size_t i = 0; i < frame_data.size(); ++i) {
            auto [pos1, pos2, class1, class2] = frame_data[i];
            
            cv::Mat img(480, 640, CV_8UC3, cv::Scalar(50, 100, 50));
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            // Add two detections per frame
            mock_detector->add_mock_detection(class1, 0.8f, cv::Rect(pos1.x - 15, pos1.y - 10, 30, 20));
            mock_detector->add_mock_detection(class2, 0.7f, cv::Rect(pos2.x - 15, pos2.y - 10, 30, 20));
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            results.push_back(result);
        }
        
        // Check that we eventually get consensus results
        bool found_consensus = false;
        for (const auto& result : results) {
            if (result.has_value()) {
                found_consensus = true;
                auto detections = result.value().get_detections();
                CHECK(detections.size() >= 1);
                
                // Verify consensus metadata
                for (const auto& detection : detections) {
                    auto meta_data = detection.get_meta_data();
                    REQUIRE(meta_data.has_value());
                    CHECK(meta_data.value()["detection_type"] == "track_consensus");
                }
            }
        }
        CHECK(found_consensus);
    }
}

TEST_CASE("SingleClassSequenceDetector - Path Length Threshold") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        3,      // minimum_number_detections
        0.5f,   // iou_threshold
        5,      // max_frames_without_detection
        30.0f   // Small path length threshold
    );
    
    mock_detector->should_return_detection = true;
    
    SUBCASE("Large movement should reset track") {
        std::vector<cv::Point> positions = {
            {100, 100},  // Start position
            {105, 105},  // Small movement
            {200, 200},  // Large jump - should reset track
            {205, 205},  // Continue from new position
            {210, 210}   // Another small movement
        };
        
        int consensus_count = 0;
        
        for (size_t i = 0; i < positions.size(); ++i) {
            cv::Mat img = create_pigeon_track_image(positions[i].x, positions[i].y, i);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            cv::Rect bbox(positions[i].x - 15, positions[i].y - 10, 30, 20);
            mock_detector->add_mock_detection("pigeon", 0.8f, bbox);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            
            if (result.has_value()) {
                consensus_count++;
            }
        }
        
        // Due to the large jump resetting the track, we shouldn't get consensus
        // from the first few detections before the reset
        CHECK(consensus_count <= 1);
    }
}

TEST_CASE("SingleClassSequenceDetector - Frame Dropout") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        3,    // minimum_number_detections
        0.5f, // iou_threshold
        2,    // max_frames_without_detection (small value for testing)
        100.0f // max_path_length_threshold
    );
    
    SUBCASE("Track should be dropped after missing frames") {
        // Create detections with gaps
        std::vector<std::pair<bool, cv::Point>> frame_sequence = {
            {true, {100, 100}},   // Frame 0: Detection
            {true, {105, 105}},   // Frame 1: Detection
            {false, {110, 110}},  // Frame 2: No detection
            {false, {115, 115}},  // Frame 3: No detection  
            {false, {120, 120}},  // Frame 4: No detection (should drop track)
        };
        
        mock_detector->should_return_detection = true;
        std::vector<std::optional<DetectionEvent>> results;
        
        for (size_t i = 0; i < frame_sequence.size(); ++i) {
            auto [has_detection, pos] = frame_sequence[i];
            
            cv::Mat img = create_pigeon_track_image(pos.x, pos.y, i);
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 100);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            if (has_detection) {
                cv::Rect bbox(pos.x - 15, pos.y - 10, 30, 20);
                mock_detector->add_mock_detection("pigeon", 0.8f, bbox);
            }
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            results.push_back(result);

            if (result.has_value()) {
                auto detections = result.value().get_detections();
                INFO("  -> Consensus reached with " << detections.size() << " detections");
            } else {
                INFO("  -> No consensus yet");
            }
        }
        
        // Count consensus events
        int consensus_count = 0;
        for (const auto& result : results) {
            if (result.has_value()) {
                consensus_count++;
                auto detections = result.value().get_detections();
                INFO("Consensus reached with " << detections.size() << " detections");
            }
        }
        
        CHECK(consensus_count == 0);
        INFO("Total consensus events: " << consensus_count);
    }
}

TEST_CASE("SingleClassSequenceDetector - Real Image Integration") {
    auto image_store = std::make_shared<ImageStore>(20);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        3
    );
    
    mock_detector->should_return_detection = true;
    
    SUBCASE("Using actual test images") {
        // Load real test images
        cv::Mat pigeon_img = load_test_image("pigeon.jpg");
        cv::Mat tree_img = load_test_image("tree.png");
        
        // Simulate a sequence where pigeon appears in different positions
        std::vector<std::tuple<cv::Mat, std::string, float, cv::Rect>> sequence = {
            {pigeon_img, "pigeon", 0.9f, cv::Rect(50, 50, 100, 80)},
            {pigeon_img, "pigeon", 0.85f, cv::Rect(60, 55, 100, 80)},
            {tree_img, "sparrow", 0.3f, cv::Rect(70, 60, 20, 15)}, // Misclassification
            {pigeon_img, "pigeon", 0.88f, cv::Rect(70, 60, 100, 80)},
        };
        
        std::optional<DetectionEvent> consensus_result;
        
        for (size_t i = 0; i < sequence.size(); ++i) {
            auto [img, label, confidence, bbox] = sequence[i];
            
            Timestamp timestamp = test_now() + std::chrono::milliseconds(i * 200);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            mock_detector->add_mock_detection(label, confidence, bbox);
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            
            if (result.has_value()) {
                consensus_result = result;
                break;
            }
        }
        
        if (consensus_result.has_value()) {
            auto detections = consensus_result.value().get_detections();
            CHECK(detections.size() == 1);
            
            auto detection = detections[0];
            auto labels = detection.get_labels();
            auto meta_data = detection.get_meta_data();
            
            REQUIRE(labels.has_value());
            CHECK(labels.value()[0] == "pigeon"); // Should converge to pigeon
            
            REQUIRE(meta_data.has_value());
            CHECK(meta_data.value().count("track_id") > 0);
            CHECK(meta_data.value().count("total_detections_in_track") > 0);
        }
    }
}

TEST_CASE("SingleClassSequenceDetector - Stress Test with Many Tracks") {
    auto image_store = std::make_shared<ImageStore>(50);
    auto mock_detector = std::make_shared<MockDetector>(image_store);
    
    SingleClassSequenceDetector sequence_detector(
        mock_detector,
        image_store,
        4,      // minimum_number_detections
        0.3f,   // iou_threshold
        3,      // max_frames_without_detection
        100.0f  // max_path_length_threshold
    );
    
    mock_detector->should_return_detection = true;
    
    SUBCASE("Multiple simultaneous tracks") {
        const int num_tracks = 5;
        const int num_frames = 8;
        
        // Generate multiple tracks with different starting positions and classes
        std::vector<std::vector<std::tuple<cv::Point, std::string, float>>> tracks(num_tracks);
        std::vector<std::string> classes = {"pigeon", "crow", "sparrow", "blue_tit", "robin"};
        
        for (int track = 0; track < num_tracks; ++track) {
            int start_x = 50 + track * 100;
            int start_y = 50 + track * 50;
            std::string track_class = classes[track % classes.size()];
            
            for (int frame = 0; frame < num_frames; ++frame) {
                cv::Point pos(start_x + frame * 10, start_y + frame * 5);
                float confidence = 0.7f + (frame * 0.05f);
                tracks[track].push_back({pos, track_class, confidence});
            }
        }
        
        int total_consensus_events = 0;
        
        for (int frame = 0; frame < num_frames; ++frame) {
            cv::Mat img(600, 800, CV_8UC3, cv::Scalar(50, 100, 50));
            Timestamp timestamp = test_now() + std::chrono::milliseconds(frame * 100);
            image_store->put(timestamp, img);
            
            mock_detector->clear_mock_detections();
            
            // Add all active tracks for this frame
            for (int track = 0; track < num_tracks; ++track) {
                if (frame < static_cast<int>(tracks[track].size())) {
                    auto [pos, class_name, confidence] = tracks[track][frame];
                    cv::Rect bbox(pos.x - 20, pos.y - 15, 40, 30);
                    mock_detector->add_mock_detection(class_name, confidence, bbox);
                }
            }
            
            auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
            auto result = sequence_detector.detect(frame_event);
            
            if (result.has_value()) {
                auto detections = result.value().get_detections();
                total_consensus_events += detections.size();
                
                // Verify each consensus detection
                for (const auto& detection : detections) {
                    auto meta_data = detection.get_meta_data();
                    REQUIRE(meta_data.has_value());
                    CHECK(meta_data.value()["detection_type"] == "track_consensus");
                    CHECK(meta_data.value().count("track_id") > 0);
                }
            }
        }
        
        // Should eventually get consensus for multiple tracks
        CHECK(total_consensus_events >= 1);
        INFO("Total consensus events: " << total_consensus_events);
    }
}
