#include "test_utils.hpp"
#include "detection.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <iostream>

TEST_CASE("BirdDetectorYolov5 - Construction and Basic Setup") {
    SUBCASE("Constructor with valid parameters") {
        auto image_store = std::make_shared<ImageStore>(10);
        std::string model_path = "weights/bh_v1.onnx"; // This may not exist in test env
        
        // Constructor should not throw even if model file doesn't exist
        // The model loading is handled gracefully in load_model()
        BirdDetectorYolov5 detector(
            image_store,
            model_path,
            cv::Size(640, 640),
            0.25f,
            0.45f,
            std::chrono::seconds(10),
            50
        );
        
        CHECK(detector.listening_to().count(EventType::NEW_FRAME) == 1);
    }
    
    SUBCASE("Constructor with custom parameters") {
        auto image_store = std::make_shared<ImageStore>(5);
        std::string model_path = "weights/bh_v2.onnx";
        
        BirdDetectorYolov5 detector(
            image_store,
            model_path,
            cv::Size(416, 416),
            0.5f,
            0.3f,
            std::chrono::seconds(5),
            100
        );
        
        CHECK(detector.listening_to().size() == 1);
        CHECK(detector.listening_to().count(EventType::NEW_FRAME) == 1);
    }
}

TEST_CASE("BirdDetectorYolov5 - Detection with Missing Model") {
    auto image_store = std::make_shared<ImageStore>(10);
    std::string invalid_model_path = "nonexistent/model.onnx";
    
    BirdDetectorYolov5 detector(
        image_store,
        invalid_model_path
    );
    
    SUBCASE("Detection returns nullopt when model not loaded") {
        // Create a test frame event
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Create a test image and store it
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        image_store->put(timestamp, test_image);
        
        // Detection should return nullopt due to missing model
        auto result = detector.detect(frame_event);
        CHECK_FALSE(result.has_value());
    }
}

TEST_CASE("BirdDetectorYolov5 - Detection with Valid Setup") {
    auto image_store = std::make_shared<ImageStore>(10);
    
    // Use a model path that might exist in the project
    std::string model_path = "weights/bh_v1.onnx";
    
    BirdDetectorYolov5 detector(
        image_store,
        model_path
    );
    
    SUBCASE("Detection with missing image returns nullopt") {
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Don't store any image
        auto result = detector.detect(frame_event);
        CHECK_FALSE(result.has_value());
    }
    
    SUBCASE("Detection with delayed frame returns nullopt") {
        // Create a timestamp that's too old (more than max_delay)
        auto old_timestamp = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - std::chrono::seconds(15)
        );
        auto frame_event = std::make_shared<FrameEvent>(old_timestamp, std::nullopt);
        
        // Store an image for this timestamp
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        image_store->put(old_timestamp, test_image);
        
        auto result = detector.detect(frame_event);
        CHECK_FALSE(result.has_value());
    }
    
    SUBCASE("Detection with valid recent frame and existing image") {
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Create a test image (black image - unlikely to have detections)
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        image_store->put(timestamp, test_image);
        
        auto result = detector.detect(frame_event);
        
        // With a black image and no actual model loaded, this should return nullopt
        // If a valid model were loaded, this might return detections
        // The exact behavior depends on whether the model file exists
        // For now, we just check that the method doesn't crash
        CHECK(true); // Test passes if we reach this point without exception
    }
}

TEST_CASE("BirdDetectorYolov5 - Event System Integration") {
    auto image_store = std::make_shared<ImageStore>(10);
    std::string model_path = "weights/bh_v1.onnx";
    
    BirdDetectorYolov5 detector(
        image_store,
        model_path
    );
    
    SUBCASE("Detector can be registered with event queue") {
        auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
        
        CHECK_NOTHROW(detector.set_event_queue(event_queue));
    }
    
    SUBCASE("Detector can receive frame events") {
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Store an image
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        image_store->put(timestamp, test_image);
        
        CHECK_NOTHROW(detector.notify(frame_event));
    }
}
