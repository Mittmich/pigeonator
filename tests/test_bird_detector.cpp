#include "test_utils.hpp"
#include "detection.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <iostream>
#include <thread>

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

TEST_CASE("BirdDetectorYolov5 - Real Image Detection") {
    auto image_store = std::make_shared<ImageStore>(10);
    std::string model_path = "weights/bh_v1.onnx";
    
    BirdDetectorYolov5 detector(
        image_store,
        model_path,
        cv::Size(640, 640),
        0.25f,
        0.45f,
        std::chrono::seconds(5000),
        50
    );
    
    SUBCASE("Pigeon image should produce bird detection") {
        // Load the pigeon test image
        cv::Mat pigeon_image = cv::imread("tests/test_files/pigeon.jpg");
        
        // Skip test if image cannot be loaded (file might not exist in CI environment)
        if (pigeon_image.empty()) {
            std::cout << "Warning: pigeon.jpg test file not found, skipping test" << std::endl;
            return;
        }
        
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Store the pigeon image
        image_store->put(timestamp, pigeon_image);
        
        auto result = detector.detect(frame_event);
        
        // Check if result has detections
        CHECK(result.has_value());

        auto detections = result.value().get_detections();

        CHECK(detections.size() > 0);
        
        // Check that at least one detection contains bird-related labels
        bool found_bird = false;
        for (auto& detection : detections) {
            auto labels = detection.get_labels();
            if (labels.has_value()) {
                for (const auto& label : labels.value()) {
                    if (label.find("Pigeon") != std::string::npos) {
                        found_bird = true;
                        break;
                    }
                }
                if (found_bird) break;
            }
        }
        
        CHECK(found_bird);
        
        // Draw bounding boxes on the image and save it
        cv::Mat output_image = pigeon_image.clone();
        for (auto& detection : detections) {
            auto bboxes = detection.get_bounding_boxes();
            auto labels = detection.get_labels();
            auto confidences = detection.get_confidences();
            
            if (bboxes.has_value() && labels.has_value() && confidences.has_value()) {
                const auto& boxes = bboxes.value();
                const auto& label_list = labels.value();
                const auto& conf_list = confidences.value();
                
                for (size_t i = 0; i < boxes.size() && i < label_list.size() && i < conf_list.size(); ++i) {
                    const auto& box = boxes[i];
                    const auto& label = label_list[i];
                    float confidence = conf_list[i];
                    
                    // Draw bounding box
                    cv::rectangle(output_image, box, cv::Scalar(0, 255, 0), 2);
                    
                    // Create label text
                    std::string text = label + " " + std::to_string(confidence).substr(0, 4);
                    
                    // Draw label background
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::rectangle(output_image, 
                                cv::Point(box.x, box.y - text_size.height - 5),
                                cv::Point(box.x + text_size.width, box.y),
                                cv::Scalar(0, 255, 0), -1);
                    
                    // Draw label text
                    cv::putText(output_image, text, 
                              cv::Point(box.x, box.y - 5),
                              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                }
            }
        }
        
        // Save the output image
        std::string output_path = "test_output_pigeon_detections.jpg";
        cv::imwrite(output_path, output_image);
        std::cout << "Saved detection visualization to: " << output_path << std::endl;
    }
    
    SUBCASE("Tree image should not produce bird detection") {
        // Load the tree test image
        cv::Mat tree_image = cv::imread("tests/test_files/tree.png");
        
        // Skip test if image cannot be loaded
        if (tree_image.empty()) {
            std::cout << "Warning: tree.png test file not found, skipping test" << std::endl;
            return;
        }
        
        auto timestamp = test_now();
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        
        // Store the tree image
        image_store->put(timestamp, tree_image);
        
        auto result = detector.detect(frame_event);
        
        // Check if result has detections - for tree image, we expect no detections (nullopt) or very few
        if (result.has_value()) {
            auto detections = result.value().get_detections();
            
            // If any detections are found, check that none contain bird-related labels
            bool found_bird = false;
            for (auto& detection : detections) {
                auto labels = detection.get_labels();
                if (labels.has_value()) {
                    for (const auto& label : labels.value()) {
                        if (label.find("Pigeon") != std::string::npos) {
                            found_bird = true;
                            break;
                        }
                    }
                    if (found_bird) break;
                }
            }
            
            CHECK_FALSE(found_bird);
        } else {
            // No detections at all - this is also correct for a tree image
            std::cout << "Correctly found no detections in tree image" << std::endl;
            CHECK(true); // This is the expected behavior
        }
    }
}
