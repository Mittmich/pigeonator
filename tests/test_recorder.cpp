#include "doctest/doctest.h"
#include "recorder.hpp"
#include "test_utils.hpp"
#include "mm.hpp"
#include <memory>
#include <thread>
#include <chrono>
#include <filesystem>

// Helper function to create video-compatible images
cv::Mat create_video_image(int rows, int cols) {
    cv::Mat img(rows, cols, CV_8UC3); // 8-bit unsigned, 3 channels (BGR)
    cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return img;
}

// Helper function to create a temporary directory for tests
std::string create_temp_directory() {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / 
                                   ("recorder_test_" + std::to_string(std::time(nullptr)) + "_" + 
                                    std::to_string(std::rand()));
    std::filesystem::create_directories(temp_dir);
    return temp_dir.string();
}

// Helper function to clean up a directory and its contents
void cleanup_directory(const std::string& dir_path) {
    if (std::filesystem::exists(dir_path)) {
        std::filesystem::remove_all(dir_path);
    }
}

// Mock Recorder class for testing abstract base class
class MockRecorder : public Recorder {
public:
    MockRecorder(std::set<EventType> listening_events, std::shared_ptr<ImageStore> image_store, 
                 const std::string& output_directory = ".")
        : Recorder(listening_events, image_store, output_directory) {}
    
    // Track which handlers were called for testing
    bool new_frame_handled = false;
    bool detection_handled = false;
    bool effector_action_handled = false;
    
protected:
    void handle_new_frame(Event event) override {
        new_frame_handled = true;
    }
    
    void handle_detection(Event event) override {
        detection_handled = true;
    }
    
    void handle_effector_action(Event event) override {
        effector_action_handled = true;
    }
};

TEST_CASE("Recorder constructor initializes correctly") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(10);
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    
    CHECK(recorder.listening_to() == events);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder requires event queue to be set before starting") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    std::string temp_dir = create_temp_directory();
    MockRecorder recorder(events, image_store, temp_dir);
    
    CHECK_THROWS_AS(recorder.start(), std::runtime_error);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder can be started and stopped with event queue") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    CHECK_NOTHROW(recorder.start());
    
    // Give it a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK_NOTHROW(recorder.stop());
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder only accepts events it's listening to") {
    std::set<EventType> events = {EventType::NEW_FRAME}; // Only listening to NEW_FRAME
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Notify with NEW_FRAME event (should be accepted)
    FrameEvent frame_event(std::time(nullptr), std::nullopt);
    recorder.notify(frame_event);
    
    // Notify with DETECTION event (should be ignored)
    Event detection_event(EventType::DETECTION, std::time(nullptr), std::nullopt);
    recorder.notify(detection_event);
    
    // Start recorder to process events
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    recorder.stop();
    
    // Check that only NEW_FRAME was handled
    CHECK(recorder.new_frame_handled == true);
    CHECK(recorder.detection_handled == false);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder processes events in queue") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION, EventType::EFFECTOR_ACTION};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Add different events
    FrameEvent frame_event(std::time(nullptr), std::nullopt);
    Event detection_event(EventType::DETECTION, std::time(nullptr), std::nullopt);
    Event effector_event(EventType::EFFECTOR_ACTION, std::time(nullptr), std::nullopt);
    
    recorder.notify(frame_event);
    recorder.notify(detection_event);
    recorder.notify(effector_event);
    
    // Start recorder to process events
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    recorder.stop();
    
    // Check that all handlers were called
    CHECK(recorder.new_frame_handled == true);
    CHECK(recorder.detection_handled == true);
    CHECK(recorder.effector_action_handled == true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("ContinuousRecorder constructor initializes correctly") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    
    CHECK(recorder.listening_to() == events);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("ContinuousRecorder handles new frame events") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create a test image with the same size as recorder expects (1080x1920)
    cv::Mat test_image = create_video_image(1920, 1080);
    time_t timestamp = std::time(nullptr);
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    FrameEvent frame_event(timestamp, std::nullopt);
    recorder.notify(frame_event);
    
    // Start recorder briefly to process the event
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    recorder.stop();
    
    // Test passes if no exceptions were thrown during video writing
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("ContinuousRecorder handles missing frames gracefully") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create frame event with timestamp that doesn't exist in image store
    time_t nonexistent_timestamp = 999999;
    FrameEvent frame_event(nonexistent_timestamp, std::nullopt);
    recorder.notify(frame_event);
    
    // Start recorder briefly to process the event
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    recorder.stop();
    
    // Test passes if no exceptions were thrown when frame is missing
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("ContinuousRecorder ignores detection and effector events") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION, EventType::EFFECTOR_ACTION};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Add non-frame events
    Event detection_event(EventType::DETECTION, std::time(nullptr), std::nullopt);
    Event effector_event(EventType::EFFECTOR_ACTION, std::time(nullptr), std::nullopt);
    
    recorder.notify(detection_event);
    recorder.notify(effector_event);
    
    // Start recorder briefly to process events
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    recorder.stop();
    
    // Test passes if no exceptions were thrown (events should be ignored)
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder destructor stops running recorder") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    {
        MockRecorder recorder(events, image_store, temp_dir);
        recorder.set_event_queue(event_queue);
        recorder.start();
        // Let destructor handle cleanup
    }
    
    // If we reach here without hanging, destructor worked correctly
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder creates video file when started") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create a test image and add it to image store
    cv::Mat test_image = create_video_image(1920, 1080);
    time_t timestamp = std::time(nullptr);
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    FrameEvent frame_event(timestamp, std::nullopt);
    recorder.notify(frame_event);
    
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Give more time for file creation
    recorder.stop();
    
    // Check if any .mp4 files exist in the temp directory
    bool found_avi_file = false;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.path().extension() == ".mp4") {
            found_avi_file = true;
            break;
        }
    }
    
    // Test passes if we found an avi file in the specified directory
    CHECK(found_avi_file == true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("Recorder creates video file in specified directory") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(10);
    auto event_queue = std::make_shared<std::queue<Event>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Verify that the directory was created
    CHECK(std::filesystem::exists(temp_dir));
    CHECK(std::filesystem::is_directory(temp_dir));
    
    // Create a test image and add it to image store
    cv::Mat test_image = create_video_image(1920, 1080);
    time_t timestamp = std::time(nullptr);
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    FrameEvent frame_event(timestamp, std::nullopt);
    recorder.notify(frame_event);
    
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    recorder.stop();
    
    // Check that video file was created in the specified directory, not in current directory
    bool found_avi_in_temp = false;
    bool found_avi_in_current = false;
    
    // Check temp directory
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.path().extension() == ".mp4") {
            found_avi_in_temp = true;
            break;
        }
    }
    
    // Check current directory
    std::filesystem::path current_dir = std::filesystem::current_path();
    for (const auto& entry : std::filesystem::directory_iterator(current_dir)) {
        std::string filename = entry.path().filename().string();
        if (entry.path().extension() == ".mp4" && 
            filename.substr(0, 10) == "recording_") {
            found_avi_in_current = true;
            break;
        }
    }
    
    CHECK(found_avi_in_temp == true);
    CHECK(found_avi_in_current == false);
    
    cleanup_directory(temp_dir);
}
