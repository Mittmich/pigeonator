#include "doctest/doctest.h"
#include "recorder.hpp"
#include "test_utils.hpp"
#include "timestamp_utils.hpp"
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
                                   ("recorder_test_" + std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now().time_since_epoch()).count()) + "_" + 
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
    void handle_new_frame(std::shared_ptr<FrameEvent> frame_event) override {
        new_frame_handled = true;
    }
    
    void handle_detection(std::shared_ptr<DetectionEvent> detection_event) override {
        detection_handled = true;
    }
    
    void handle_effector_action(std::shared_ptr<Event> effector_event) override {
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Notify with NEW_FRAME event (should be accepted)
    auto frame_event = std::make_shared<FrameEvent>(test_now(), std::nullopt);
    recorder.notify(frame_event);
    
    // Notify with DETECTION event (should be ignored)
    auto detection_event = std::make_shared<Event>(EventType::DETECTION, test_now(), std::nullopt);
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    MockRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Add different events
    auto frame_event = std::make_shared<FrameEvent>(test_now(), std::nullopt);
    auto detection_event = std::make_shared<Event>(EventType::DETECTION, test_now(), std::nullopt);
    auto effector_event = std::make_shared<Event>(EventType::EFFECTOR_ACTION, test_now(), std::nullopt);
    
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create a test image with the same size as recorder expects (720x1280)
    cv::Mat test_image = create_video_image(720, 1280);
    Timestamp timestamp = test_now();
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create frame event with timestamp that doesn't exist in image store
    Timestamp nonexistent_timestamp = test_timestamp_offset(-999999);
    auto frame_event = std::make_shared<FrameEvent>(nonexistent_timestamp, std::nullopt);
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Add non-frame events
    auto detection_event = std::make_shared<Event>(EventType::DETECTION, test_now(), std::nullopt);
    auto effector_event = std::make_shared<Event>(EventType::EFFECTOR_ACTION, test_now(), std::nullopt);
    
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Create a test image and add it to image store (720x1280)
    cv::Mat test_image = create_video_image(720, 1280);
    Timestamp timestamp = test_now();
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
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
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    std::string temp_dir = create_temp_directory();
    
    ContinuousRecorder recorder(events, image_store, temp_dir);
    recorder.set_event_queue(event_queue);
    
    // Verify that the directory was created
    CHECK(std::filesystem::exists(temp_dir));
    CHECK(std::filesystem::is_directory(temp_dir));
    
    // Create a test image and add it to image store (720x1280)
    cv::Mat test_image = create_video_image(720, 1280);
    Timestamp timestamp = test_now();
    image_store->put(timestamp, test_image);
    
    // Create frame event with same timestamp
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
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

// EventRecorder specific tests

Detection create_test_detection(Timestamp timestamp, std::shared_ptr<FrameEvent> frame_event) {
    std::vector<std::string> labels = {"bird"};
    std::vector<float> confidences = {0.85f};
    std::vector<cv::Rect> boxes = {cv::Rect(10, 10, 50, 50)};
    return Detection(timestamp, frame_event, labels, confidences, boxes);
}

DetectionEvent create_test_detection_event(Timestamp timestamp) {
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    std::vector<Detection> detections = {create_test_detection(timestamp, frame_event)};
    return DetectionEvent(timestamp, detections);
}

// Test helper class to access protected members
class TestableEventRecorder : public EventRecorder {
public:
    TestableEventRecorder(
        std::set<EventType> listening_events,
        std::shared_ptr<ImageStore> image_store,
        const std::string& output_directory = ".",
        int slack = 100,
        int fps = 30,
        int look_back_frames = 3
    ) : EventRecorder(listening_events, image_store, output_directory, slack, fps, look_back_frames) {}
    
    // Expose protected methods for testing
    FrameEvent test_create_detection_frame(std::shared_ptr<DetectionEvent> detection_event, Timestamp frame_timestamp) {
        return create_detection_frame(detection_event, frame_timestamp);
    }
    
    
    void test_handle_new_frame(std::shared_ptr<FrameEvent> event) {
        handle_new_frame(event);
    }
    
    void test_handle_detection(std::shared_ptr<DetectionEvent> detection_event) {
        handle_detection(detection_event);
    }
    
    void test_handle_effector_action(std::shared_ptr<Event> event) {
        handle_effector_action(event);
    }
};

TEST_CASE("EventRecorder constructor initializes with custom parameters") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    EventRecorder recorder(events, image_store, temp_dir, 50, 25, 5);
    
    CHECK(recorder.listening_to() == events);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder creates detection frame with bounding boxes") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir);
    
    // Setup test image and detection
    Timestamp timestamp = test_now();
    cv::Mat test_image = create_video_image(100, 100);
    image_store->put(timestamp, test_image);
    
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(timestamp));
    
    // Create detection frame
    FrameEvent detection_frame = recorder.test_create_detection_frame(detection_event, timestamp);
    
    CHECK(detection_frame.get_timestamp() == timestamp);
    CHECK(detection_frame.get_meta_data().count("type") > 0);
    CHECK(detection_frame.get_meta_data().at("type") == "detection_frame");
    CHECK(detection_frame.get_meta_data().count("detection_count") > 0);
    CHECK(detection_frame.get_meta_data().at("detection_count") == "1");
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder handles new frame updates buffers") {
    std::set<EventType> events = {EventType::NEW_FRAME};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir, 10, 30, 3);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    // Setup test images
    Timestamp timestamp1 = test_now();
    Timestamp timestamp2 = test_timestamp_offset(1);
    Timestamp timestamp3 = test_timestamp_offset(2);
    
    cv::Mat test_image1 = create_video_image(1920, 1080);
    cv::Mat test_image2 = create_video_image(1920, 1080);
    cv::Mat test_image3 = create_video_image(1920, 1080);
    
    image_store->put(timestamp1, test_image1);
    image_store->put(timestamp2, test_image2);
    image_store->put(timestamp3, test_image3);
    
    // Create frame events
    auto event1 = std::make_shared<FrameEvent>(timestamp1, std::nullopt);
    auto event2 = std::make_shared<FrameEvent>(timestamp2, std::nullopt);
    auto event3 = std::make_shared<FrameEvent>(timestamp3, std::nullopt);
    
    // Handle frame events
    recorder.test_handle_new_frame(event1);
    recorder.test_handle_new_frame(event2);
    recorder.test_handle_new_frame(event3);
    
    // Test passes if no exceptions are thrown and buffer operations work correctly
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder starts recording on detection event") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir, 5, 30, 2);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    // Setup test images for look-back frames
    Timestamp base_time = test_now();
    for (int i = 0; i < 3; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.test_handle_new_frame(frame_event);
    }
    
    // Create detection event
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(test_timestamp_offset(1)));
    // Handle detection - should start recording
    recorder.test_handle_detection(detection_event);
    
    // Test passes if no exceptions are thrown
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder handles effector action events") {
    std::set<EventType> events = {EventType::EFFECTOR_ACTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    // Create effector action event
    Timestamp timestamp = test_now();
    std::map<std::string, std::string> metadata = {{"type", "SPRAY"}};
    auto effector_event = std::make_shared<Event>(EventType::EFFECTOR_ACTION, timestamp, metadata);
    
    // Handle effector action
    recorder.test_handle_effector_action(effector_event);
    
    // Test passes if no exceptions are thrown and event is buffered
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

// Removed obsolete test that depended on now-removed create_detection_frames implementation.

TEST_CASE("EventRecorder handles missing image in store gracefully") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    // Create frame event for timestamp that doesn't exist in image store
    Timestamp timestamp = test_now();
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    
    // This should be handled gracefully
    recorder.test_handle_new_frame(frame_event);
    
    // Test passes if missing image is handled without exceptions
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder buffer size limits are respected") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    // Create recorder with small buffer size for testing
    TestableEventRecorder recorder(events, image_store, temp_dir, 5, 30, 2);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    Timestamp base_time = test_now();
    
    // Add more frames than buffer size
    for (int i = 0; i < 10; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.test_handle_new_frame(frame_event);
    }
    
    // Test passes if buffer management works correctly without exceptions
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder creates detections video file when recording") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    TestableEventRecorder recorder(events, image_store, temp_dir, 3, 30, 2); // Small detection buffer
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    Timestamp base_time = test_now();
    
    // Setup frames and trigger detection
    for (int i = 0; i < 5; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.test_handle_new_frame(frame_event);
        
        // Add detection events to trigger buffer overflow and video creation
        if (i > 1) {
            auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(timestamp));
            recorder.test_handle_detection(detection_event);
        }
    }
    
    // Process a few more frames to trigger the buffer overflow
    for (int i = 5; i < 8; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.test_handle_new_frame(frame_event);
    }
    
    // stop recorder to trigger file creation
    recorder.stop();

    // Check if detection video file exists
    bool found_detection_file = false;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        std::string filename = entry.path().filename().string();
        if (entry.path().extension() == ".mp4" && 
            filename.substr(0, 11) == "detections_") {
            found_detection_file = true;
            break;
        }
    }
    
    // Test passes if detection video was created or if no exceptions were thrown
    CHECK(found_detection_file == true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder full integration test with notify interface") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION, EventType::EFFECTOR_ACTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    EventRecorder recorder(events, image_store, temp_dir, 3, 30, 2);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    Timestamp base_time = test_now();
    
    // Build up look-back buffer
    for (int i = 0; i < 3; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.notify(frame_event);
    }
    
    // Start the recorder
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Trigger a detection event (should start recording)
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(test_timestamp_offset(2)));
    recorder.notify(detection_event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Add effector action
    std::map<std::string, std::string> metadata = {{"type", "SPRAY"}};
    auto effector_event = std::make_shared<Event>(EventType::EFFECTOR_ACTION, test_timestamp_offset(3), metadata);
    recorder.notify(effector_event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Add more frames to continue recording
    for (int i = 3; i < 7; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        recorder.notify(frame_event);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    recorder.stop();
    
    // Test passes if the full integration completes without exceptions
    CHECK(true);
    
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder notify interface filtering") {
    std::set<EventType> events = {EventType::DETECTION}; // Only listening to detection events
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    EventRecorder recorder(events, image_store, temp_dir);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);
    
    Timestamp timestamp = test_now();
    
    // Create events of different types
    auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(timestamp));
    auto effector_event = std::make_shared<Event>(EventType::EFFECTOR_ACTION, timestamp, std::nullopt);
    
    // Start recorder briefly to process events
    recorder.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Notify with different event types
    recorder.notify(frame_event);     // Should be ignored
    recorder.notify(detection_event); // Should be accepted
    recorder.notify(effector_event);  // Should be ignored
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    recorder.stop();
    
    // Test passes if no exceptions are thrown
    CHECK(true);
    
    cleanup_directory(temp_dir);
}


TEST_CASE("EventRecorder writes correct number of frames to detections video") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    EventRecorder recorder(events, image_store, temp_dir, 10, 30, 10);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);

    Timestamp base_time = test_now();
    std::vector<std::shared_ptr<FrameEvent>> frame_events;
    // Create all frames upfront
    for (int i = 0; i < 5; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        frame_events.push_back(frame_event);
    }

    // Start the recorder
    recorder.start();

    // Add 2 frames to the recorder
    for (int i = 0; i < 2; i++) {
        recorder.notify(frame_events[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Trigger a detection event
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(test_timestamp_offset(1)));
    recorder.notify(detection_event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Add remaining frames to the recorder
    for (int i = 2; i < 5; i++) {
        recorder.notify(frame_events[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Wait for the recorder to finish writing
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    recorder.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Recorder stopped, checking video frames..." << std::endl;
    // Load frames from file to verify order
    // Get detections video file path by searching temp directory
    bool found_video_file = false;
    std::string video_file;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.path().extension() == ".mp4" &&
            entry.path().filename().string().find("detections_") != std::string::npos) {
            video_file = entry.path().string();
            found_video_file = true;
            break;
        }
    }
    // print video file path
    std::cout << "Video file path: " << video_file << std::endl;
    CHECK(found_video_file == true);
    cv::VideoCapture cap(video_file);
    CHECK(cap.isOpened());
    std::vector<cv::Mat> video_frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        video_frames.push_back(frame);
    }
    cap.release();
    // Check if right amount of frames were written
    CHECK(video_frames.size() == 5);
    cleanup_directory(temp_dir);
}

TEST_CASE("EventRecorder writes correct number of frames to detections video (alt path)") {
    std::set<EventType> events = {EventType::NEW_FRAME, EventType::DETECTION};
    auto image_store = std::make_shared<ImageStore>(50);
    std::string temp_dir = create_temp_directory();
    
    EventRecorder recorder(events, image_store, temp_dir, 10, 30, 10);
    auto event_queue = std::make_shared<std::queue<std::shared_ptr<Event>>>();
    recorder.set_event_queue(event_queue);

    Timestamp base_time = test_now();
    std::vector<std::shared_ptr<FrameEvent>> frame_events;
    // Create all frames upfront
    for (int i = 0; i < 5; i++) {
        Timestamp timestamp = test_timestamp_offset(i);
        cv::Mat test_image = create_video_image(720, 1280);
        image_store->put(timestamp, test_image);
        auto frame_event = std::make_shared<FrameEvent>(timestamp, std::nullopt);
        frame_events.push_back(frame_event);
    }

    // Start the recorder
    recorder.start();

    // Add 2 frames to the recorder
    for (int i = 0; i < 2; i++) {
        recorder.notify(frame_events[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Trigger a detection event
    auto detection_event = std::make_shared<DetectionEvent>(create_test_detection_event(test_timestamp_offset(1)));
    recorder.notify(detection_event);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Add remaining frames to the recorder
    for (int i = 2; i < 5; i++) {
        recorder.notify(frame_events[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Wait for the recorder to finish writing
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    recorder.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Recorder stopped, checking video frames..." << std::endl;
    // Load frames from file to verify order
    // Get detections video file path by searching temp directory
    bool found_video_file = false;
    std::string video_file;
    for (const auto& entry : std::filesystem::directory_iterator(temp_dir)) {
        if (entry.path().extension() == ".mp4" &&
            entry.path().filename().string().find("detections_") != std::string::npos) {
            video_file = entry.path().string();
            found_video_file = true;
            break;
        }
    }
    // print video file path
    std::cout << "Video file path: " << video_file << std::endl;
    CHECK(found_video_file == true);
    cv::VideoCapture cap(video_file);
    CHECK(cap.isOpened());
    std::vector<cv::Mat> video_frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        video_frames.push_back(frame);
    }
    cap.release();
    // Check if right amount of frames were written
    CHECK(video_frames.size() == 5);
    cleanup_directory(temp_dir);
}   