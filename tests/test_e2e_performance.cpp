#include "test_utils.hpp"
#include "detection.hpp"
#include "orchestration.hpp"
#include "video.hpp"
#include "timestamp_utils.hpp"
#include <doctest/doctest.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

// LoopingVideoFileCapture: loops the video file indefinitely
class LoopingVideoFileCapture : public CameraCapture {
public:
    LoopingVideoFileCapture(const std::string& video_path) : video_path_(video_path) {
        cap_.open(video_path);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Could not open video file: " + video_path);
        }
    }

    ~LoopingVideoFileCapture() override {
        if (cap_.isOpened()) cap_.release();
    }

    cv::Mat getNextFrame() override {
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) {
            // Loop back to start
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap_.read(frame);
        }
        return frame;
    }

    void startStreaming() override {}
    void stopStreaming() override {}

private:
    std::string video_path_;
    cv::VideoCapture cap_;
};

// FrameCountSubscriber: counts NEW_FRAME events atomically
class FrameCountSubscriber : public Subscriber {
public:
    std::atomic<int> frame_count{0};

    std::set<EventType> listening_to() override {
        return {EventType::NEW_FRAME};
    }
    void start() override {}
    void stop() override {}
    void set_event_queue(std::shared_ptr<std::queue<std::shared_ptr<Event>>> eq) override {
        (void)eq;
    }
    void notify(std::shared_ptr<Event> event) override {
        if (event->type == EventType::NEW_FRAME) {
            frame_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

TEST_CASE("Performance - raw frame throughput") {
    std::string video_path = "tests/test_videos/video_with_single_sitting_bird.mp4";
    auto image_store = std::make_shared<ImageStore>(800);

    auto video_capture = std::make_unique<LoopingVideoFileCapture>(video_path);
    auto video_stream = std::make_shared<Stream>(image_store, video_capture.get());

    auto frame_counter = std::make_shared<FrameCountSubscriber>();

    VideoEventManager event_manager(*video_stream);
    event_manager.add_subscriber(frame_counter);

    std::thread pipeline_thread([&event_manager]() {
        event_manager.run();
    });

    constexpr int duration_seconds = 5;
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    event_manager.stop();
    if (pipeline_thread.joinable()) pipeline_thread.join();

    double fps = static_cast<double>(frame_counter->frame_count.load()) / duration_seconds;
    MESSAGE("Raw FPS: " << fps << " (" << frame_counter->frame_count.load() << " frames in " << duration_seconds << "s)");
    CHECK(fps > 0.0);
}

TEST_CASE("Performance - full detection pipeline throughput") {
    std::string video_path = "tests/test_videos/video_with_single_sitting_bird.mp4";
    auto image_store = std::make_shared<ImageStore>(800);

    auto video_capture = std::make_unique<LoopingVideoFileCapture>(video_path);
    auto video_stream = std::make_shared<Stream>(image_store, video_capture.get());

    auto motion_detector = std::make_shared<MotionDetector>(
        image_store, 24, 21, 5, 100, 0, std::chrono::seconds(100));

    auto bird_detector = std::make_shared<BirdDetectorYolov5>(
        image_store, "weights/bh_v3.onnx", cv::Size(640, 640),
        0.25f, 0.45f, std::chrono::seconds(500), 50);

    auto motion_activated_detector = std::make_shared<MotionActivatedDetector>(
        motion_detector, bird_detector, image_store, 3, 5);

    auto frame_counter = std::make_shared<FrameCountSubscriber>();
    auto mock_subscriber = std::make_shared<MockSubscriber>();

    VideoEventManager event_manager(*video_stream);
    event_manager.add_subscriber(motion_activated_detector);
    event_manager.add_subscriber(frame_counter);
    event_manager.add_subscriber(mock_subscriber);

    std::thread pipeline_thread([&event_manager]() {
        event_manager.run();
    });

    constexpr int duration_seconds = 10;
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    event_manager.stop();
    if (pipeline_thread.joinable()) pipeline_thread.join();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    double frame_fps = static_cast<double>(frame_counter->frame_count.load()) / duration_seconds;

    size_t det_count = 0;
    for (const auto& event : mock_subscriber->received_events) {
        if (event->type == EventType::DETECTION) det_count++;
    }
    double det_per_sec = static_cast<double>(det_count) / duration_seconds;

    MESSAGE("Frame FPS: " << frame_fps
            << " | Detection events/s: " << det_per_sec
            << " (" << det_count << " detections in " << duration_seconds << "s)");
    CHECK(frame_fps > 0.0);
}

TEST_CASE("Performance - bird detector only throughput") {
    std::string video_path = "tests/test_videos/video_with_single_sitting_bird.mp4";
    auto image_store = std::make_shared<ImageStore>(800);

    // Load video frames into memory first to isolate detector performance
    cv::VideoCapture cap(video_path);
    REQUIRE(cap.isOpened());

    std::vector<std::shared_ptr<FrameEvent>> frames;
    cv::Mat raw;
    while (cap.read(raw) && !raw.empty()) {
        auto ts = now();
        auto frame_event = std::make_shared<FrameEvent>(ts, std::nullopt);
        image_store->put(ts, raw);
        frames.push_back(frame_event);
    }
    cap.release();
    REQUIRE(frames.size() > 0);

    // Create detector (not wired into event pipeline)
    BirdDetectorYolov5 detector(
        image_store, "weights/bh_v3.onnx", cv::Size(640, 640),
        0.25f, 0.45f, std::chrono::seconds(500), 1);

    // Run inference on every frame in a tight loop, cycling through frames
    int total_inferences = 0;
    int total_detections = 0;
    auto start = std::chrono::steady_clock::now();
    constexpr int duration_seconds = 10;
    auto deadline = start + std::chrono::seconds(duration_seconds);

    while (std::chrono::steady_clock::now() < deadline) {
        for (size_t i = 0; i < frames.size(); ++i) {
            if (std::chrono::steady_clock::now() >= deadline) break;
            auto result = detector.detect(frames[i]);
            total_inferences++;
            if (result.has_value()) total_detections++;
        }
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double inference_fps = total_inferences / elapsed;
    double det_per_sec = total_detections / elapsed;

    MESSAGE("Bird Detector inference FPS: " << inference_fps
            << " | Detection events/s: " << det_per_sec
            << " (" << total_inferences << " inferences, "
            << total_detections << " detections in " << elapsed << "s)");
    CHECK(inference_fps > 0.0);
}
