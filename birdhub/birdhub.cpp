#include "video/video.hpp"
#include "orchestration/orchestration.hpp"
#include "detection/detection.hpp"
#include "memory_management/mm.hpp"
#include "recording/recorder.hpp"
#include "effectors/mock_effector.hpp"
#include "effectors/sound_effector.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>

// Parse comma-separated target classes string to vector
static std::vector<std::string> parse_classes(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) result.push_back(token);
    }
    return result;
}

// Build a CameraCapture from a device string (index or path)
static std::unique_ptr<CameraCapture> make_camera(const std::string& device) {
#ifdef __linux__
    if (device.rfind("/dev/", 0) == 0) {
        return std::make_unique<V4l2CameraCapture>(device, 640, 480, V4L2_PIX_FMT_MJPEG);
    }
#endif
    try {
        int idx = std::stoi(device);
        return std::make_unique<OpenCVCameraCapture>(idx);
    } catch (const std::invalid_argument&) {
        return std::make_unique<OpenCVCameraCapture>(device);
    }
}

int main(int argc, char** argv) {
    VideoEventManager::setup_signal_handlers();

    CLI::App app{"Pigeonator - bird detection and deterrence system"};
    app.require_subcommand(1);

    // ─── record ────────────────────────────────────────────────────────────
    auto* rec = app.add_subcommand("record", "Recording commands");
    rec->require_subcommand(1);

    // record continuous
    std::string cont_device, cont_outdir;
    int cont_fps = 10;
    auto* cont_cmd = rec->add_subcommand("continuous", "Record all frames continuously");
    cont_cmd->add_option("device", cont_device, "Camera device index or path")->required();
    cont_cmd->add_option("outputdir", cont_outdir, "Output directory")->required();
    cont_cmd->add_option("--fps", cont_fps)->default_val(10);

    // record motion
    std::string mot_device, mot_outdir;
    int mot_fps = 10, mot_slack = 100;
    auto* mot_cmd = rec->add_subcommand("motion", "Record triggered by motion");
    mot_cmd->add_option("device", mot_device, "Camera device index or path")->required();
    mot_cmd->add_option("outputdir", mot_outdir, "Output directory")->required();
    mot_cmd->add_option("--fps", mot_fps)->default_val(10);
    mot_cmd->add_option("--slack", mot_slack, "Post-event slack frames")->default_val(100);

    // record birds
    std::string birds_device, birds_outdir, birds_model;
    int birds_fps = 10, birds_slack = 100;
    auto* birds_cmd = rec->add_subcommand("birds", "Record triggered by bird detection");
    birds_cmd->add_option("device", birds_device, "Camera device index or path")->required();
    birds_cmd->add_option("outputdir", birds_outdir, "Output directory")->required();
    birds_cmd->add_option("--fps", birds_fps)->default_val(10);
    birds_cmd->add_option("--slack", birds_slack, "Post-event slack frames")->default_val(100);
    birds_cmd->add_option("--model", birds_model, "Path to ONNX model")->default_val("weights/bh_v1.onnx");

    // ─── deter ─────────────────────────────────────────────────────────────
    std::string deter_device, deter_outdir;
    std::string deter_target_classes = "Pigeon,Crow";
    int deter_fps = 10, deter_slack = 100, deter_min_detections = 5;
    int deter_motion_th_area = 2000;
    std::string deter_model, deter_effector, deter_sound_path;
    bool deter_record = true;

    auto* deter_cmd = app.add_subcommand("deter", "Full deterrence pipeline");
    deter_cmd->add_option("device", deter_device, "Camera device index or path")->required();
    deter_cmd->add_option("outputdir", deter_outdir, "Output directory")->required();
    deter_cmd->add_option("--target_classes", deter_target_classes, "Comma-separated target classes")->default_val("Pigeon,Crow");
    deter_cmd->add_option("--fps", deter_fps)->default_val(10);
    deter_cmd->add_option("--slack", deter_slack, "Post-event slack frames")->default_val(100);
    deter_cmd->add_option("--model", deter_model, "Path to ONNX model")->default_val("weights/bh_v3.onnx");
    deter_cmd->add_option("--effector", deter_effector, "Effector type: mock or sound")->default_val("mock");
    deter_cmd->add_option("--sound_path", deter_sound_path, "Sound file for SoundEffector")->default_val("sounds/crow_1.mp3");
    deter_cmd->add_flag("--no-record{false}", deter_record, "Disable recording");
    deter_cmd->add_option("--minimum_detections", deter_min_detections, "Min detections for consensus")->default_val(5);
    deter_cmd->add_option("--motion_th_area", deter_motion_th_area, "Motion threshold area in pixels")->default_val(2000);

    CLI11_PARSE(app, argc, argv);

    // ─── handle: record continuous ─────────────────────────────────────────
    if (*cont_cmd) {
        auto camera = make_camera(cont_device);
        auto image_store = std::make_shared<ImageStore>(5000);
        auto stream = Stream(image_store, camera.get());
        auto recorder = std::make_shared<ContinuousRecorder>(
            std::set<EventType>({EventType::NEW_FRAME}),
            image_store, cont_outdir
        );
        auto manager = VideoEventManager(stream);
        manager.add_subscriber(recorder);
        manager.run();
        return 0;
    }

    // ─── handle: record motion ─────────────────────────────────────────────
    if (*mot_cmd) {
        auto camera = make_camera(mot_device);
        auto image_store = std::make_shared<ImageStore>(5000);
        auto stream = Stream(image_store, camera.get());
        auto motion_detector = std::make_shared<MotionDetector>(
            image_store, 20, 21, 5, 1000, 1, std::chrono::seconds(5)
        );
        auto recorder = std::make_shared<EventRecorder>(
            std::set<EventType>({EventType::NEW_FRAME, EventType::DETECTION}),
            image_store, mot_outdir, mot_slack, mot_fps, mot_slack
        );
        auto manager = VideoEventManager(stream);
        manager.add_subscriber(motion_detector);
        manager.add_subscriber(recorder);
        manager.run();
        return 0;
    }

    // ─── handle: record birds ──────────────────────────────────────────────
    if (*birds_cmd) {
        auto camera = make_camera(birds_device);
        auto image_store = std::make_shared<ImageStore>(5000);
        auto stream = Stream(image_store, camera.get());
        auto motion_detector = std::make_shared<MotionDetector>(
            image_store, 20, 21, 5, 1000, 1, std::chrono::seconds(2)
        );
        auto bird_detector = std::make_shared<BirdDetectorYolov5>(
            image_store, birds_model
        );
        auto combined = std::make_shared<MotionActivatedDetector>(
            motion_detector, bird_detector, image_store
        );
        auto recorder = std::make_shared<EventRecorder>(
            std::set<EventType>({EventType::NEW_FRAME, EventType::DETECTION}),
            image_store, birds_outdir, birds_slack, birds_fps, birds_slack
        );
        auto manager = VideoEventManager(stream);
        manager.add_subscriber(combined);
        manager.add_subscriber(recorder);
        manager.run();
        return 0;
    }

    // ─── handle: deter ─────────────────────────────────────────────────────
    if (*deter_cmd) {
        auto target_classes = parse_classes(deter_target_classes);
        auto camera = make_camera(deter_device);
        auto image_store = std::make_shared<ImageStore>(5000);
        auto stream = Stream(image_store, camera.get());

        auto motion_detector = std::make_shared<MotionDetector>(
            image_store, 20, 21, 5, deter_motion_th_area, 1, std::chrono::seconds(2)
        );
        auto bird_detector = std::make_shared<BirdDetectorYolov5>(
            image_store, deter_model, cv::Size(640, 640), 0.6f, 0.45f,
            std::chrono::seconds(2), deter_motion_th_area
        );
        auto combined = std::make_shared<MotionActivatedDetector>(
            motion_detector, bird_detector, image_store, deter_motion_th_area
        );

        std::shared_ptr<Effector> effector;
        if (deter_effector == "sound") {
            effector = std::make_shared<SoundEffector>(target_classes, deter_sound_path);
        } else {
            effector = std::make_shared<MockEffector>(target_classes);
        }

        auto manager = VideoEventManager(stream);
        manager.add_subscriber(combined);
        manager.add_subscriber(effector);

        if (deter_record) {
            auto recorder = std::make_shared<EventRecorder>(
                std::set<EventType>({EventType::NEW_FRAME, EventType::DETECTION, EventType::EFFECTOR_ACTION}),
                image_store, deter_outdir, deter_slack, deter_fps, deter_slack
            );
            manager.add_subscriber(recorder);
        }

        manager.run();
        return 0;
    }

    return 0;
}
