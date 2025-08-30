#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "events.hpp"
#include "mm.hpp"
#include "detection.hpp"

namespace fs = std::filesystem;

struct Args {
    std::string video_path = "tests/test_videos/20250813_173458.mp4";
    std::string weights_path = "weights/bh_v3.onnx";
    std::string out_dir = "output_cpp";
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.45f;
    int max_frames = 0; // 0 = no limit
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [--video PATH] [--weights PATH] [--out-dir DIR] [--imgsz N] [--conf F] [--iou F] [--max-frames N]\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto next = [&](void) -> const char* {
            if (i + 1 < argc) return argv[++i];
            return nullptr;
        };
        if (key == "-h" || key == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (key == "--video") {
            if (const char* v = next()) a.video_path = v; else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--weights") {
            if (const char* v = next()) a.weights_path = v; else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--out-dir") {
            if (const char* v = next()) a.out_dir = v; else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--imgsz") {
            if (const char* v = next()) a.imgsz = std::stoi(v); else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--conf") {
            if (const char* v = next()) a.conf = std::stof(v); else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--iou") {
            if (const char* v = next()) a.iou = std::stof(v); else { print_usage(argv[0]); std::exit(1);}        
        } else if (key == "--max-frames") {
            if (const char* v = next()) a.max_frames = std::stoi(v); else { print_usage(argv[0]); std::exit(1);}        
        } else {
            std::cerr << "Unknown arg: " << key << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return a;
}

static void ensure_out_dir(const fs::path& p) {
    std::error_code ec;
    fs::create_directories(p, ec);
    if (ec) {
        throw std::runtime_error(std::string("Failed to create output directory: ") + ec.message());
    }
}

static void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        auto boxes_opt = det.get_bounding_boxes();
        auto labels_opt = det.get_labels();
        auto confs_opt = det.get_confidences();
        const std::vector<cv::Rect> empty_boxes;
        const std::vector<std::string> empty_labels;
        const std::vector<float> empty_confs;

        const auto& boxes = boxes_opt.has_value() ? boxes_opt.value() : empty_boxes;
        const auto& labels = labels_opt.has_value() ? labels_opt.value() : empty_labels;
        const auto& confs = confs_opt.has_value() ? confs_opt.value() : empty_confs;

        for (size_t i = 0; i < boxes.size(); ++i) {
            const cv::Rect& r = boxes[i];
            cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 2);

            std::ostringstream oss;
            if (i < labels.size()) oss << labels[i]; else oss << "object";
            if (i < confs.size()) oss << " " << std::fixed << std::setprecision(2) << confs[i];

            std::string text = oss.str();
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(r.y, labelSize.height);
            cv::rectangle(frame, cv::Point(r.x, top - labelSize.height), cv::Point(r.x + labelSize.width, top + baseLine), cv::Scalar(0, 0, 255), cv::FILLED);
            cv::putText(frame, text, cv::Point(r.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Validate inputs
    if (!fs::exists(args.video_path)) {
        std::cerr << "[ERROR] Video not found: " << args.video_path << "\n";
        return 1;
    }
    if (!fs::exists(args.weights_path)) {
        std::cerr << "[ERROR] Weights not found: " << args.weights_path << "\n";
        return 1;
    }

    ensure_out_dir(args.out_dir);

    // Core components
    auto image_store = std::make_shared<ImageStore>(MAX_IMAGE_STORE_SIZE);

    // Configure detector
    BirdDetectorYolov5 detector(
        image_store,
        args.weights_path,
        cv::Size(args.imgsz, args.imgsz),
        args.conf,
        args.iou,
        std::chrono::seconds(10),
        50
    );

    // Video reader
    cv::VideoCapture cap(args.video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open video: " << args.video_path << "\n";
        return 2;
    }

    int frame_idx = 0;
    int saved = 0;
    auto t0 = std::chrono::steady_clock::now();
    Timestamp last_ts = now();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        // Generate a near-now unique timestamp in milliseconds
        Timestamp ts = now();
        if (ts <= last_ts) {
            // ensure strictly increasing to avoid map key collisions
            auto inc = std::chrono::milliseconds(1);
            ts = last_ts + inc;
        }
        last_ts = ts;

        // Put frame in store and create event
        image_store->put(ts, frame);
        auto frame_event = std::make_shared<FrameEvent>(ts, std::nullopt);

        // Run detection
        std::optional<DetectionEvent> det_event = detector.detect(frame_event);

        // Draw and save regardless; draw if detections exist
        if (det_event.has_value()) {
            draw_detections(frame, det_event->get_detections());
        }

        // Save frame
        std::ostringstream fname;
        fname << "frame_" << std::setw(6) << std::setfill('0') << frame_idx << ".jpg";
        fs::path out_path = fs::path(args.out_dir) / fname.str();
        cv::imwrite(out_path.string(), frame);
        ++saved;

        ++frame_idx;
        if (args.max_frames > 0 && frame_idx >= args.max_frames) break;
    }

    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Done. Saved " << saved << " frames to " << args.out_dir << " in " << (dt / 1000.0) << "s" << std::endl;
    return 0;
}
