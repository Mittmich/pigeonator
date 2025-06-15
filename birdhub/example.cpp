#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>

int main() {
    // Initialize video capture from default camera (usually webcam)
    cv::VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam!" << std::endl;
        return -1;
    }
    
    // Get webcam properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // If fps is not available or invalid, set a default
    if (fps <= 0) {
        fps = 30.0;
    }
    
    std::cout << "Webcam properties:" << std::endl;
    std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    
    // Create output filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::string filename = "webcam_recording_" + std::to_string(time_t) + ".mp4";
    
    // Define the codec and create VideoWriter object
    cv::VideoWriter writer(filename, 
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          fps,
                          cv::Size(frame_width, frame_height));
    
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer!" << std::endl;
        return -1;
    }
    
    std::cout << "Recording to: " << filename << std::endl;
    std::cout << "Recording will stop automatically after 30 seconds..." << std::endl;
    std::cout << "Press 'q' to quit early" << std::endl;
    
    // Recording variables
    cv::Mat frame;
    auto start_time = std::chrono::steady_clock::now();
    const std::chrono::seconds recording_duration(30);
    int frame_count = 0;
    
    while (true) {
        // Capture frame-by-frame
        cap >> frame;
        
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame!" << std::endl;
            break;
        }
        
        // Write the frame to the output file
        writer.write(frame);
        frame_count++;
        
        // Display the frame (optional - comment out if running headless)
        cv::imshow("Webcam Recording", frame);
        
        // Check for 'q' key press to quit early
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q') {
            std::cout << "Recording stopped by user input." << std::endl;
            break;
        }
        
        // Check if 30 seconds have elapsed
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        
        if (elapsed >= recording_duration) {
            std::cout << "30 seconds elapsed. Stopping recording." << std::endl;
            break;
        }
        
        // Print progress every 5 seconds
        if (elapsed.count() % 5 == 0 && elapsed.count() > 0) {
            static int last_reported = -1;
            if (elapsed.count() != last_reported) {
                std::cout << "Recording... " << elapsed.count() << " seconds elapsed" << std::endl;
                last_reported = elapsed.count();
            }
        }
    }
    
    // Clean up
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    std::cout << "Recording completed!" << std::endl;
    std::cout << "Frames recorded: " << frame_count << std::endl;
    std::cout << "Output file: " << filename << std::endl;
    
    return 0;
}