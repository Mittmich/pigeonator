#include <video.cpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    try {
        // Create the camera capture object with the desired settings
        CameraCapture camera("/dev/video0", 640, 480, V4L2_PIX_FMT_MJPEG);

        // Define the codec and create a VideoWriter object to write the output video
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // Codec for MP4 (MPEG-4)
        cv::VideoWriter videoWriter("output_video.mp4", fourcc, 10.0, cv::Size(640, 480));

        // Check if the video writer is initialized correctly
        if (!videoWriter.isOpened()) {
            std::cerr << "Could not open the video file for writing" << std::endl;
            return -1;
        }

        // Capture and write frames to the video file
        for (int i = 0; i < 20; ++i) {  // Capture 300 frames (~10 seconds of video at 30 FPS)
            cv::Mat frame = camera.getNextFrame();
            if (!frame.empty()) {
                // Write the frame to the MP4 file
                videoWriter.write(frame);
            } else {
                std::cerr << "Empty frame captured" << std::endl;
            }
        }

        // Release the VideoWriter
        videoWriter.release();
        std::cout << "Video has been saved to output_video.mp4" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
