#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img(2, 4, CV_64FC1);
    double low = -500.0;
    double high = +500.0;
    cv::randu(img, cv::Scalar(low), cv::Scalar(high));
    std::cout << "M = " << std::endl << " "  << img << std::endl << std::endl;
    return 0;
}