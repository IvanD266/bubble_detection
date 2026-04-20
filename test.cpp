#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cv::Mat frame;

    if (!cap.isOpened()) return -1;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) continue;

        cv::imshow("cam", frame);
        if (cv::waitKey(1) == 27) break;
    }
}
