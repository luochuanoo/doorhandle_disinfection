/*
 * @Description: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-23 11:29:09
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-04-01 09:38:56
 */

#include "camera.hpp"

Camera::Camera(int camera_num)
{
    // frame = new Mat();
    dev_number = camera_num;
	cap.open(dev_number);
}

Camera::~Camera()
{
    // delete frame;
    cap.release();
}

void Camera::getCameraPixel(Mat& image, int* image_size)
{
    image_size[0] = image.cols;
    image_size[1] = image.rows;
}

void Camera::display(cv::Mat& image, cv::Rect2f& roi, const string& winname)
{
    if (roi.empty())
    {
	    cv::namedWindow(winname, WINDOW_NORMAL);
	    cv::imshow(winname, image);
    }
    else
    {
        cv::rectangle(image, cv::Point(roi.x, roi.y), cv::Point(roi.x + roi.width, roi.y + roi.height), cv::Scalar(0, 255, 0), 3, LINE_8);
	    cv::namedWindow(winname, WINDOW_NORMAL);
	    cv::imshow(winname, image);
    }
}

void Camera::calculatePosition(cv::Rect2f& roi, cv::Mat& image, int* delta)
{
    // 目标框中心
	float cx = roi.x + roi.width / 2.0f;
	float cy = roi.y + roi.height / 2.0f;

	delta[0] = int(cx) - image.cols / 2;
	delta[1] = image.rows / 2 - int(cy);
}

// cv::Mat Camera::equalize_RGB(cv::Mat& image)
// {
//     cv::Mat hsv;
//     cv::Mat dst;

//     cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

//     std::vector<cv::Mat> channels;
//     cv::split(hsv, channels);

//     // 均衡化 v 通道
//     cv::equalizeHist(channels[2], channels[2]);

//     cv::merge(channels, hsv);

//     cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR );
// //    imshow( "Equalize", result );
//     return dst;
// }