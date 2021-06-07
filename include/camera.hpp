/*
 * @Description: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-23 10:57:34
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-04-01 09:42:11
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "object.h"

using namespace std;
using namespace cv;

class Camera
{
public:
    cv::VideoCapture cap;
    Camera(int camera_num);
    virtual ~Camera();
    void getCameraPixel(cv::Mat& image, int* image_size);
    void display(cv::Mat& image, cv::Rect2f& roi, const string& winname);
    void calculatePosition(cv::Rect2f& roi, cv::Mat& image, int* delta);
    // cv::Mat equalize_RGB(cv::Mat& image);

private:
    
    int dev_number;
    // cv::Mat frame;
    int image_width;
    int image_height;
};



#endif