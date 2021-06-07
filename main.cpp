/*
 * @Descripttion: 图像读取 + 目标检测
 * @version: 
 * @Author: LuoChuan
 * @Date: 2020-08-03 15:37:20
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-04-01 10:54:33
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "device.hpp"
#include "ssd.hpp"
#include "camera.hpp"
#include "move.h"


using namespace std;
using namespace cv;

vector<string> VOC_CLASSES = {"__background__", "handrail", "circle", "vertical"};

int* delta_position = new int[2];
int* image_size = new int[2];
bool robot_completed = false;
// bool detect_completed = false;
// pthread_mutex_t mutex;
bool is_detected = true;

int main(int argc, char* argv[])
{	
	// 开 pthread_robot 线程
	DevicesModule* devs = new DevicesModule(&robot_completed);

	// 搭建 SSD 训练网络
	SSDetection SSDnet("../weight/ssd300_mobilenetv1_doorhandle_best.pt");				

	// 打开相机，读取图像
	Camera cam(0);
	Mat frame;
	cam.cap >> frame;
	cam.getCameraPixel(frame, image_size);
	
	Rect2f roi;
	// 行走之前预检测，过渡到循环检测
	roi = SSDnet.predict(frame);
	cout << "Pre-Detection Successful!" << endl;

	// 发送行走指令
	devs->send_motion_command(MOTION_FORWARD);

	while(1)
	{
		// 读取图像
		cam.cap >> frame;
		// frame = cam.equalize_RGB(frame);
		
		// 检测目标
		roi = SSDnet.predict(frame);
		if (roi.empty())
			continue;
		cam.display(frame, roi, "detect.jpg");

		// 计算 roi 中心相对于 frame 中心的偏移
		cam.calculatePosition(roi, frame, delta_position);

		if (!is_detected && delta_position[0] < 0)
		{
			// 发送停车指令
			devs->send_motion_command(MOTION_STOP);

			cam.cap >> frame;
			// frame = cam.equalize_RGB(frame);
			roi = SSDnet.predict(frame);
			if (roi.empty())
				continue;
			// cam.display(frame, roi, "OK.jpg");

			// 目标框中心
			cam.calculatePosition(roi, frame, delta_position);
			cout << "delta_x: " << delta_position[0] << '\t' << "delta_y: " << delta_position[1] << endl;
			
			// 发现目标，发送目标位置
			devs->send_object_position_command(delta_position, image_size);
			is_detected = true;

			// 等待 robot 完成指定工作
			while (!robot_completed) ;
			robot_completed = false;

			// 发送行走指令
			devs->send_motion_command(MOTION_FORWARD);
		}
		else if (delta_position[0] > 0)
			is_detected = false;

		// 按 ESC 键退出
		while (waitKey(1) == 27)
		{
			delete []delta_position;
			delete []image_size;
			return 0;
		}
	}
	
}