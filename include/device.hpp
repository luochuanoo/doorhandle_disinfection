/*
 * @Description: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-22 15:33:13
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-03-28 16:42:30
 */

#ifndef DEVICE_H
#define DEVICE_H

#include <iostream>
#include <pthread.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include "move.h"

using namespace std;

class DevicesModule 
{
public:
    DevicesModule(bool *robot_completed);
    ~DevicesModule();
    
    static void *thread_function_robot(void *param);
    bool handle_frame_robot(char *frame);
    speed_t getBaudrate(int baudrate);
    int open_tty(char const *path, int baudrate, int flags);
    int get_fd(char* devs);

    void send_motion_command(int motion_status);
    void stop_command();
    void go_command();
    void send_object_position_command(int* position, int* image_size);


private:
    int move_fd = -1;
    int robot_fd = -1;

    pthread_mutex_t mutex;
    pthread_t robot_thread;
    bool thread_running = true;
};

#endif