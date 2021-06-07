/*
 * @Description: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-22 15:37:57
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-04-01 09:36:43
 */

#include "device.hpp"

typedef struct {
    bool* robot_completed;;
    DevicesModule* ptr;
    // pthread_mutex_t *p_mutex;
} ThreadRobotParam;

DevicesModule::DevicesModule(bool* robot_completed)
{
    pthread_mutex_init(&mutex, nullptr);
    ThreadRobotParam param {
        .robot_completed = robot_completed,
        .ptr = this
        // .p_mutex = p_mutex
    };
    pthread_create(&robot_thread, NULL, thread_function_robot, (void*)(&param));

    char* agvWheelPath = "/dev/agv_wheel";
    move_fd = open_tty(agvWheelPath, 9600, 0);

    char* robotPath = "/dev/agv_nav";
    robot_fd = open_tty(robotPath, 9600, 0);
}

DevicesModule::~DevicesModule()
{

    thread_running = false;
    pthread_join(robot_thread, NULL);

    if (move_fd > 0)
    {
        close(move_fd);
    }
    if (robot_fd > 0)
    {
        close(robot_fd);
    }
    pthread_mutex_destroy(&mutex);
}

void *DevicesModule::thread_function_robot(void* param)
{
    ThreadRobotParam* p = (ThreadRobotParam *)param;
    bool* robot_completed = p->robot_completed;
    DevicesModule* ptr = p->ptr;

    int byte_index = 0, len = 0, new_frame = 0;
    char in[100];
    char frame[100];

    while (ptr->thread_running)
    {
        cout << "############### thread_running ###############" << endl;

        int ret = read(ptr->robot_fd, in, 8);

        if (ret > 0)
        {
            for (int i = 0; i < ret; i++)
            {
                if (new_frame == 0)
                {
                    if (in[i] == 0x13)
                    {
                        byte_index = 0;
                        len = 0;
                        new_frame = 1;
                    }
                }
                else if (new_frame == 1)
                {
                    frame[byte_index] = in[i];
                    byte_index++;
                    if (byte_index == 1)
                        len = in[i];
                    if (byte_index == len)
                    {
                        if (check_Sum(frame, len) == frame[len-1])
                        {
                            for (int j = 0; j < len; j++)
		                    {
			                    cout << "received frame[" << j << "]: " << (int)frame[j] << endl;
		                    }
                            *robot_completed = ptr->handle_frame_robot(frame);
                        }
                        memset(frame, 0, 100);
                        new_frame = 0;
                    }
                }
            }
        }
        cout << "############### thread_end ###############" << endl;
    }

    return nullptr;
}

bool DevicesModule::handle_frame_robot(char *frame)
{
    bool robot_work_status = false;
    if (frame[2] == 0x01)   // 0x13 0x03 0x01 0x04  机器人完成指定工作后，小车继续前进
    {
        robot_work_status = true;
    }
    else                    // 0x13 0x03 0x00 0x04  待修改
    {
        robot_work_status = true;
    }

    return robot_work_status;
}

speed_t DevicesModule::getBaudrate(int baudrate)
{
    switch (baudrate)
    {
    case 0:
        return B0;
    case 50:
        return B50;
    case 75:
        return B75;
    case 110:
        return B110;
    case 134:
        return B134;
    case 150:
        return B150;
    case 200:
        return B200;
    case 300:
        return B300;
    case 600:
        return B600;
    case 1200:
        return B1200;
    case 1800:
        return B1800;
    case 2400:
        return B2400;
    case 4800:
        return B4800;
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
    case 230400:
        return B230400;
    case 460800:
        return B460800;
    case 500000:
        return B500000;
    case 576000:
        return B576000;
    case 921600:
        return B921600;
    case 1000000:
        return B1000000;
    case 1152000:
        return B1152000;
    case 1500000:
        return B1500000;
    case 2000000:
        return B2000000;
    case 2500000:
        return B2500000;
    case 3000000:
        return B3000000;
    case 3500000:
        return B3500000;
    case 4000000:
        return B4000000;
    default:
        return -1;
    }
}

int DevicesModule::open_tty(char const *path, int baudrate, int flags)
{
    int fd;
    speed_t speed;

    if ((speed = getBaudrate(baudrate)) == -1)
    {
        printf("FAILED  to open %s\n", path);
        return -1;
    }
    if ((fd = open(path, O_RDWR | flags)) == -1)
    {
        printf("FAILED  to open %s\n", path);
        return -1;
    }

    struct termios cfg;
    if (tcgetattr(fd, &cfg))
    {
        close(fd);
        printf("FAILED  to open %s\n", path);
        return -1;
    }
    cfmakeraw(&cfg);
    cfsetispeed(&cfg, speed);
    cfsetospeed(&cfg, speed);
    cfg.c_cc[VMIN] = 1;
    cfg.c_cc[VTIME] = 1;
    if (tcsetattr(fd, TCSANOW, &cfg))
    {
        close(fd);
        printf("FAILED  to open %s\n", path);
        return -1;
    }

    printf("open %s ok,fd=%d\n", path, fd);
    return fd;
}

bool str_compare(char *s1, char *s2)
{
    int a = 0, b = 0;
    while (1)
    {
        if (s1[a] == 0 || s2[b] == 0)
        {
            if (s1[a] == 0 && s2[b] == 0) return true;
            else return false;
        }
        if (s1[a] != s2[a]) return false;
        a++;
        b++;
    }
    return true;
}

int DevicesModule::get_fd(char* devs)
{
    if (str_compare(devs, "agvwheel"))
        return move_fd;
    if (str_compare(devs, "robot"))
        return robot_fd;
}

void DevicesModule::send_motion_command(int motion_status)
{
	if (motion_status == 0)
	{
		stop_command();
		cout << "Stop!!!!!!" << endl;
		sleep(1);   // 等待小车稳定停下，精准检测目标
	}
	else if (motion_status == 1)
	{
		go_command();
		cout << "Go!!!!!!" << endl;
		sleep(1);	// 等待小车行驶，摆脱当前检测目标
	}
}

void DevicesModule::stop_command()
{
	char out[20] = {0x02, 0x09, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
		
	for (int i = 1; i < 9; i++) // 校验和
	{
		out[9] += out[i];
	}
	out[10] = 0;
	write(move_fd, out, 10);
}

void DevicesModule::go_command()
{
    // 0.5m/s: 01 F4; 0.8m/s: 03 20; 1.0m/s: 03 E8
	char out[20] = {0x02, 0x09, 0x01, 0x01, 0x00, 0x96, 0x01, 0x00, 0x96};
		
	for (int i = 1; i < 9; i++) // 校验和
	{
		out[9] += out[i];
	}
	out[10] = 0;
	write(move_fd, out, 10);
}

void DevicesModule::send_object_position_command(int* position, int* image_size)
{
	int delta_x = position[0];
	int delta_y = position[1];
	int image_width = image_size[0];
	int image_height = image_size[1];

	char out[20];
	int temp = 0;
	out[0] = 0x03; 									// 起始符

	out[1] = 0x12; 									// 数据长度

	out[2] = (delta_x >= 0) ? 1 : 0; 				// x方向
	temp = (delta_x >= 0) ? delta_x : 0 - delta_x;
	out[3] = temp / 256; 							// x
	out[4] = temp % 256;

	out[5] = (delta_y >= 0) ? 1 : 0; 				// y方向
	temp = (delta_y >= 0) ? delta_y : 0 - delta_y;
	out[6] = temp / 256; 							// y
	out[7] = temp % 256;

	temp = (image_width > 0) ? image_width : 640;
	out[8] = temp / 256; 							// image_width
	out[9] = temp % 256;

	temp = (image_height > 0) ? image_height : 480;
	out[10] = temp / 256; 							// image_height
	out[11] = temp % 256;

	for (int i = 1; i < 12; i++) 					// 校验和
	{
		out[12] += out[i];
	}
	out[13] = 0;
	write(robot_fd, out, 13);

    cout << "check: " << (int)out[12] << endl;
	cout << "Send object position OK!!!!!!" << endl;
}