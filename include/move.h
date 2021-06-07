/*
 * @Description: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-19 20:15:35
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-03-26 16:48:05
 */

#ifndef MOVE_H
#define MOVE_H

#include <unistd.h>
#include <ctime>


typedef enum MOTION_STATUS_LIST
{
    MOTION_STOP            =0, 
    MOTION_FORWARD         =1, 
    MOTION_BACKWARD        =2, 
    MOTION_LEFT            =3, 
    MOTION_RIGHT           =4, 
    MOTION_LEFT_FRONT      =5, 
    MOTION_RIGHT_FRONT     =6, 
    MOTION_LEFT_REAR       =7, 
    MOTION_RIGHT_REAR      =8,
} MOTION_STATUS_LIST;
typedef int MOTION_STATUS;

/**
 * @description: 
 * @param {int} time_ms
 * @return {*}
 */
void delay_ms(int time_ms);

/**
 * @description: 
 * @param {int} len
 * @return {*}
 */
int check_Sum(char* frame, int len);

#endif