/*
 * @Descripttion: 
 * @version: 
 * @Author: LuoChuan
 * @Date: 2021-03-19 20:21:28
 * @LastEditors: LuoChuan
 * @LastEditTime: 2021-03-26 16:48:17
 */

#include "move.h"

int check_Sum(char* frame, int len)
{
    int sum = 0;
    for (int i = 0; i < len-1; i++)
        sum += frame[i];
        
    return sum;
}

void delay_ms(int time_ms)
{
	clock_t now = clock();
	while (clock() - now < time_ms) ; 
}
