#ifndef DEC_ENC_INTERFACE_H
#define DEC_ENC_INTERFACE_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
// 解码后数据接口
class DecDataCallListner
{
public:
    virtual void OnRGBData(cv::Mat frame) = 0;
    virtual void OnPCMData(unsigned char **data, int data_len) = 0; // data是原生的输出数据，指针数组，data_len是单通道样本个数
};
// 编码后数据接口
class EncDataCallListner
{
public:
    virtual void OnVideoEncData(unsigned char *data, int data_len, int64_t pts /*deprecated*/) = 0;
    virtual void OnAudioEncData(unsigned char *data, int data_len) = 0;
};
#endif