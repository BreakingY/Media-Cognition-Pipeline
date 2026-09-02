#ifndef DEC_ENC_INTERFACE_H
#define DEC_ENC_INTERFACE_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
typedef enum CODEC_TYPE_e{
    CODEC_NONE = 0,
    CODEC_H264,
    CODEC_H265,
}CODEC_TYPE;
// 解码后数据接口
class DecDataCallListner
{
public:
    virtual void OnRGBData(cv::Mat frame, int64_t timestamp/*ms*/) = 0; // timestamp解码前时间戳(包含解码时延)
    virtual void OnPCMData(unsigned char **data, int data_len, int64_t timestamp/*ms*/) = 0; // data是原生的输出数据，指针数组，data_len是单通道样本个数，timestamp解码前时间戳(包含解码时延)
};
// 编码后数据接口
class EncDataCallListner
{
public:
    virtual void OnVideoEncData(unsigned char *data, int data_len, int64_t pts) = 0;
    virtual void OnAudioEncData(unsigned char *data, int data_len, int64_t pts) = 0;
};
#endif