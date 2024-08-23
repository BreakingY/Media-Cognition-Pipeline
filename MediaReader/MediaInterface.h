#ifndef MEDIAINTERFACE_H
#define MEDIAINTERFACE_H
#include <functional>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct AudioDataSt {
    unsigned char *data;
    int data_len;
    int64_t pts;
    int64_t dts;
    int profile;
    int samplerate;
    int channels;
} AudioData;
typedef struct VideoDataSt {
    unsigned char *data;
    int data_len;
    int64_t pts;
    int64_t dts;
} VideoData;
class MediaDataListner
{
public:
    virtual void OnVideoData(VideoData data) = 0; // with startcode
    virtual void OnAudioData(AudioData data) = 0; // with adts
};
using CloseCallbackFunc = std::function<void(void)>;

#endif