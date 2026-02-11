#ifndef VIDEOREADER_H
#define VIDEOREADER_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <list>
#include <mutex>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>
#include <libavutil/time.h>
}
#include "TypeDef.h"
#include "MediaInterface.h"
#include "AAC.h"
#include "log_helpers.h"
using namespace std::chrono_literals;
static const uint64_t NANO_SECOND = UINT64_C(1000000000);

enum BufFrame_e {
    READ = 1,
    WRITE,
    OVER,
};

struct BufSt {
    unsigned char *buf;
    int buf_len;
    int stat;
    int pos;
};
struct FrameSt {
    unsigned char *frame;
    int frame_len;
    int startcode;
    int stat;
};

class MediaReader
{
public:
    MediaReader(const char *file_path);
    enum VideoType GetVideoType();
    enum AudioType GetAudioType();
    virtual ~MediaReader();
    void SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb);
    void GetVideoCon(int &width, int &height, int &fps);
    void GetAudioCon(int &channels, int &sample_rate, int &profile, int &bit_per_sample);
    
private:
    static void *MediaReaderThread(void *arg);
    void PraseFrame();
    void VideoInit(const char *filename);

private:
    std::string file_;
    struct BufSt *buffer_ = NULL;
    struct FrameSt *frame_ = NULL;
    std::thread th_file_;
    std::atomic<bool> file_finish_ = {false};
    bool abort_ = false;
    MediaDataListner *data_listner_ = NULL;
    CloseCallbackFunc colse_cb_ = NULL;

    AVFormatContext *format_ctx_;
    AVPacket packet_;
    bool is_mp4_;
    // H264 H265
    int video_index_ = -1;
    int fps_ = 25;
    AVBSFContext *bsf_ctx_ = NULL;
    // AAC
    int audio_index_ = -1;
};

#endif