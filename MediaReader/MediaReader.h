#ifndef VIDEOREADER_H
#define VIDEOREADER_H
#include "MediaInterface.h"
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
using namespace std::chrono_literals; // 时间库由C++14支持
static const uint64_t NANO_SECOND = UINT64_C(1000000000);
#define BUFF_MAX 50000000
#define DEBUGPRINT printf
/*buf和frame的状态*/
enum BufFrame_e {
    READ = 1,
    WRITE,
    OVER, // 文件读取完毕
};
/*MP4缓冲区*/
struct buf_st {
    unsigned char buf[BUFF_MAX];
    int bufsize;
    int stat; // buf状态,READ表示可读 WRITE表示可写
    int pos;  ////frame读取buf的位置记录
};
/*NALU数据读取*/
struct frame_st {
    unsigned char frame[BUFF_MAX];
    int frameSize;
    int startCode;
    int stat;
};

class MediaReader
{
public:
    MediaReader() = delete;
    MediaReader(char *file_path);
    enum VideoType getVideoType();
    enum AudioType getAudioType();
    virtual ~MediaReader();
    static void *MediaReaderThread(void *arg);
    static void *VideoSyncThread(void *arg);
    static void *AudioSyncThread(void *arg);
    static void *CheckThread(void *arg);
    void praseFrame();
    void setDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb);
    void VideoInit(char *filename);
    int mp4toannexb(AVPacket &m_packet);
    bool haveAudio();
    void getAudioCon(int &channels, int &sample_rate, int &audio_object_type, int &bit_per_sample);
    void reset();

public:
    std::string file;
    struct buf_st buffer;
    struct frame_st frame;
    std::thread th_file;
    std::thread th_video;
    std::thread th_audio;
    bool Loop;
    std::atomic<bool> videoFinish = {false};
    std::atomic<bool> audioFinish = {false};
    std::atomic<bool> fileFinish = {false};
    bool abort = false;
    MediaDataListner *DataListner = NULL;
    CloseCallbackFunc colseCb = NULL;

    AVFormatContext *m_pFormatCtx;
    AVPacket m_packet;
    bool isMP4;
    // H264 H265
    int video_index = -1;
    int m_fps = 25;
    AVBitStreamFilterContext *m_ph26xbsfc;

    std::list<AVPacket> video_list;
    std::mutex video_mtx;
    std::condition_variable video_cond;
    int64_t video_start_timestamp = -1;

    // AAC
    int audio_index = -1;
    std::list<AVPacket> audio_list;
    std::mutex audio_mtx;
    std::condition_variable audio_cond;
    int64_t audio_start_timestamp = -1;

    std::atomic<int64_t> video_now_time = {0};
    std::atomic<int64_t> audio_now_time = {0};
    int sync_threshold = 0;
    bool audio_reset = false;
    bool video_reset = false;
};

#endif