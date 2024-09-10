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
using namespace std::chrono_literals; // 时间库由C++14支持
static const uint64_t NANO_SECOND = UINT64_C(1000000000);
#define DEBUGPRINT printf
/*buf和frame的状态*/
enum BufFrame_e {
    READ = 1,
    WRITE,
    OVER, // 文件读取完毕
};
/*MP4缓冲区*/
struct BufSt {
    unsigned char *buf;
    int buf_len;
    int stat; // buf状态,READ表示可读 WRITE表示可写
    int pos;  ////frame读取buf的位置记录
};
/*NALU数据读取*/
struct FrameSt {
    unsigned char *frame;
    int frame_len;
    int startcode;
    int stat;
};

class MediaReader
{
public:
    MediaReader() = delete;
    MediaReader(char *file_path);
    enum VideoType GetVideoType();
    enum AudioType GetAudioType();
    virtual ~MediaReader();
    void SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb);
    bool HaveAudio();
    void GetVideoCon(int &width, int &height, int &fps);
    void GetAudioCon(int &channels, int &sample_rate, int &profile, int &bit_per_sample);
    void Reset();
    
private:
    static void *MediaReaderThread(void *arg);
    static void *VideoSyncThread(void *arg);
    static void *AudioSyncThread(void *arg);
    static void *CheckThread(void *arg);
    void PraseFrame();
    void VideoInit(char *filename);

private:
    std::string file_;
    struct BufSt *buffer_ = NULL;
    struct FrameSt *frame_ = NULL;
    std::thread th_file_;
    std::thread th_video_;
    std::thread th_audio_;
    bool loop_;
    std::atomic<bool> video_finish_ = {false};
    std::atomic<bool> audio_finish_ = {false};
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

    std::list<AVPacket> video_list_;
    std::mutex video_mtx_;
    std::condition_variable video_cond_;
    int64_t video_start_timestamp_ = -1;

    // AAC
    int audio_index_ = -1;
    std::list<AVPacket> audio_list_;
    std::mutex audio_mtx_;
    std::condition_variable audio_cond_;
    int64_t audio_start_timestamp_ = -1;

    std::atomic<int64_t> video_now_time_ = {0};
    std::atomic<int64_t> audio_now_time_ = {0};
    int sync_threshold_ = 0;
    bool audio_reset_ = false;
    bool video_reset_ = false;
};

#endif