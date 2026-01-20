#pragma once
#include <iostream>
#include <list>
#include <string>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <thread>
#include "log_helpers.h"
#include "TypeDef.h"
#include "AAC.h"
#include "MediaInterface.h"
#include "RtmpCommon.h"
extern "C" {
#include "flv.h"
#include "amf0.h"
#include "rtmp.h"
}
void audioCallBack(enum FLVAudioType type, int profile, int sample_rate_index, int channel, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg);
void videoCallBack(enum FLVVideoType type, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg);
void scriptDataCallBack(AMFDict dict, void* arg);
class RtmpPullClient{
public:
    RtmpPullClient(std::string url, FLVOutMode type = FLV_RTMP);
    ~RtmpPullClient();
    void GetVideoCon(int &width, int &height, int &fps){width = width_; height = height_; fps = fps_;}
    void GetAudioCon(int &sample_rate_index, int &channels, int &profile){sample_rate_index = sample_rate_index_; channels = channels_; profile = profile_;}
    enum VideoType GetVideoType(){return video_type_;}
    enum AudioType GetAudioType(){return audio_type_;}
    void SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb){data_listner_ = lisnter; colse_cb_ = cb; return;}
private:
    void FLVStreamReadThread();
private:
    std::string url_;
    FLVOutMode type_;
    FLVContext *context_demuxer_ = nullptr; // libflv非线程安全
    std::thread th_flv_handle_;
    bool abort_ = false;
    MediaDataListner *data_listner_ = nullptr;
    CloseCallbackFunc colse_cb_ = nullptr;
    enum VideoType video_type_ = VideoType::VIDEO_NONE;
    int width_ = -1;
    int height_ = -1;
    int fps_ = -1;
    enum AudioType audio_type_ = AudioType::AUDIO_NONE;
    int profile_ = -1;
    int sample_rate_index_ = -1;
    int channels_ = -1;
    std::atomic<uint64_t> start_timestamp_ = {0};
    uint8_t *vps_ = nullptr;
    uint8_t *sps_ = nullptr;
    uint8_t *pps_ = nullptr;
    int vps_buffer_len_ = 0;
    int sps_buffer_len_ = 0;
    int pps_buffer_len_ = 0;
    int vps_len_ = 0;
    int sps_len_ = 0;
    int pps_len_ = 0;
    bool probe_over_flag_ = false;
    int probe_frame_num_ = 50; // 探测帧数，用于计算视频fps
    int64_t last_timestamp_ = -1;
    int64_t interval_sum_ = 0;
    int probe_cnt_ = 0;
     
    friend void audioCallBack(enum FLVAudioType type, int profile, int sample_rate_index, int channel, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg);
    friend void videoCallBack(enum FLVVideoType type, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg);
    friend void scriptDataCallBack(AMFDict dict, void* arg);
};