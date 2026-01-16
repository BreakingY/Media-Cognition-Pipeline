#pragma once
#include <iostream>
#include <list>
#include <string>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <thread>
#include "MediaInterface.h"
#include "log_helpers.h"
#include "TypeDef.h"
#include "AAC.h"
extern "C" {
#include "flv.h"
#include "amf0.h"
}
void WriteCallBack(enum FLVWriteType type, uint8_t* data, uint32_t data_len, void* arg);
class RtmpPushClient{
public:
    RtmpPushClient(std::string rtmp_url);
    ~RtmpPushClient();
    // h264/h265
    void SetVideoInfo(enum VideoType type);
    void InputVideoData(uint8_t *data, int data_len, int64_t timestamp); // with startcode
    // aac
    void SetAudioInfo(enum AudioType type);
    void InputAudioData(uint8_t *data, int data_len, int64_t timestamp); // with adts
private:
    int ConnectServer();
    void CloseConnect();
    int OpencvFLVHandle();
    void CloseFLVHandle();
    void VideoStreamThread();
    void AudioStreamThread();
    void RtmpReconnectThread();
private:
    typedef struct MediaDataSt {
        unsigned char *data;
        int data_len;
        int64_t pts;
        int64_t dts;
    } MediaData;
    bool abort_ = false;
    std::string rtmp_url_;
    FLVContext *context_muxer_ = nullptr; // libflv非线程安全
    std::mutex flv_mtx_;

    std::list<MediaData> video_list_;
    std::mutex video_mtx_;
    std::condition_variable video_cond_;
    std::thread th_video_;
    bool video_ready_ = false;
    enum VideoType video_type_;
    uint8_t *vps_ = nullptr;
    uint8_t *sps_ = nullptr;
    uint8_t *pps_ = nullptr;
    int vps_buffer_len_ = 0;
    int sps_buffer_len_ = 0;
    int pps_buffer_len_ = 0;
    int vps_len_ = 0;
    int sps_len_ = 0;
    int pps_len_ = 0;

    std::list<MediaData> audio_list_;
    std::mutex audio_mtx_;
    std::condition_variable audio_cond_;
    std::thread th_audio_;
    bool audio_ready_ = false;
    enum AudioType audio_type_;
    std::atomic<uint64_t> start_timestamp = {0};

    std::atomic<bool> rtmp_connect_stat_ = {false};
    std::thread th_rtmp_reconnect_;
    uint8_t *send_buffer_[1024 * 1024 * 4];
    int send_buffer_len_ = 0;
    bool skip_flv_header_ = false;
    
    friend void WriteCallBack(enum FLVWriteType type, uint8_t* data, uint32_t data_len, void* arg);
};