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
#include "MediaInterface.h"
#include "AAC.h"
#include "TsCommon.h"
extern "C" {
#include "mpeg2core_ts.h"
#include "mpeg2core_common.h"
}
class TsDemuxerClient{
public:
    TsDemuxerClient(std::string url, TSMode type = TS_FILE);
    ~TsDemuxerClient();
    void GetVideoCon(int &width, int &height, int &fps){width = width_; height = height_; fps = fps_;}
    void GetAudioCon(int &sample_rate_index, int &channels, int &profile){sample_rate_index = sample_rate_index_; channels = channels_; profile = profile_;}
    enum VideoType GetVideoType(){return video_type_;}
    enum AudioType GetAudioType(){return audio_type_;}
    void SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb){data_listner_ = lisnter; colse_cb_ = cb; return;}
private:
    int OpenTsHandle();
    void CloseTsHandle();
    void TsStreamReadThread();
    // for protocol
    void ReconnectThread();
    int ConnectServer();
    void CloseConnect();
private:
    std::string url_;
    TSMode type_;
    mpeg2_ts_context *context_demuxer_ = nullptr; // libmpeg2core非线程安全
    int save_program_number_ = -1;
    std::thread th_ts_handle_;
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
    int64_t start_timestamp_video_ = -1;
    int64_t start_timestamp_audio_ = -1;
    int64_t last_timestamp_video_ = -1;
    int64_t last_timestamp_audio_ = -1;
    
    bool probe_over_flag_ = false;
    bool media_ready_ = false;
    bool media_over_ = false;
    int probe_frame_num_ = 50; // 探测帧数，用于计算视频fps
    int64_t last_timestamp_ = -1;
    int64_t interval_sum_ = 0;
    int probe_cnt_ = 0;

     FILE *ts_fd_ = nullptr;

    // for protocol
    std::atomic<bool> connect_stat_ = {false};
    std::thread th_reconnect_;
    bool video_ready_ = false;
    uint8_t recv_buffer_[1024 * 1024 * 4];
     
    friend void video_read_callback(int program_number, int stream_pid, int type, int64_t pts, int64_t dts, uint8_t *data, int data_len, void *arg);
    friend void audio_read_callback(int program_number, int stream_pid, int type, int64_t pts, int64_t dts, uint8_t *data, int data_len, void *arg);
};