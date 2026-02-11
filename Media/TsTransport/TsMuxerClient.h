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
#include "TsCommon.h"
extern "C" {
#include "mpeg2core_ts.h"
}
#if defined(ENABLE_LIBSRT)
#include <srt/srt.h>
#if defined(_WIN32) || defined(_WIN64)
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#endif
void media_write_callback(int program_number, int stream_pid, int stream_type, uint8_t *data, int data_len, void *arg);
class TsMuxerClient{
public:
    TsMuxerClient(std::string url, TSMode type = TS_FILE);
    ~TsMuxerClient();
    // h264/h265
    void SetVideoInfo(enum VideoType type);
    void InputVideoData(uint8_t *data, int data_len, int64_t timestamp); // with startcode
    // aac(with adts) g711a g711u
    void SetAudioInfo(enum AudioType type);
    void InputAudioData(uint8_t *data, int data_len, int64_t timestamp);
private:
    // for protocol
    int ConnectServer();
    void CloseConnect();
    void ReconnectThread();

    int OpenTsHandle();
    void CloseTsHandle();

    void VideoStreamThread();
    void AudioStreamThread();

private:
    typedef struct MediaDataSt {
        unsigned char *data;
        int data_len;
        int64_t pts;
        int64_t dts;
    } MediaData;
    bool abort_ = false;
    TSMode type_;
    std::string url_;
    mpeg2_ts_context *context_muxer_ = nullptr; // libmpeg2core非线程安全
    uint16_t program_number_ = 1;
    int stream_type_video_;
    int stream_type_audio_;
    int stream_pid_video_ = -1;
    int stream_pid_audio_ = -1;
    bool file_rewrit_PSI_flag_ = false;
    std::mutex ts_mtx_;
    FILE *ts_fd_ = nullptr;

    std::list<MediaData> video_list_;
    std::mutex video_mtx_;
    std::condition_variable video_cond_;
    std::thread th_video_;
    bool video_ready_ = false;
    enum VideoType video_type_;

    std::list<MediaData> audio_list_;
    std::mutex audio_mtx_;
    std::condition_variable audio_cond_;
    std::thread th_audio_;
    bool audio_ready_ = false;
    enum AudioType audio_type_;

    std::atomic<uint64_t> start_timestamp_ = {0};
    
    friend void media_write_callback(int program_number, int stream_pid, int stream_type, uint8_t *data, int data_len, void *arg);

    // protocol 
    std::atomic<bool> connect_stat_ = {false};
    std::thread th_reconnect_;

    #if defined(ENABLE_LIBSRT)
    SRTSOCKET sock_;
    uint64_t last_stat_ = 0;
    uint64_t now_ = 0;
    #endif
};