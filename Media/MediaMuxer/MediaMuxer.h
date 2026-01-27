#ifndef MEDIAMUXER_H
#define MEDIAMUXER_H
#include <iostream>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
};
#include "TypeDef.h"
#include "log_helpers.h"
typedef struct ExtraDataSt {
    uint8_t *vps = NULL;
    uint8_t *sps = NULL;
    uint8_t *pps = NULL;
    int vps_len = -1;
    int sps_len = -1;
    int pps_len = -1;
} ExtraData;
class Muxer
{
public:
    Muxer(const char *url);
    ~Muxer();
public:
    /**
     * thread-safe
     * SetMediaInfo-->WriteVideo2File/WriteAudio2File
     */
    // h264/h265 aac
    void SetMediaInfo(enum VideoType video_type, enum AudioType audio_type);
    int WriteVideo2File(uint8_t *data_nalus, int len_nalus); // h264/h265 with startcode
    int WriteAudio2File(uint8_t *data, int len);// aac with adts
public:
    /**
     * non-thread-safe
     * AddVideo/AddAudio-->Open-->SendHeader-->SendPacket-->SendTrailer
     */
    int AddVideo(int time_base, VideoType type, ExtraData &extra, int width, int height); // h264 h265
    int AddAudio(int channels, int sample_rate, int profile, AudioType type);            // aac
    int Open();

    int SendHeader();
    int SendPacket(unsigned char *data, int size, int64_t pts, int64_t dts, int stream_index); // video:one NALU without startcode; audio:aac without adts
    int SendTrailer();

    int GetAudioStreamIndex();
    int GetVideoStreamIndex();
private:
    int Init(const char *url);
    void DeInit();

    void H264WriteExtra(unsigned char *extra_data, int &extra_data_size);
    void H265WriteExtra(unsigned char *extra_data, int &extra_data_size);
    void RewriteVideoExtraData();
    void AACWriteExtra(int channels, int sample_rate, int profile, AVCodecParameters *params);
    bool ParametersChange(unsigned char *vps, int vps_len, unsigned char *sps, int sps_len, unsigned char *pps, int pps_len);

private:
    AVFormatContext *fmt_ctx_ = NULL;
    AVOutputFormat *ofmt_ = NULL;
    std::string url_ = "";

    AVStream *aud_stream_ = NULL;
    AudioType audio_type_;
    int frames_video_ = 0;
    int64_t start_pts_video_ = 0;
    int64_t start_dts_video_ = 0;
    int64_t last_pts_video_ = 0;
    int64_t last_dts_video_ = 0;

    AVStream *vid_stream_ = NULL;
    enum VideoType video_type_;
    int frames_audio_ = 0;
    int64_t start_pts_audio_ = 0;
    int64_t start_dts_audio_ = 0;
    int64_t last_pts_audio_ = 0;
    int64_t last_dts_audio_ = 0;

    int64_t start_media_pts_ = 0;
    bool find_first_frame_ = false;

    int audio_index_ = -1;
    int video_index_ = -1;

    uint8_t *vps_buf_[16];  // max 16
    uint8_t *sps_buf_[32];  // max 32
    uint8_t *pps_buf_[256]; // max 256
    int vps_len_[16];
    int sps_len_[32];
    int pps_len_[256];
    int vps_number_ = 0;
    int sps_number_ = 0;
    int pps_number_ = 0;
    int width_;
    int height_;

    uint8_t *vps_last_ = nullptr;
    uint8_t *sps_last_  = nullptr;
    uint8_t *pps_last_  = nullptr;
    int vps_last_buffer_len_ = 0;
    int sps_last_buffer_len_ = 0;
    int pps_last_buffer_len_ = 0;
    int vps_last_len_ = 0;
    int sps_last_len_ = 0;
    int pps_last_len_ = 0;
    bool video_ready_ = false;
    bool audio_ready_ = false;
    bool have_video_ = true;
    bool have_audio_ = true;
    // video
    std::chrono::steady_clock::time_point time_now_video_;
    std::chrono::steady_clock::time_point time_pre_video_;
    uint64_t nframe_counter_video_ = 0;
    uint64_t time_ts_accum_video_ = 0;
    // audio
    std::chrono::steady_clock::time_point time_now_audio_;
    std::chrono::steady_clock::time_point time_pre_audio_;
    uint64_t nframe_counter_audio_ = 0;
    uint64_t time_ts_accum_audio_ = 0;

    int video_stream_ = -1;
    int audio_stream_ = -1;

    std::mutex mtx_;
    bool write_header_flag_ = false;
    bool found_idr_ = false;
    AVPacket pkt_;
    bool global_header_ = false;
    bool write_trailer_ = false;
};

#endif