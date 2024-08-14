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
/**
 *   Init-> AddVideo/AddAudio->Open->SendHeader->SendPacket->SendTrailer
 *   input without startCode
 */
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
    Muxer();
    ~Muxer();

    int Init(const char *url);
    void DeInit();

    int AddVideo(int time_base, VideoType type, ExtraData &extra, int width, int height, int fps); // H264 h265
    int AddAudio(int channels, int sample_rate, int profile, AudioType type);            // AAC
    int Open();

    int SendHeader();
    int SendPacket(unsigned char *data, int size, int64_t pts, int64_t dts, int stream_index); // video:one NALU without startCodes
    int SendTrailer();

    int GetAudioStreamIndex();
    int GetVideoStreamIndex();

private:
    void H264WriteExtra(unsigned char *extra_data, int &extra_data_size);
    void H265WriteExtra(unsigned char *extra_data, int &extra_data_size);
    void RewriteVideoExtraData();
    void AACWriteExtra(int channels, int sample_rate, int profile, AVCodecParameters *params);
    bool ParametersChange(unsigned char *vps, int vps_len, unsigned char *sps, int sps_len, unsigned char *pps, int pps_len);

public:
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
    VideoType video_type_;
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

    std::mutex mtx_;
    bool found_idr_ = false;
    AVPacket pkt_;
    bool global_header_ = false;
};

#endif