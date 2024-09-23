#ifndef _HARD_DEC_H
#define _HARD_DEC_H

#include "DecEncInterface.h"
#include "log_helpers.h"
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
// #define USE_FFMPEG_NVIDIA
typedef struct HardDataNodeSt {
    unsigned char *es_data;
    int es_data_len;
    HardDataNodeSt()
    {
        es_data = NULL;
        es_data_len = 0;
    }
    virtual ~HardDataNodeSt()
    {
        if (es_data) {
            free(es_data);
            es_data = NULL;
        }
    }
} HardDataNode;
#ifdef USE_FFMPEG_NVIDIA
typedef enum AVPixelFormat (*get_format)(struct AVCodecContext *s, const enum AVPixelFormat *fmt);
// 支持cuda硬解码加速，如果ffmpeg或者显卡不支持自动切换软解码
class HardVideoDecoder
{

public:
    HardVideoDecoder(bool is_h265 = false);
    virtual ~HardVideoDecoder();
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    int HardDecInit(bool is_h265 = false);
    int SoftDecInit(bool is_h265 = false);
    int hwDecoderInit(AVCodecContext *ctx, const enum AVHWDeviceType type);
    static void *DecodeThread(void *arg);
    void DecodeVideo(HardDataNode *data);
    static void *ScaleThread(void *arg);
    void ScaleVideo(AVFrame *frame);

private:
    bool is_hard_ = false;
    enum AVCodecID decodec_id_;
    DecDataCallListner *callback_ = NULL;
    AVCodecContext *codec_ctx_ = NULL;
    AVCodec *codec_ = NULL;

    AVPacket packet_;
    AVFrame *frame_ = NULL;
    AVFrame *sw_frame_ = NULL;
    struct SwsContext *img_convert_ctx_ = NULL;
    enum AVPixelFormat out_pix_fmt_ = AV_PIX_FMT_NONE;
    // hard dec
    enum AVHWDeviceType type_ = AV_HWDEVICE_TYPE_NONE;
    enum AVPixelFormat hw_pix_fmt_;
    AVBufferRef *hw_device_ctx_ = NULL;

    pthread_mutex_t packet_mutex_;
    pthread_cond_t packet_cond_;
    pthread_cond_t frame_cond_;
    pthread_mutex_t frame_mutex_;
    std::list<HardDataNode *> es_packets_;
    std::list<AVFrame *> yuv_frames_;
    pthread_t dec_thread_id_;
    pthread_t sws_thread_id_;
    bool abort_;

    int now_frames_;
    int pre_frames_;
    struct timeval time_now_;
    struct timeval time_pre_;
    int time_inited_;

    unsigned char *image_ptr_ = NULL;
};
#endif
#ifdef USE_FFMPEG_SOFT
class HardVideoDecoder
{

public:
    HardVideoDecoder(bool is_h265 = false);
    virtual ~HardVideoDecoder();
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    int SoftDecInit(bool is_h265 = false);
    static void *DecodeThread(void *arg);
    void DecodeVideo(HardDataNode *data);
    static void *ScaleThread(void *arg);
    void ScaleVideo(AVFrame *frame);

private:
    enum AVCodecID decodec_id_;
    DecDataCallListner *callback_ = NULL;
    AVCodecContext *codec_ctx_ = NULL;
    AVCodec *codec_ = NULL;

    AVPacket packet_;
    AVFrame *frame_ = NULL;
    struct SwsContext *img_convert_ctx_ = NULL;
    enum AVPixelFormat out_pix_fmt_ = AV_PIX_FMT_NONE;

    pthread_mutex_t packet_mutex_;
    pthread_cond_t packet_cond_;
    pthread_cond_t frame_cond_;
    pthread_mutex_t frame_mutex_;
    std::list<HardDataNode *> es_packets_;
    std::list<AVFrame *> yuv_frames_;
    pthread_t dec_thread_id_;
    pthread_t sws_thread_id_;
    bool abort_;

    int now_frames_;
    int pre_frames_;
    struct timeval time_now_;
    struct timeval time_pre_;
    int time_inited_;

    unsigned char *image_ptr_ = NULL;
};
#endif
#endif