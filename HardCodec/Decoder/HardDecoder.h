#ifndef _HARD_DEC_H
#define _HARD_DEC_H

#include "DecEncInterface.h"
#include "log_helpers.h"
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

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
    HardVideoDecoder(CODEC_TYPE codec_type);
    virtual ~HardVideoDecoder();
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    int HardDecInit();
    int SoftDecInit();
    int hwDecoderInit(AVCodecContext *ctx, const enum AVHWDeviceType type);
    static void *DecodeThread(void *arg);
    void DecodeVideo(HardDataNode *data);
    static void *ScaleThread(void *arg);
    void ScaleVideo(AVFrame *frame);

private:
    bool is_hard_ = false;
    CODEC_TYPE codec_type_;
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

    std::mutex packet_mutex_;
    std::condition_variable packet_cond_;
    std::condition_variable frame_cond_;
    std::mutex frame_mutex_;
    std::list<HardDataNode *> es_packets_;
    std::list<AVFrame *> yuv_frames_;
    std::thread dec_thread_id_;
    std::thread sws_thread_id_;
    bool abort_;

    int now_frames_;
    int pre_frames_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    int time_inited_;

    unsigned char *image_ptr_ = NULL;
};
#endif
#ifdef USE_FFMPEG_SOFT
class HardVideoDecoder
{

public:
    HardVideoDecoder(CODEC_TYPE codec_type);
    virtual ~HardVideoDecoder();
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    int SoftDecInit();
    static void *DecodeThread(void *arg);
    void DecodeVideo(HardDataNode *data);
    static void *ScaleThread(void *arg);
    void ScaleVideo(AVFrame *frame);

private:
    CODEC_TYPE codec_type_;
    enum AVCodecID decodec_id_;
    DecDataCallListner *callback_ = NULL;
    AVCodecContext *codec_ctx_ = NULL;
    AVCodec *codec_ = NULL;

    AVPacket packet_;
    AVFrame *frame_ = NULL;
    struct SwsContext *img_convert_ctx_ = NULL;
    enum AVPixelFormat out_pix_fmt_ = AV_PIX_FMT_NONE;

    std::mutex packet_mutex_;
    std::condition_variable packet_cond_;
    std::condition_variable frame_cond_;
    std::mutex frame_mutex_;
    std::list<HardDataNode *> es_packets_;
    std::list<AVFrame *> yuv_frames_;
    std::thread dec_thread_id_;
    std::thread sws_thread_id_;
    bool abort_;

    int now_frames_;
    int pre_frames_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    int time_inited_;

    unsigned char *image_ptr_ = NULL;
};
#endif
#ifdef USE_DVPP_MPI
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
class HardVideoDecoder
{

public:
    HardVideoDecoder(CODEC_TYPE codec_type);
    virtual ~HardVideoDecoder();
    void Init(int32_t device_id, int width, int height);
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    void VdecResetChn();
    static void *SendStream(void *arg);
    void DecodeVideo(HardDataNode *data);
    static void *GetPic(void *arg);
    void Stop();
    void *GetOutAddr();
    void PutOutAddr(void *addr);
private:
    CODEC_TYPE codec_type_;
    int32_t device_id_ = 0;
    int width_;
    int height_;
    int32_t channel_id_;
    hi_vdec_chn_attr chn_attr_;
    hi_data_bit_width bit_width_ = HI_DATA_BIT_WIDTH_8;// HI_DATA_BIT_WIDTH_8、HI_DATA_BIT_WIDTH_10， 默认是HI_DATA_BIT_WIDTH_8
    hi_pixel_format out_format_ = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;

    void * in_es_buffer_ = NULL;
    uint32_t in_es_buffer_size_ = 1024 * 1024 * 4;

    uint32_t pool_num_ = 10;
    uint32_t out_buffer_size_ = 0;
    std::list<void*> out_buffer_pool_;
    std::mutex out_buffer_pool_mutex_;
    std::condition_variable out_buffer_pool_cond_;
    int width_stride_;
    int height_stride_;

    // color convert
    hi_vpc_chn channel_id_color_;
    hi_pixel_format out_format_color_ = HI_PIXEL_FORMAT_BGR_888;
    hi_vpc_pic_info input_pic_;
    hi_vpc_pic_info output_pic_;
    unsigned char *image_ptr_ = NULL;


    std::thread send_stream_thread_id_;
    std::thread get_pic_thread_id_;


    DecDataCallListner *callback_ = NULL;
    std::mutex packet_mutex_;
    std::condition_variable packet_cond_;
    std::list<HardDataNode *> es_packets_;
    bool abort_ = false;

    int now_frames_;
    int pre_frames_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    int time_inited_;
    
};
#endif

#ifdef USE_NVIDIA_X86
#include <cuda_runtime.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "ColorSpace.h"
class HardVideoDecoder
{

public:
    HardVideoDecoder(CODEC_TYPE codec_type);
    virtual ~HardVideoDecoder();
    void Init(int32_t device_id, int width, int height);
    void SetFrameFetchCallback(DecDataCallListner *call_func);
    void InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts);

private:
    static void *DecodeThread(void *arg);
    void DecodeVideo(HardDataNode *data);
private:
    CODEC_TYPE codec_type_;
    int32_t device_id_ = 0;
    int width_;
    int height_;
    NvDecoder *dec_ = NULL;
    cudaVideoCodec type_;
    void *device_frame_ = NULL;
    void *device_color_frame_ = NULL;
    void* host_frame_ = NULL;

    std::thread dec_thread_id_;

    DecDataCallListner *callback_ = NULL;
    std::mutex packet_mutex_;
    std::condition_variable packet_cond_;
    std::list<HardDataNode *> es_packets_;
    bool abort_ = false;

    int now_frames_;
    int pre_frames_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    int time_inited_;
    
};
#endif
#endif