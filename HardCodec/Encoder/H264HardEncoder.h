#ifndef H264_HARD_ENC_H
#define H264_HARD_ENC_H

#include "DecEncInterface.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <thread>
#include <mutex>
#include <chrono>
#include <list>
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
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#define DROP_FRAME
#ifdef USE_FFMPEG_NVIDIA
class HardVideoEncoder
{
public:
    HardVideoEncoder();
    ~HardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame);
    int Init(cv::Mat init_frame, int fps);
    void SetDataCallback(EncDataCallListner *call_func);

private:
    int HardEncInit(int width, int height, int fps);
    int SoftEncInit(int width, int height, int fps);
    static void *VideoScaleThread(void *arg);
    static void *VideoEncThread(void *arg);

private:
    EncDataCallListner *callback_ = NULL;
    AVCodecContext *h264_codec_ctx_ = NULL;
    AVCodec *h264_codec_ = NULL;
    SwsContext *sws_context_ = NULL;
    enum AVPixelFormat sw_pix_format_ = AV_PIX_FMT_YUV420P;
    // hard enc
    enum AVHWDeviceType type_ = AV_HWDEVICE_TYPE_NONE;
    bool is_hard_enc_ = false;
    enum AVCodecID decodec_id_;

    std::list<cv::Mat> bgr_frames_;
    std::list<AVFrame *> yuv_frames_;
    std::mutex bgr_mutex_;
    std::condition_variable bgr_cond_;
    std::mutex yuv_mutex_;
    std::condition_variable yuv_cond_;
    std::thread scale_id_;
    std::thread encode_id_;

    bool abort_;
    uint64_t nframe_counter_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
#ifdef USE_FFMPEG_SOFT
class HardVideoEncoder
{
public:
    HardVideoEncoder();
    ~HardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame);
    int Init(cv::Mat init_frame, int fps);
    void SetDataCallback(EncDataCallListner *call_func);

private:
    int SoftEncInit(int width, int height, int fps);
    static void *VideoScaleThread(void *arg);
    static void *VideoEncThread(void *arg);

private:
    EncDataCallListner *callback_ = NULL;
    AVCodecContext *h264_codec_ctx_ = NULL;
    AVCodec *h264_codec_ = NULL;
    SwsContext *sws_context_ = NULL;
    enum AVPixelFormat sw_pix_format_ = AV_PIX_FMT_YUV420P;
    enum AVCodecID decodec_id_;

    std::list<cv::Mat> bgr_frames_;
    std::list<AVFrame *> yuv_frames_;
    std::mutex bgr_mutex_;
    std::condition_variable bgr_cond_;
    std::mutex yuv_mutex_;
    std::condition_variable yuv_cond_;
    std::thread scale_id_;
    std::thread encode_id_;

    bool abort_;
    uint64_t nframe_counter_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
#ifdef USE_DVPP_MPI
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#include "sample_comm.h"
#include "sample_api.h"
#include "sample_encoder_manage.h"
void vencStreamOut(uint32_t channelId, void* buffer, void *arg);
class HardVideoEncoder
{
public:
    HardVideoEncoder();
    ~HardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame);
    void SetDevice(int device_id);
    int Init(cv::Mat init_frame, int fps);
    void SetDataCallback(EncDataCallListner *call_func);

private:
    friend void vencStreamOut(uint32_t channelId, void* buffer, void *arg);
    static void *VideoScaleThread(void *arg);
    static void *VideoEncThread(void *arg);
    void *GetColorAddr();
    void PutColorAddr(void *addr);
    void InitEncParams(VencParam* encParam);
private:
    int32_t device_id_ = 0;
    EncDataCallListner *callback_ = NULL;

    std::list<cv::Mat> bgr_frames_;
    std::list<void *> yuv_frames_;
    std::mutex bgr_mutex_;
    std::condition_variable bgr_cond_;
    std::mutex yuv_mutex_;
    std::condition_variable yuv_cond_;
    bool abort_;
    std::thread scale_id_;
    std::thread encode_id_;
    // color convert
    // mem pool
    uint32_t pool_num_ = 10;
    uint32_t out_buffer_size_ = 0;
    std::list<void*> out_buffer_pool_;
    std::mutex out_buffer_pool_mutex_;
    std::condition_variable out_buffer_pool_cond_;
    int width_;
    int height_;
    int fps_;
    int width_stride_;
    int height_stride_;
    // color input
    void * in_img_buffer_ = NULL;
    uint32_t in_img_buffer_size_;
    hi_pixel_format in_format_ = HI_PIXEL_FORMAT_BGR_888;
    // color output
    hi_pixel_format out_format_ = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    // color context
    hi_vpc_chn channel_id_color_;
    hi_vpc_pic_info input_pic_;
    hi_vpc_pic_info output_pic_;

    // dvpp enc
    int32_t enc_channel_;
    int32_t codec_type_;
    int32_t bit_rate_ = 0;
    IHWCODEC_HANDLE enc_handle_;
    VencParam enc_param_;

    unsigned char *image_ptr_ = NULL;
    uint64_t image_ptr_size_ = 1024 * 1024 * 2;

    uint64_t nframe_counter_;
    uint64_t nframe_counter_recv_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
#ifdef USE_NVIDIA_X86
#include <cuda_runtime.h>
#include "NvEncoderCuda.h"
#include "NvEncoderCLIOptions.h"
#include "NvCodecUtils.h"
class HardVideoEncoder
{
public:
    virtual ~HardVideoEncoder() {}
    virtual int AddVideoFrame(cv::Mat bgr_frame) = 0;
    virtual void SetDevice(int device_id) = 0;
    virtual int Init(cv::Mat init_frame, int fps) = 0;
    virtual void SetDataCallback(EncDataCallListner *call_func) = 0;
};
class NVSoftVideoEncoder: public HardVideoEncoder
{
public:
    NVSoftVideoEncoder();
    virtual ~NVSoftVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame) override;
    virtual void SetDevice(int device_id) override;
    int Init(cv::Mat init_frame, int fps) override;
    void SetDataCallback(EncDataCallListner *call_func) override;

private:
    int SoftEncInit(int width, int height, int fps);
    static void *VideoScaleThread(void *arg);
    static void *VideoEncThread(void *arg);

private:
    EncDataCallListner *callback_ = NULL;
    AVCodecContext *h264_codec_ctx_ = NULL;
    AVCodec *h264_codec_ = NULL;
    SwsContext *sws_context_ = NULL;
    enum AVPixelFormat sw_pix_format_ = AV_PIX_FMT_YUV420P;
    enum AVCodecID decodec_id_;

    std::list<cv::Mat> bgr_frames_;
    std::list<AVFrame *> yuv_frames_;
    std::mutex bgr_mutex_;
    std::condition_variable bgr_cond_;
    std::mutex yuv_mutex_;
    std::condition_variable yuv_cond_;
    std::thread scale_id_;
    std::thread encode_id_;

    bool abort_;
    uint64_t nframe_counter_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
class NVHardVideoEncoder: public HardVideoEncoder
{
public:
    NVHardVideoEncoder();
    virtual ~NVHardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame) override;
    void SetDevice(int device_id) override;
    int Init(cv::Mat init_frame, int fps) override;
    void SetDataCallback(EncDataCallListner *call_func) override;

private:
    static void *VideoEncThread(void *arg);
private:
    int32_t device_id_ = 0;
    EncDataCallListner *callback_ = NULL;

    std::list<cv::Mat> bgr_frames_;
    std::mutex bgr_mutex_;
    std::condition_variable bgr_cond_;

    NvEncoderInitParam init_param_;
    NV_ENC_BUFFER_FORMAT eformat_;
    void *ptr_image_bgr_device_ = NULL;
    void *ptr_image_bgra_device_ = NULL;
    NvEncoderCuda *enc_ = NULL;
    
    bool abort_;
    std::thread encode_id_;
    
    int width_;
    int height_;
    int fps_;
    
    uint64_t nframe_counter_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
#ifdef USE_NVIDIA_ARM
#include <cuda_runtime.h>
#include "JetsonEnc.h"
class HardVideoEncoder : public JetsonEncListner
{
public:
    HardVideoEncoder();
    ~HardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame);
    void SetDevice(int device_id){device_id_ = device_id;}
    int Init(cv::Mat init_frame, int fps);
    void SetDataCallback(EncDataCallListner *call_func);
private:
    void OnJetsonEncData(unsigned char *data, int data_len);

private:
    EncDataCallListner *callback_ = nullptr;
    JetsonEnc *jetson_enc_obj_ = nullptr;
    int width_;
    int height_;
    int32_t device_id_ = 0;
    unsigned char *d_bgr_ = nullptr;
    unsigned char *d_rgb_ = nullptr;
    unsigned char *d_yuv_ = nullptr;
    uint64_t nframe_counter_;
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t time_ts_accum_;

    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
#endif
