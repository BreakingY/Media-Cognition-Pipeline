#ifndef H264_HARD_ENC_H
#define H264_HARD_ENC_H

#include "DecEncInterface.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>
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

class HardVideoEncoder
{
public:
    HardVideoEncoder();
    ~HardVideoEncoder();
    int AddVideoFrame(cv::Mat bgr_frame);
    int HardEncInit(int width, int height, int fps);
    int SoftEncInit(int width, int height, int fps);
    int Init(cv::Mat init_frame, int fps);
    void SetDataCallback(EncDataCallListner *call_func);

private:
    static void *VideoScaleThread(void *arg);
    static void *VideoEncThread(void *arg);

private:
    EncDataCallListner *callback_;
    AVCodecContext *h264_codec_ctx_;
    AVCodec *h264_codec_;
    SwsContext *sws_context_;
    enum AVPixelFormat sw_pix_format_ = AV_PIX_FMT_YUV420P;
    // hard enc
    enum AVHWDeviceType type_ = AV_HWDEVICE_TYPE_NONE;
    bool is_hard_enc_ = false;
    enum AVCodecID decodec_id_;

    std::list<cv::Mat> bgr_frames_;
    std::list<AVFrame *> yuv_frames_;
    pthread_mutex_t bgr_mutex_;
    pthread_cond_t bgr_cond_;
    pthread_mutex_t yuv_mutex_;
    pthread_cond_t yuv_cond_;
    pthread_t scale_id_;
    pthread_t encode_id_;

    bool abort_;
    uint64_t nframe_counter_;
    struct timeval time_now_;
    struct timeval time_pre_;
    uint64_t time_ts_accum_;

    struct timeval time_now_1_;
    struct timeval time_pre_1_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
