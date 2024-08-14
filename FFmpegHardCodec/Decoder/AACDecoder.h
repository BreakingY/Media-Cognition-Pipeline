#ifndef AAC_DEC_H
#define AAC_DEC_H

#include "DecEncInterface.h"
#include "log_helpers.h"
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
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
#include <libavutil/pixfmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

typedef struct AACDataNodeSt {
    unsigned char *es_data;
    int es_data_len;
    AACDataNodeSt()
    {
        es_data = NULL;
        es_data_len = 0;
    }
    virtual ~AACDataNodeSt()
    {
        if (es_data) {
            free(es_data);
            es_data = NULL;
        }
    }
} AACDataNode;

class AACDecoder
{

public:
    AACDecoder();
    virtual ~AACDecoder();
    void SetCallback(DecDataCallListner *call_func);
    void SetResampleArg(enum AVSampleFormat fmt, int channels, int ratio);
    void InputAACData(unsigned char *data, int data_len);

private:
    static void *AACDecodeThread(void *arg);
    void DecodeAudio(AACDataNode *data);
    static void *AACScaleThread(void *arg);
    void ScaleAudio(AVFrame *frame);

public:
    AVCodecContext *audio_codec_ctx_ = NULL;
    AVCodec *audio_codec_ = NULL;
    AVPacket packet_;
    AVFrame *frame_ = NULL;

    SwrContext *swr_ctx_ = NULL;
    // 解码重采样
    enum AVSampleFormat src_sample_fmt_;
    enum AVSampleFormat dst_sample_fmt_ = AV_SAMPLE_FMT_S16;
    int src_nb_channels_;
    int dst_nb_channels_ = 2;
    int src_ratio_;
    int dst_ratio_ = 44100;
    // 单个通道的一帧采样点个数
    int src_nb_samples_;
    int dst_nb_samples_;

    DecDataCallListner *callback_ = NULL;

    std::list<AACDataNode *> es_packets_;
    std::list<AVFrame *> yuv_frames_;
    pthread_mutex_t packet_mutex_;
    pthread_cond_t packet_cond_;
    pthread_cond_t frame_cond_;
    pthread_mutex_t frame_mutex_;

    pthread_t dec_thread_id_;
    pthread_t sws_thread_id_;
    bool aborted_;
    int now_frames_;
    int pre_frames_;
    struct timeval time_now_;
    struct timeval time_pre_;
    int time_inited_;
};

#endif