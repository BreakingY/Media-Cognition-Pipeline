#ifndef AACENCODECER_H
#define AACENCODECER_H

#include "DecEncInterface.h"
#include "log_helpers.h"
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
// 只接受packed数据
typedef struct AACPCMNodeSt {
    unsigned char *PCMData;
    int data_len;
    AACPCMNodeSt(unsigned char *data, int len)
    {
        PCMData = (unsigned char *)malloc(len);
        memcpy(PCMData, data, len);
        data_len = len;
    }
    virtual ~AACPCMNodeSt()
    {
        free(PCMData);
    }
} AACPCMNode;

class AACEncoder
{
public:
    AACEncoder();
    ~AACEncoder();
    int AddPCMFrame(unsigned char *data, int data_len);
    int Init(enum AVSampleFormat fmt, int channels, int ratio, int nb_samples);
    void SetCallback(EncDataCallListner *call_func);

public:
    static void *AACScaleThread(void *arg);
    static void *AACEncThread(void *arg);

private:
    EncDataCallListner *callback_;
    enum AVSampleFormat src_sample_fmt_;
    enum AVSampleFormat dst_sample_fmt_ = AV_SAMPLE_FMT_S16;
    int src_nb_channels_;
    int dst_nb_channels_ = 2;
    int src_ratio_;
    int dst_ratio_ = 44100;
    // 单个通道的一帧采样点个数
    int src_nb_samples_;
    int dst_nb_samples_;
    SwrContext *encode_swr_ctx_;
    AVCodecContext *c_ctx_;
    AVCodec *codec_;
    AVPacket pkt_enc_;

    pthread_mutex_t pcm_mutex_;
    pthread_cond_t pcm_cond_;
    std::list<AACPCMNode *> pcm_frames_;
    pthread_mutex_t frame_mutex_;
    pthread_cond_t frame_cond_;
    std::list<AVFrame *> dec_frames_;
    pthread_t scale_id_;
    pthread_t encode_id_;

    bool abort_;
    struct timeval time_now_;
    struct timeval time_pre_;
    int time_inited_;
    int now_frames_;
    int pre_frames_;
};
#endif
