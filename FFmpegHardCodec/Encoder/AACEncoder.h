#ifndef AACENCODECER_H
#define AACENCODECER_H

#include "DecEncInterface.h"
#include "log_helpers.h"
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <list>

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
typedef struct AACPCMNodeSt {
    unsigned char *pcm_data;
    int data_len;
    AACPCMNodeSt(unsigned char *data, int len)
    {
        pcm_data = (unsigned char *)malloc(len);
        memcpy(pcm_data, data, len);
        data_len = len;
    }
    virtual ~AACPCMNodeSt()
    {
        free(pcm_data);
    }
} AACPCMNode;
// 单通道样本输入数量必须是1024(LC-AAC)或者2048(HE-AAC)
class AACEncoder
{
public:
    AACEncoder();
    ~AACEncoder();
    int AddPCMFrame(unsigned char *data, int data_len);
    int Init(enum AVSampleFormat fmt, int channels, int ratio, int nb_samples);
    void SetCallback(EncDataCallListner *call_func);
    void GetAudioCon(int &channels, int &sample_rate, int &profile);

private:
    static void *AACScaleThread(void *arg);
    static void *AACEncThread(void *arg);

private:
    EncDataCallListner *callback_ = NULL;
    enum AVSampleFormat src_sample_fmt_;
    enum AVSampleFormat dst_sample_fmt_ = AV_SAMPLE_FMT_S16;
    int src_nb_channels_;
    int dst_nb_channels_ = 2;
    int src_ratio_;
    int dst_ratio_ = 44100;
    // 单个通道的一帧采样点个数
    int src_nb_samples_;
    int dst_nb_samples_;
    SwrContext *encode_swr_ctx_ = NULL;
    AVCodecContext *c_ctx_ = NULL;
    AVCodec *codec_ = NULL;
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
