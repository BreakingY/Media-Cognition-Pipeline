#ifndef VIDEOWARPPER_H
#define VIDEOWARPPER_H
#include "AAC.h"
#include "AACDecoder.h"
#include "AACEncoder.h"
#include "DecEncInterface.h"
#include "H264HardEncoder.h"
#include "HardDecoder.h"
#include "MediaInterface.h"
#include "MediaMuxer.h"
#include "MediaReader.h"
#include "log_helpers.h"
#include "rtsp_client_proxy.h"
#include <opencv2/opencv.hpp>
class MiedaWrapper : public MediaDataListner, public DecDataCallListner, public EncDataCallListner
{
public:
    MiedaWrapper() = delete;
    MiedaWrapper(char *input, char *ouput);
    virtual ~MiedaWrapper();
    // 音视频解封装接口
    void OnVideoData(VideoData data);
    void OnAudioData(AudioData data);
    void MediaOverhandle();

    // 解码后数据接口
    void OnRGBData(cv::Mat frame);
    void OnPCMData(unsigned char **data, int data_len);

    // 编码后的数据接口
    void OnVideoEncData(unsigned char *data, int data_len, int64_t pts);
    void OnAudioEncData(unsigned char *data, int data_len);

    bool OverHandle() { return over_flag_; }
    int WriteVideo2File(uint8_t *data, int len);
    int WriteAudio2File(uint8_t *data, int len);

public:
    bool over_flag_ = false;
    // video
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t nframe_counter_ = 0;
    uint64_t time_ts_accum_ = 0;
    // audio
    std::chrono::steady_clock::time_point time_now_1_;
    std::chrono::steady_clock::time_point time_pre_1_;
    uint64_t nframe_counter_1_ = 0;
    uint64_t time_ts_accum_1_ = 0;

    MediaReader *reader_ = NULL;
    RtspClientProxy *rtsp_client_proxy_ = NULL;
    bool rtsp_flag_ = false;
    int width_;
    int height_;
    int fps_ = 25;
    enum VideoType video_type_;
    enum AudioType audio_type_;
    unsigned char *buffer_pcm_ = NULL;
    int buffer_pcm_len_ = 0;

    HardVideoDecoder *hard_decoder_ = NULL;
    HardVideoEncoder *hard_encoder_ = NULL;
    AACDecoder *aac_decoder_ = NULL;
    AACEncoder *aac_encoder_ = NULL;

    uint8_t *vps_ = NULL;
    uint8_t *sps_ = NULL;
    uint8_t *pps_ = NULL;
    int vps_buffer_len_ = 0;
    int sps_buffer_len_ = 0;
    int pps_buffer_len_ = 0;
    int vps_len_ = 0;
    int sps_len_ = 0;
    int pps_len_ = 0;
    bool extra_ready_ = false;

    Muxer *mp4_muxer_ = NULL;
    int video_stream_ = -1;
    int audio_stream_ = -1;

    // GPU
    int32_t device_id_ = 0;

};
#endif