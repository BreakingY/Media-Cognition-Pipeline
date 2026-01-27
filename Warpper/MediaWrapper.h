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
#include "RtmpPushClient.h"
#include "RtmpPullClient.h"
#include <opencv2/opencv.hpp>
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#include "NodeFlow.h"
class MediaWrapper : public MediaDataListner, public DecDataCallListner, public EncDataCallListner, public InferDataListner
#else
class MediaWrapper : public MediaDataListner, public DecDataCallListner, public EncDataCallListner
#endif
{
public:
    MediaWrapper() = delete;
    MediaWrapper(const char *input, const char *output, const char *eng_path/*for Cognition*/ = nullptr, int device_id/*for nvidia ascend*/ = 0);
    virtual ~MediaWrapper();
    // 音视频解封装接口
    void OnVideoData(VideoData data);
    void OnAudioData(AudioData data);
    void MediaOverhandle();

    // 解码后数据接口
    void OnRGBData(cv::Mat frame);
    void OnPCMData(unsigned char **data, int data_len);

    // 编码后的数据接口
    // 音视频接口中的pts是独立的，没有同步
    void OnVideoEncData(unsigned char *data, int data_len, int64_t pts);
    void OnAudioEncData(unsigned char *data, int data_len, int64_t pts);

    bool OverHandle() { return over_flag_; }
    int WriteVideo2File(uint8_t *data, int len);
    int WriteAudio2File(uint8_t *data, int len);

    // for nvidia
    void UseNVEnc() {use_nv_enc_flag_ = true; return;}

#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
    void OnInferData(cv::Mat& img, DetectionInfo& info);
#endif

public:
    bool over_flag_ = false;

    MediaReader *reader_ = nullptr;
    RtspClientProxy *rtsp_client_proxy_ = nullptr;
    RtmpPullClient *rtmp_pull_client_ = nullptr;
    int width_;
    int height_;
    int fps_ = 25;
    enum VideoType video_type_;
    enum AudioType audio_type_;
    unsigned char *buffer_pcm_ = nullptr;
    int buffer_pcm_len_ = 0;

    HardVideoDecoder *hard_decoder_ = nullptr; // h264/h265
    HardVideoEncoder *hard_encoder_ = nullptr; // h264
    AACDecoder *aac_decoder_ = nullptr; // aac
    AACEncoder *aac_encoder_ = nullptr; // aac
    
    Muxer *mp4_muxer_ = nullptr; // MP4 requires knowing whether audio/video tracks exist in advance
    bool SetMP4MediaInfo();
    std::mutex mp4_mtx_;
    std::atomic<bool> set_mp4_info_over_ = {false};
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t nframe_counter_ = 0;

    RtmpPushClient *rtmp_push_client_ = nullptr; // RTMP/FLV does not require knowing whether audio/video streams exist in advance

    // NPU GPU
    int32_t device_id_ = 0;
    bool use_nv_enc_flag_ = false;

#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
    std::string eng_path_;
    void *context_ = nullptr;
#endif
};
#endif