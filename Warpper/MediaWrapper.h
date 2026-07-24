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
#include "TsMuxerClient.h"
#include "TsDemuxerClient.h"
#include <opencv2/opencv.hpp>
#if defined(MCP_PYBIND)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using FrameCallbackFuncVideo = std::function<void(py::array_t<uint8_t>)>;
using FrameCallbackFuncAudio = std::function<void(py::array_t<uint8_t>, int/*单通道当本个数*/, int/*每个样本占用的字节数*/, int/*通道数量*/)>; // packed格式的pcm音频
#endif
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
#include "NodeFlow.h"
class MediaWrapper : public MediaDataListner, public DecDataCallListner, public EncDataCallListner, public InferDataListner
#else
class MediaWrapper : public MediaDataListner, public DecDataCallListner, public EncDataCallListner
#endif
{
public:
    MediaWrapper() = delete;
    MediaWrapper(const char *input, const char *output, const char *eng_path/*for Cognition*/ = nullptr, int device_id/*for nvidia ascend hygon*/ = 0);
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

#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
    void OnInferData(cv::Mat& img, DetectionInfo& info);
#endif
#if defined(MCP_PYBIND)
    /**
     * 流媒体处理库：为Python提供高性能的视频处理功能，包括视频采集、编解码加速、帧数据交互以及实时流媒体处理。
     * 典型应用场景：
     *    C++负责视频采集(文件，实时流)、编解码加速、输出(文件，实时流)
     *    Python负责目标检测、人脸识别、行为分析等模型算法。
     * 开启MCP_PYBIND时此项目仅作为媒体库使用，不能和
     */
    // cb：python回调函数, 接收解码图像DETECTION_*同时使用
    void InitImgPycallback(FrameCallbackFuncVideo video_cb, FrameCallbackFuncAudio audio_cb) {video_cb_ = video_cb; audio_cb_ = audio_cb;}
    // frame：python处理后的图像，用于输出到文件或推流
    void PyAddVideoFrame(cv::Mat frame);
    // packed格式的pcm音频
    void PyAddAudioFrame(uint8_t* data, int data_len, int spb, int channels);
#endif
public:
    bool over_flag_ = false;

    MediaReader *reader_ = nullptr;
    RtspClientProxy *rtsp_client_proxy_ = nullptr;
    RtmpPullClient *rtmp_pull_client_ = nullptr;
    TsDemuxerClient *ts_demuxer_client_ = nullptr;
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
    
    bool SetMediaInfo();
    std::mutex media_info_mtx_;
    std::atomic<bool> set_media_info_over_ = {false};
    std::chrono::steady_clock::time_point time_now_;
    std::chrono::steady_clock::time_point time_pre_;
    uint64_t nframe_counter_ = 0;

    Muxer *mp4_muxer_ = nullptr; // MP4 requires knowing whether audio/video tracks exist in advance
    RtmpPushClient *rtmp_push_client_ = nullptr; // RTMP/FLV does not require knowing whether audio/video streams exist in advance
    TsMuxerClient *ts_muxer_client_ = nullptr; // Live streaming: No need to know in advance whether there is an audio/video stream. File: Need to know audio and video information in advance

    // NPU GPU DCU
    int32_t device_id_ = 0;
    bool use_nv_enc_flag_ = false;

#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
    std::string eng_path_;
    void *context_ = nullptr;
#endif
#if defined(MCP_PYBIND)
    FrameCallbackFuncVideo video_cb_ = nullptr;
    FrameCallbackFuncAudio audio_cb_ = nullptr;
#endif
};
#endif