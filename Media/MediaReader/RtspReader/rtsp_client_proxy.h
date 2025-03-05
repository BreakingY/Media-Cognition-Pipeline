#ifndef RTSP_CLIENT_PROXY
#define RTSP_CLIENT_PROXY

#include <thread>
#include <mutex>
#include <chrono>
#include <list>
#include <condition_variable>
#include "rtsp_client.h"
#include "MediaInterface.h"
#include "TypeDef.h"
#include "AAC.h"
#define PROBEFRAME 50 // 探测帧数，用于计算视频fps
class RtspClientProxy:public RtspMediaInterface{
public:
    RtspClientProxy(char *rtsp_url);
    ~RtspClientProxy();
    int ProbeVideoFps();
    void GetVideoCon(int &width, int &height, int &fps);
    void GetAudioCon(int &sample_rate_index, int &channels, int &profile);
    enum VideoType GetVideoType();
    enum AudioType GetAudioType();
    void SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb){data_listner_ = lisnter; colse_cb_ = cb; return;}
    
private:
    void RtspVideoData(int64_t pts, const uint8_t* data, size_t size);
    void RtspAudioData(int64_t pts,  const uint8_t* data, size_t size);
    static void *ReconnectThread(void *arg);
private:
    std::string rtsp_url_;
    enum TRANSPORT transport_ = TRANSPORT::RTP_OVER_TCP;
    RtspClient *client_ = NULL;
    std::thread tid_;
    bool run_flag_ = true;
    int width_ = -1;
    int height_ = -1;
    int fps_ = -1;
    bool video_ready_ = false;

    MediaDataListner *data_listner_ = NULL;
    CloseCallbackFunc colse_cb_ = NULL;

    int64_t last_timestamp_ = -1;
    int64_t interval_sum_ = 0;
    int probe_cnt_ = 0;
};

#endif