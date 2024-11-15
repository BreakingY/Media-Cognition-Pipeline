#ifndef RTSP_CLIENT
#define RTSP_CLIENT
#include <iostream>
#include <string>
#include "rtsp_common.h"
#include "sdp.h"
#include "rtp_demuxer.h"
#define USER_AGENT "simple-rtsp-client"
#define READ_SOCK_DATA_LEN 1500
enum TRANSPORT{
    RTP_OVER_TCP = 0,
    RTP_OVER_UDP,
};
enum RTSPCMDSTAT{
    RTSP_NONE = 0,
    RTSP_OPTIONS,
    RTSP_DESCRIBE,
    RTSP_STEUP,
    RTSP_STEUP_VIDEO,
    RTSP_STEUP_ADUIO,
    RTSP_PLAY,
    RTSP_PLAYING,
    RTSP_COMPLETE,
};
class RtspMediaInterface {
public:
  virtual void RtspVideoData(int64_t pts, const uint8_t* data, size_t size) = 0;
  virtual void RtspAudioData(int64_t pts,  const uint8_t* data, size_t size) = 0;
};
enum ParseState
{
    EMPTY_STATE,
    RTP_TCP_HEADER_STATE,
    RTP_TCP_CONTENT_STATE,
    RTSP_MESSAGE_STATE,
};

class RtspClient : public RTPDemuxerInterface {
public:
    RtspClient(enum TRANSPORT transport = TRANSPORT::RTP_OVER_UDP);
    ~RtspClient();
    int Connect(char *url);
    enum MediaEnum GetVideoType() {return sdp_->GetVideoType();}
    enum MediaEnum GetAudioType() {return sdp_->GetAudioType();}
    void SetCallBack(RtspMediaInterface *call_back){call_back_ = call_back; return;}
    void GetAudioInfo(int &sample_rate_index, int &channels, int &profile) {sdp_->GetAudioInfo(sample_rate_index, channels, profile); return;}
    bool GetOpenStat(){return connected_;}
private:
    void OnVideoData(int64_t pts, const uint8_t* data, size_t size);
    void OnAudioData(int64_t pts,  const uint8_t* data, size_t size);

    int SendOPTIONS(const char *url);
    int DecodeOPTIONS(const char *buffer, int len);

    std::string GenerateAuthHeader(std::string url, std::string response);

    int SendDESCRIBE(const char *url, const char *authorization = NULL);
    int DecodeDESCRIBE(const char *url, const char *buffer, int len);

    int SendSTEUP(const char *url);
    int DecodeSTEUP(const char *url, const char *buffer, int len);

    int SendPLAY(const char *url);
    int DecodePLAY(const char *url, const char *buffer, int len);
    
    static void *RecvPacketThd(void *arg);
    int ReadPacketUdp();
    int ReadPacketTcp();
    
private:
    std::string rtsp_url_ = "";
    int rtsp_sd_ = -1;
    int cseq = 1;
    struct RTSPUrlInfo url_info_;
    std::string realm_ = "";
    std::string nonce_ = "";
    std::string content_base_ = "";
    SDPParse *sdp_ = NULL;
    std::string video_url_ = "";
    std::string audio_url_ = "";
    bool video_setup_ = false;
    bool audio_setup_ = false;
    enum TRANSPORT rtp_transport_;
    std::string session_ = "" ;
    int timeout_ = 60; // 秒
    bool connected_ = false;
    enum RTSPCMDSTAT rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_NONE;
    char buffer_cmd_[4096] = {0};
    int buffer_cmd_used_ = 0;
    int buffer_cmd_size_ = 0;
    // udp 
    int rtp_port_video_ = -1;
    int rtcp_port_video_ = -1;
    int rtp_sd_video_ = -1;
    int rtcp_sd_video_ = -1;
    int rtp_port_video_server_ = -1;
    int rtcp_port_video_server_ = -1;

    int rtp_port_audio_ = -1;
    int rtcp_port_audio_ = -1;
    int rtp_sd_audio_ = -1;
    int rtcp_sd_audio_ = -1;
    int rtp_port_audio_server_ = -1;
    int rtcp_port_audio_server_ = -1;
    // tcp 
    int sig0_video_ = 0;
    int sig0_audio_ = 2;

    pthread_t tid_;
    bool run_flag_ = true;
    int recv_rtp_packet_timeout_ = 2; // 秒

    RTPDemuxer *rtp_video_demuxer_ = NULL;
    RTPDemuxer *rtp_audio_demuxer_ = NULL;

    RtspMediaInterface *call_back_ = NULL;
    bool video_frame_ready_ = false;

    struct rtp_tcp_header header_;
    // 缓存rtp over tcp头部
    uint8_t buffer_header_[4];
    int pos_buffer_header_ = 0;
    // 缓存rtp数据包
    uint8_t buffer_[4 * 1024 * 1024];
    int pos_buffer_ = 0;
    enum ParseState stat_ = EMPTY_STATE;
};

#endif