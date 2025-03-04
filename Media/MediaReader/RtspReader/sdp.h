#ifndef SDP_H_
#define SDP_H_
#include <iostream>
#include <string>
#include <vector>
#include "rtsp_common.h"
#include "AAC.h"
/*
v=0
o=- 91720590340 1 IN IP4 192.168.10.17
c=IN IP4 192.168.10.17
t=0 0
a=control:*
m=video 0 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
a=control:track0
m=audio 0 RTP/AVP 97
a=rtpmap:97 MPEG4-GENERIC/44100/2
a=fmtp:97 streamtype=5;profile-level-id=1;mode=AAC-hbr;config=1390;sizelength=13;indexlength=3;indexdeltalength=3
a=control:track1
*/
struct MediaInfo{
    std::string media_name; // video audio
    MediaEnum media_type;
    std::string contorl;    // track0 track1 or url/track0 url/track1

    int payload;            // rtp payload
    int sample_rate;
    int sample_rate_index;
    int channels;           // only audio
    int profile;            // only audio from config=1390
};
struct SdpInfo{
    std::string contorl;            // * or url
    struct MediaInfo media_info[2]; // 0-video 1-audio  
    int media_count;
};
class SDPParse{
public:
    SDPParse(std::string sdp, std::string base_url);
    ~SDPParse();
    int Parse();
    std::string GetVideoUrl();
    std::string GetAudioUrl();
    int GetVideoPayload(){return sdp_info_.media_info[0].payload;}
    int GetAudioPayload(){return sdp_info_.media_info[1].payload;}
    enum MediaEnum GetVideoType() {return sdp_info_.media_info[0].media_type;}
    enum MediaEnum GetAudioType() {return sdp_info_.media_info[1].media_type;}
    void GetAudioInfo(int &sample_rate_index, int &channels, int &profile) {sample_rate_index = sdp_info_.media_info[1].sample_rate_index; channels = sdp_info_.media_info[1].channels; profile = sdp_info_.media_info[1].profile; return;}
private:
    int ParseSession();
    int ParseVideo();
    int ParseAudio();
private:
    std::string sdp_;
    std::string base_url_;
    std::string sdp_session_;
    std::vector<std::string> media_descriptions_;
    std::string sdp_video_ = "";
    std::string sdp_audio_ = "";
    struct SdpInfo sdp_info_;
};
#endif