#include <sstream>
#include <string.h>
#include <stdlib.h>
#include "sdp.h"
#define SDP_DEBUG
SDPParse::SDPParse(std::string sdp, std::string base_url){
    sdp_ = sdp;
    base_url_ = base_url;
    int pos = sdp_.find("m=");
    if(pos != std::string::npos){
        sdp_session_ = sdp_.substr(0, pos);
    }
    else{
        sdp_session_ = sdp_;
    }
    while ((pos = sdp_.find("m=", pos)) != std::string::npos) {
        size_t next_pos = sdp_.find("m=", pos + 1);
        std::string media_block;
        if (next_pos != std::string::npos) {
            media_block = sdp_.substr(pos, next_pos - pos);
        } else {
            media_block = sdp_.substr(pos);
        }
        media_descriptions_.push_back(media_block);
        if(media_block.find("video") != std::string::npos){
            sdp_video_ = media_block;
        }
        if(media_block.find("audio") != std::string::npos){
            sdp_audio_ = media_block;
        }
        pos = next_pos;
    }
}
SDPParse::~SDPParse(){

}
static void PrintMedia(struct MediaInfo info){
    std::cout << "info.media_name:" << info.media_name << std::endl;
    std::cout << "info.media_type:" << info.media_type << std::endl;
    std::cout << "info.contorl:" << info.contorl << std::endl;
    std::cout << "info.payload:" << info.payload << std::endl;
    std::cout << "info.sample_rate:" << info.sample_rate << std::endl;
    std::cout << "info.channels:" << info.channels << std::endl;
    std::cout << "info.profile:" << info.profile << std::endl;
    std::cout << "info.framerate:" << info.framerate << std::endl;
    std::cout << std::endl;
    return;
}
int SDPParse::Parse(){
    sdp_info_.media_count =0;
    sdp_info_.media_info[0].media_type = MediaEnum::NONE;
    sdp_info_.media_info[1].media_type = MediaEnum::NONE;
    ParseSession();
    ParseVideo();
    ParseAudio();
#ifdef SDP_DEBUG
    if(sdp_info_.media_info[0].media_type !=  MediaEnum::NONE)
        PrintMedia(sdp_info_.media_info[0]);
    if(sdp_info_.media_info[1].media_type !=  MediaEnum::NONE)
        PrintMedia(sdp_info_.media_info[1]);
#endif
    return 0;
}
std::string SDPParse::GetVideoUrl(){
    if(sdp_info_.media_info[0].media_type !=  MediaEnum::NONE){
        if(sdp_info_.contorl == std::string("*")){
            std::string url;
            char ch = base_url_[base_url_.size()-1];
            if(ch != '/'){
                url = base_url_ + std::string("/");
            }
            else{
                url = base_url_;
            }
            return url + sdp_info_.media_info[0].contorl;
        }
        else{
            // session a=control:rtsp://192.168.0.63/media/video1/
            // meida a=control:rtsp://192.168.0.63/media/video1/trackID=1
            if((sdp_info_.media_info[0].contorl.size() >= sdp_info_.contorl.size()) && 
                (memcmp(sdp_info_.contorl.c_str(), sdp_info_.media_info[0].contorl.c_str(), sdp_info_.contorl.size()) == 0)){
                return sdp_info_.media_info[0].contorl;
            }
            // session a=control:rtsp://192.168.0.63/media/video1/
            // meida a=trackID=1
            std::string url;
            char ch = sdp_info_.contorl[sdp_info_.contorl.size()-1];
            if(ch != '/'){
                url = sdp_info_.contorl + std::string("/");
            }
            else{
                url = sdp_info_.contorl;
            }
            return url + sdp_info_.media_info[0].contorl;
        }
    }
    return "";
}
std::string SDPParse::GetAudioUrl(){
    if(sdp_info_.media_info[1].media_type !=  MediaEnum::NONE){
        if(sdp_info_.contorl == std::string("*")){
            std::string url;
            char ch = base_url_[base_url_.size()-1];
            if(ch != '/'){
                url = base_url_ + std::string("/");
            }
            else{
                url = base_url_;
            }
            return url + sdp_info_.media_info[1].contorl;
        }
        else{
            // session a=control:rtsp://192.168.0.63/media/video1/
            // meida a=control:rtsp://192.168.0.63/media/video1/trackID=1
            if((sdp_info_.media_info[1].contorl.size() >= sdp_info_.contorl.size()) && 
                (memcmp(sdp_info_.contorl.c_str(), sdp_info_.media_info[1].contorl.c_str(), sdp_info_.contorl.size()) == 0)){
                return sdp_info_.media_info[1].contorl;
            }
            // session a=control:rtsp://192.168.0.63/media/video1/
            // meida a=trackID=1
            std::string url;
            char ch = sdp_info_.contorl[sdp_info_.contorl.size()-1];
            if(ch != '/'){
                url = sdp_info_.contorl + std::string("/");
            }
            else{
                url = sdp_info_.contorl;
            }
            return url + sdp_info_.media_info[1].contorl;
        }
    }
    return "";
}
/*
v=0
o=- 91720590340 1 IN IP4 192.168.10.17
c=IN IP4 192.168.10.17
t=0 0
a=control:*
*/
int SDPParse::ParseSession(){
    int start = sdp_session_.find("a=control:");
    int end = sdp_session_.find("\r\n", start);
    sdp_info_.contorl = sdp_session_.substr(start + strlen("a=control:"), end - start - strlen("a=control:"));
    return 0;
}
static std::vector<std::string> SplitString(const std::string& str){
    std::vector<std::string> result;
    std::istringstream iss(str);
    std::string word;
    while (iss >> word) {
        result.push_back(word);
    }
    return result;
}
/*
sdp_video_:m=video 0 RTP/AVP 96
a=rtpmap:96 H264/90000
a=fmtp:96 packetization-mode=1
a=control:track0
*/
int SDPParse::ParseVideo(){
    if(sdp_video_.empty()){
        return 0;
    }
    sdp_info_.media_count++;
    sdp_info_.media_info[0].media_name = "video";
    int start = sdp_video_.find("m=video ");
    int end = sdp_video_.find("\r\n");
    std::string m_line = sdp_video_.substr(start + strlen("m=video "), end - start - strlen("m=video "));
    std::vector<std::string> res = SplitString(m_line);
    sdp_info_.media_info[0].payload = atoi(res[2].c_str()); // Select the first payload

    start = sdp_video_.find("a=control:");
    end = sdp_video_.find("\r\n", start);
    sdp_info_.media_info[0].contorl = sdp_video_.substr(start + strlen("a=control:"), end - start - strlen("a=control:"));

    std::string rtpmap = std::string("a=rtpmap:") + std::to_string(sdp_info_.media_info[0].payload);
    std::string fmtp = std::string("a=fmtp:") + std::to_string(sdp_info_.media_info[0].payload);
    // rtpmap
    start = sdp_video_.find(rtpmap.c_str());
    end = sdp_video_.find("\r\n", start);
    std::string rtpmap_line = sdp_video_.substr(start + rtpmap.size(), end - start - rtpmap.size());

    char buffer[512] = {0};
    int sample_rate;
    sscanf(rtpmap_line.c_str(), " %[^/]/%d", buffer, &sample_rate);
    sdp_info_.media_info[0].sample_rate = sample_rate;
    if(std::string(buffer) == std::string("H264")){
        sdp_info_.media_info[0].media_type = MediaEnum::H264;
    }
    else if((std::string(buffer) == std::string("H265")) || (std::string(buffer) == std::string("HEVC"))){
        sdp_info_.media_info[0].media_type = MediaEnum::H265;
    }
    else{
        std::cout << "only support H264/H265, but sdp media type:" << buffer << std::endl;
        return -1;
    }
    // fmtp

    sdp_info_.media_info[0].channels = 0;
    sdp_info_.media_info[0].profile = 0;

    if(sdp_video_.find("a=framerate:") != std::string::npos){
        start = sdp_video_.find("a=framerate:");
        end = sdp_video_.find("\r\n", start);
        std::string framerate = sdp_video_.substr(start + strlen("a=framerate:"), end - start - strlen("a=framerate:"));
        sdp_info_.media_info[0].framerate = std::stoi(framerate);
    }
    else{
        sdp_info_.media_info[0].framerate = 0;
    }
    return 0;
}
/*
m=audio 0 RTP/AVP 97
a=rtpmap:97 MPEG4-GENERIC/44100/2
a=fmtp:97 streamtype=5;profile-level-id=1;mode=AAC-hbr;config=1390;sizelength=13;indexlength=3;indexdeltalength=3
a=control:track1
*/
int SDPParse::ParseAudio(){
    if(sdp_audio_.empty()){
        return 0;
    }
    sdp_info_.media_count++;
    sdp_info_.media_info[1].media_name = "audio";
    int start = sdp_audio_.find("m=audio ");
    int end = sdp_audio_.find("\r\n");
    std::string m_line = sdp_audio_.substr(start + strlen("m=audio "), end - start - strlen("m=audio "));
    std::vector<std::string> res = SplitString(m_line);
    sdp_info_.media_info[1].payload = atoi(res[2].c_str()); // Select the first payload

    start = sdp_audio_.find("a=control:");
    end = sdp_audio_.find("\r\n", start);
    sdp_info_.media_info[1].contorl = sdp_audio_.substr(start + strlen("a=control:"), end - start - strlen("a=control:"));

    std::string rtpmap = std::string("a=rtpmap:") + std::to_string(sdp_info_.media_info[1].payload);
    std::string fmtp = std::string("a=fmtp:") + std::to_string(sdp_info_.media_info[1].payload);
    // rtpmap
    start = sdp_audio_.find(rtpmap.c_str());
    end = sdp_audio_.find("\r\n", start);
    std::string rtpmap_line = sdp_audio_.substr(start + rtpmap.size(), end - start - rtpmap.size());

    char buffer[512] = {0};
    int sample_rate;
    int channels;
    sscanf(rtpmap_line.c_str(), " %[^/]/%d/%d", buffer, &sample_rate, &channels);
    sdp_info_.media_info[1].sample_rate = sample_rate;
    sdp_info_.media_info[1].channels = channels;
    
    if((std::string(buffer) == std::string("MPEG4-GENERIC")) || (std::string(buffer) == std::string("mpeg4-generic"))){
        sdp_info_.media_info[1].media_type = MediaEnum::AAC;
    }
    else if((std::string(buffer) == std::string("PCMA")) || (std::string(buffer) == std::string("pcma"))){
        sdp_info_.media_info[1].media_type = MediaEnum::PCMA;
    }
    else{
        std::cout << "only support AAC(MPEG4-GENERIC)/PCMA, but sdp media type:" << buffer << std::endl;
        return -1;
    }
    // fmtp
    start = sdp_audio_.find(fmtp.c_str());
    end = sdp_audio_.find("\r\n", start);
    std::string fmtp_line = sdp_audio_.substr(start + fmtp.size(), end - start - fmtp.size());

    if(fmtp_line.find("config=") != std::string::npos){
        std::string config_value;
        int pos1 =  fmtp_line.find("config=");
        int pos2 = fmtp_line.find(';', pos1);
        if(pos2 != std::string::npos){
            config_value = fmtp_line.substr(pos1 + strlen("config="), pos2 - pos1 -strlen("config="));
        }
        else{
            config_value = fmtp_line.substr(pos1 + strlen("config="));
        }
        // config_value - Hex
        long int  config = (uint16_t)strtol(config_value.c_str(), NULL, 16);
        int profile_config = (config >> 11) & 0x1f;
        int sample_rate_index_config = (config >> 7) & 0x0f;
        int channels_config = (config >> 3) & 0x0f;
        // std::cout << "profile_config:" << profile_config << "sample_rate_index_config:" << sample_rate_index_config << "channels_config:" << channels_config << std::endl;
        // If there is a config, the one in config shall prevail
        sdp_info_.media_info[1].profile = profile_config;
        sdp_info_.media_info[1].sample_rate_index = sample_rate_index_config;
        sdp_info_.media_info[1].channels = channels_config;
        int freq_arr[13] = {
            96000, 88200, 64000, 48000, 44100, 32000,
            24000, 22050, 16000, 12000, 11025, 8000, 7350
        };
        sdp_info_.media_info[1].sample_rate = freq_arr[sample_rate_index_config];
    }
    else{
        sdp_info_.media_info[1].profile = 1;
        sdp_info_.media_info[1].sample_rate_index = GetSampleRateIndex(sdp_info_.media_info[1].sample_rate);
    }
    return 0;
}