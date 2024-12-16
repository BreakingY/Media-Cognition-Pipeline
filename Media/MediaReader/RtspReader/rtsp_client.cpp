#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <vector>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <poll.h>
#include "rtsp_client.h"
#include "h264_demuxer.h"
#include "h265_demuxer.h"
#include "aac_demuxer.h"
#include "pcma_demuxer.h"
#define RTSP_DEBUG
RtspClient::RtspClient(enum TRANSPORT transport){
    rtp_transport_ = transport;
}
RtspClient::~RtspClient(){
    run_flag_ = false;
    if(run_tid_){
        int ret = pthread_join(tid_, NULL);
        if(ret < 0){
            std::cout << "pthread_join RecvPacketThd error:" << tid_ << std::endl;
        }
        run_tid_ = false;
    }
    if(rtsp_sd_ == -1){
        close(rtsp_sd_);
    }
    if(sdp_){
        delete sdp_;
    }
    if(rtp_sd_video_ >= 0 ){
        close(rtp_sd_video_);
    }
    if(rtcp_sd_video_ >= 0 ){
        close(rtcp_sd_video_);
    }
    if(rtp_sd_audio_ >= 0){
        close(rtp_sd_audio_);
    }
    if(rtcp_sd_audio_ >= 0){
        close(rtcp_sd_audio_);
    }
    if(rtp_video_demuxer_){
        delete rtp_video_demuxer_;
    }
    if(rtp_audio_demuxer_){
        delete rtp_audio_demuxer_;
    }
    std::cout << "~RtspClient" << std::endl;
}
int RtspClient::Connect(char *url){
    int ret;
    enum MediaEnum video_type;
    enum MediaEnum audio_type;
    rtsp_url_ = url;
    bool reslut = ParseRTSPUrl(rtsp_url_, url_info_);
    if(!reslut){
        std::cout << "parseRTSPUrl error" << std::endl;
        return -1;
    }
#ifdef RTSP_DEBUG
    printf("username:%s\n",url_info_.username.c_str());
    printf("password:%s\n",url_info_.password.c_str());
    printf("url:%s\n",url_info_.url.c_str());
    printf("host:%s\n",url_info_.host.c_str());
    printf("port:%d\n",url_info_.port);
#endif
    rtsp_sd_ = CreateTcpSocket();
    if(rtsp_sd_ < 0){
        std::cout << "createTcpSocket error ret:" << rtsp_sd_ << std::endl;
        return -1;
    }
    ret = ConnectToServer(rtsp_sd_, url_info_.host.c_str(), url_info_.port,5000); // 5s
    if(ret < 0){
        connected_ = false;
        std::cout << "ConnectToServer ret:" << ret << std::endl;
        return -1;
    }
    while(rtsp_cmd_stat_ != RTSPCMDSTAT::RTSP_PLAYING){
        switch (rtsp_cmd_stat_)
        {
            case RTSPCMDSTAT::RTSP_NONE:
                if(SendOPTIONS(url_info_.url.c_str()) <= 0){
                    goto faild;
                }
                rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_OPTIONS;
                break;
            case RTSPCMDSTAT::RTSP_OPTIONS:
                if(DecodeOPTIONS(buffer_cmd_, buffer_cmd_size_) < 0){
                    goto faild;
                }
                if(rtsp_cmd_stat_ == RTSPCMDSTAT::RTSP_DESCRIBE){
                    if(SendDESCRIBE(url_info_.url.c_str(), NULL) <= 0){
                        goto faild;
                    }
                }
                break;
            case RTSPCMDSTAT::RTSP_DESCRIBE:            
                if(DecodeDESCRIBE(url_info_.url.c_str(), buffer_cmd_, buffer_cmd_size_) < 0){
                    goto faild;
                }
                if(rtsp_cmd_stat_ == RTSPCMDSTAT::RTSP_STEUP){
                    if(!video_url_.empty()){
                        video_setup_ = true;
                        if(SendSTEUP(video_url_.c_str()) <= 0){
                            goto faild;
                        }
                        rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_STEUP_VIDEO;
                    }
                    else if(!audio_url_.empty()){
                        audio_setup_ = true;
                        if(SendSTEUP(audio_url_.c_str()) <= 0){
                            goto faild;
                        }
                        rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_STEUP_ADUIO;
                    }
                    else{
                        goto faild;
                    }
                }
                break;
            case RTSPCMDSTAT::RTSP_STEUP_VIDEO:
                if(DecodeSTEUP(video_url_.c_str(), buffer_cmd_, buffer_cmd_size_) < 0){
                    goto faild;
                }
                if(rtsp_cmd_stat_ == RTSPCMDSTAT::RTSP_PLAY){
                    if(SendPLAY(url_info_.url.c_str()) <= 0){
                        goto faild;
                    }
                }
                break;
            case RTSPCMDSTAT::RTSP_STEUP_ADUIO:
                if(DecodeSTEUP(audio_url_.c_str(), buffer_cmd_, buffer_cmd_size_) < 0){
                    goto faild;
                }
                if(rtsp_cmd_stat_ == RTSPCMDSTAT::RTSP_PLAY){
                    if(SendPLAY(url_info_.url.c_str()) <= 0){
                        goto faild;
                    }
                }
                break;
            case RTSPCMDSTAT::RTSP_PLAY:
                if(DecodePLAY(url_info_.url.c_str(),buffer_cmd_, buffer_cmd_size_) < 0){
                    goto faild;
                }
                break;
            default:
                break;
        }
        if(rtsp_cmd_stat_ != RTSPCMDSTAT::RTSP_PLAYING){
            if(buffer_cmd_used_ < buffer_cmd_size_){
                memmove(buffer_cmd_, buffer_cmd_ + buffer_cmd_used_, buffer_cmd_size_ - buffer_cmd_used_);
                buffer_cmd_size_ -= buffer_cmd_used_;
            }
            else{
                buffer_cmd_size_ = 0;
                
            }
            buffer_cmd_used_ = 0;
            ret= recv(rtsp_sd_, buffer_cmd_ + buffer_cmd_size_, sizeof(buffer_cmd_) - buffer_cmd_size_, 0);
            if(ret <= 0){
                goto end;
            }
            buffer_cmd_size_ += ret;
            buffer_cmd_[buffer_cmd_size_] = '\0';
#ifdef RTSP_DEBUG
            std::cout <<  __FILE__ << __LINE__ << std::endl;
            std::cout <<  buffer_cmd_ << std::endl;
#endif
        }
    }
    video_type = sdp_->GetVideoType();
    audio_type = sdp_->GetAudioType();
    if(video_type == MediaEnum::H264){
        rtp_video_demuxer_ = new H264Demuxer();
    }
    else if(video_type == MediaEnum::H265){
        rtp_video_demuxer_ = new H265Demuxer();
    }
    if(audio_type == MediaEnum::AAC){
        rtp_audio_demuxer_ = new AACDemuxer();
    }
    else if(audio_type == MediaEnum::PCMA){
        rtp_audio_demuxer_ = new PCMADemuxer();
    }
    if(rtp_video_demuxer_){
        rtp_video_demuxer_->SetCallBack(this);
        rtp_video_demuxer_->SetPayloadType(sdp_->GetVideoPayload());
    }
    if(rtp_audio_demuxer_){
        rtp_audio_demuxer_->SetCallBack(this);
        rtp_audio_demuxer_->SetPayloadType(sdp_->GetAudioPayload());
    }
    /*create recv rtp packet pthread*/
    pthread_create(&tid_, NULL, &RtspClient::RecvPacketThd, this);
    connected_ = true;
    std::cout << "Connect ok url:" << url << std::endl;
    return 0;
end:
    std::cout << "recv data error ret:" << ret << std::endl;
    connected_ = false;
    return -1;
faild:
    std::cout << "CMD error" << std::endl;
    connected_ = false;
    return -1;
}
void RtspClient::OnVideoData(int64_t pts, const uint8_t* data, size_t size){
    if(GetVideoType() == MediaEnum::H264){
        int type = data[4] & 0x1f;
        if(type == 7){
            video_frame_ready_ = true;
        }
    }
    else if(GetVideoType() == MediaEnum::H265){
        int type = (data[4] >> 1) & 0x3f;
        if(type == 32){
            video_frame_ready_ = true;
        }
    }
    if(!video_frame_ready_){
        return;
    }
    if(call_back_){
        call_back_->RtspVideoData(pts, data, size);
    }
    return;
}

void RtspClient::OnAudioData(int64_t pts, const uint8_t* data, size_t size){
    if(call_back_){
        call_back_->RtspAudioData(pts, data, size);
    }
    return;
}
int RtspClient::SendOPTIONS(const char *url){
    char result[512] = {0};
    sprintf(result, "OPTIONS %s RTSP/1.0\r\n"
                    "CSeq: %d\r\n"
                    "User-Agent: %s\r\n"
                    "\r\n",
            url,
            cseq,
            USER_AGENT);
    int ret = send(rtsp_sd_, result, strlen(result), 0);
#ifdef RTSP_DEBUG
    std::cout <<  __FILE__ << __LINE__ << std::endl;
    std::cout <<  result << std::endl;
#endif
    cseq++;
    return ret;
}
int RtspClient::DecodeOPTIONS(const char *buffer, int len){
    std::string str = buffer;
    struct ResponseMessage parsed_message;
    int used_bytes = ParseRTSPMessage(str, parsed_message);
    buffer_cmd_used_ += used_bytes;
    if(parsed_message.code < 0){ // internal error
        return -1;
    }
    rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_DESCRIBE;
    // printf("code:%d\n",parsed_message.code);
    // printf("stat:%s\n",parsed_message.message.c_str());
    // for (const auto& kvp : parsed_message.result) {
    //     std::cout << "Key: " << kvp.first << ", Value: " << kvp.second << std::endl;
    // }
    return 0;
}
int RtspClient::SendDESCRIBE(const char *url, const char *authorization){
    char result[512] = {0};
    sprintf(result, "DESCRIBE %s RTSP/1.0\r\n"
                    "CSeq: %d\r\n"
                    "User-Agent: %s\r\n"
                    "Accept: application/sdp\r\n",
            url,
            cseq,
            USER_AGENT);
    if(authorization){
        sprintf(result+strlen(result),"Authorization: %s\r\n",authorization);
    }
    sprintf(result+strlen(result),"\r\n");
    cseq++;
    int ret = send(rtsp_sd_, result, strlen(result), 0);
#ifdef RTSP_DEBUG
    std::cout <<  __FILE__ << __LINE__ << std::endl;
    std::cout <<  result << std::endl;
#endif
    return ret;
}
static void ExtractRealmAndNonce(const std::string& digest, std::string& realm, std::string& nonce) {
    size_t pos;
    // 提取realm
    pos = digest.find("realm=\"");
    if (pos != std::string::npos) {
        pos += 7; // 跳过 "realm=\""
        size_t end_pos = digest.find("\"", pos);
        if (end_pos != std::string::npos) {
            realm = digest.substr(pos, end_pos - pos);
        }
    }
    // 提取nonce
    pos = digest.find("nonce=\"");
    if (pos != std::string::npos) {
        pos += 7; // 跳过 "nonce=\""
        size_t end_pos = digest.find("\"", pos);
        if (end_pos != std::string::npos) {
            nonce = digest.substr(pos, end_pos - pos);
        }
    }
    return;
}
std::string RtspClient::GenerateAuthHeader(std::string url, std::string response){
    // Authorization: Digest username="admin", realm="_", nonce="10839044", uri="rtsp://192.168.0.49:554/11", response="f1bf854a901dc8a7379ff277ce1be0e3"
    std::string str = std::string("Digest username=\"") + url_info_.username + std::string("\", realm=\"") + realm_ + std::string("\", nonce=\"") 
                                       + nonce_ + std::string("\", uri=\"") + url + std::string("\", response=\"") + response + std::string("\"");
    return str;
}
int RtspClient::DecodeDESCRIBE(const char *url, const char *buffer, int len){
    std::string str = buffer;
    struct ResponseMessage parsed_message;
    int used_bytes = ParseRTSPMessage(str, parsed_message);
    buffer_cmd_used_ += used_bytes;
    if(parsed_message.code < 0){ // internal error
        return -1;
    }
    if(parsed_message.code == 401){ // Unauthorized
        std::string authenticate = GetValueByKey(parsed_message.result, "WWW-Authenticate");
        if(authenticate.empty()){
            buffer_cmd_used_ -= used_bytes;
            return 0;
        }
        ExtractRealmAndNonce(authenticate, realm_, nonce_);
        std::string response = GenerateAuthResponse(url_info_.username.c_str(), url_info_.password.c_str(), realm_.c_str(), nonce_.c_str(), url, "DESCRIBE");
        std::string res = GenerateAuthHeader(url, response);
        SendDESCRIBE(url, res.c_str());
        return 0;
    }
    else{ // 解析SDP
        // printf("code:%d\n",parsed_message.code);
        // printf("stat:%s\n",parsed_message.message.c_str());
        // printf("sdp:%s\n",parsed_message.sdp.c_str());
        // for (const auto& kvp : parsed_message.result) {
        //     std::cout << "Key: " << kvp.first << ", Value: " << kvp.second << std::endl;
        // }
        std::string content_len_str = GetValueByKey(parsed_message.result, "Content-Length");
        if(content_len_str.empty() || !parsed_message.find_payload){
            buffer_cmd_used_ -= used_bytes;
            return 0;
        }
        int content_len = std::stoi(content_len_str);
        if((len - used_bytes) < content_len){ // payload不完整
            buffer_cmd_used_ -= used_bytes;
            return 0;
        }
        parsed_message.sdp = str.substr(used_bytes, content_len);
        buffer_cmd_used_ += content_len;
        content_base_ = GetValueByKey(parsed_message.result, "Content-Base");
        if(content_base_.empty()){
            content_base_ = url;
        }
        if(sdp_ == NULL){
            sdp_ = new SDPParse(parsed_message.sdp, content_base_);
            sdp_->Parse();
            video_url_ = sdp_->GetVideoUrl();
            audio_url_ = sdp_->GetAudioUrl();
        }
    }
    rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_STEUP;
    return 0;
}
int RtspClient::SendSTEUP(const char *url){
    std::string authenticate;
    if(!realm_.empty() && !nonce_.empty()){
        std::string response = GenerateAuthResponse(url_info_.username.c_str(), url_info_.password.c_str(), realm_.c_str(), nonce_.c_str(), url, "SETUP");
        authenticate = GenerateAuthHeader(url, response);
    }

    char result[512] = {0};
    sprintf(result, "SETUP %s RTSP/1.0\r\n"
                    "CSeq: %d\r\n"
                    "User-Agent: %s\r\n",
            url,
            cseq,
            USER_AGENT);
    if(!authenticate.empty()){
        sprintf(result+strlen(result),"Authorization: %s\r\n",authenticate.c_str());
    }
    if(!session_.empty()){
        sprintf(result+strlen(result),"Session: %s\r\n",session_.c_str());
    }
    if(rtp_transport_ == TRANSPORT::RTP_OVER_UDP){
        if(std::string(url) == video_url_){
            if(CreateRtpSockets(&rtp_sd_video_, &rtcp_sd_video_, &rtp_port_video_, &rtcp_port_video_) < 0){
                std::cout << "video CreateRtpSockets error" << std::endl;
                return -1;
            }
            sprintf(result+strlen(result),"Transport: RTP/AVP;unicast;client_port=%d-%d\r\n",rtp_port_video_, rtcp_port_video_);
        }
        else if(std::string(url) == audio_url_){
            if(CreateRtpSockets(&rtp_sd_audio_, &rtcp_sd_audio_, &rtp_port_audio_, &rtcp_port_audio_) < 0){
                std::cout << "audio CreateRtpSockets error" << std::endl;
                return -1;
            }
            sprintf(result+strlen(result),"Transport: RTP/AVP;unicast;client_port=%d-%d\r\n",rtp_port_audio_, rtcp_port_audio_);
        }
        else{
            return -1;
        }
    }
    else{
        if(std::string(url) == video_url_)
            sprintf(result+strlen(result),"Transport: RTP/AVP/TCP;unicast;interleaved=%d-%d\r\n", sig0_video_, sig0_video_ + 1);
        else if(std::string(url) == audio_url_)
            sprintf(result+strlen(result),"Transport: RTP/AVP/TCP;unicast;interleaved=%d-%d\r\n", sig0_audio_, sig0_audio_ + 1);
        else
            return -1;
    }
    sprintf(result+strlen(result),"\r\n");
    cseq++;
    int ret = send(rtsp_sd_, result, strlen(result), 0);
#ifdef RTSP_DEBUG
    std::cout <<  __FILE__ << __LINE__ << std::endl;
    std::cout <<  result << std::endl;
#endif
    return ret;
}
int RtspClient::DecodeSTEUP(const char *url, const char *buffer, int len){
    std::string str = buffer;
    struct ResponseMessage parsed_message;
    int used_bytes = ParseRTSPMessage(str, parsed_message);
    buffer_cmd_used_ += used_bytes;
    if(parsed_message.code < 0){ // internal error
        return -1;
    }
    std::string session = GetValueByKey(parsed_message.result, "Session");
    if(session.empty()){
        buffer_cmd_used_ -= used_bytes;
        return 0;
    }
    if(session_.empty()){
        int pos = session.find(';');
        if(pos != std::string::npos)
            session_ = session.substr(0, pos);
        else
            session_ = session;
        pos = session.find("timeout=");
        if(pos != std::string::npos){
            int pos1 = session.find(';', pos);
            std::string timeout_str;
            if(pos1 == std::string::npos){
                timeout_str = session.substr(pos + strlen("timeout="));
            }
            else{
                timeout_str = session.substr(pos + strlen("timeout="), pos1 - pos - strlen("timeout="));
            }
            timeout_ = atoi(timeout_str.c_str());
            std::cout << "timeout_:" << timeout_ << std::endl;
        }
    }
    if(rtp_transport_ == TRANSPORT::RTP_OVER_UDP){
        std::string transport = GetValueByKey(parsed_message.result, "Transport");
        if(transport.empty()){
            buffer_cmd_used_ -= used_bytes;
            return 0;
        }
        int pos = transport.find("server_port=");
        if(pos == std::string::npos){
            std::cout << "server Transport error:" << transport << std::endl;
            return -1;
        }
        int pos1 = transport.find(';',pos);
        std::string port_rtp;
        if(pos1 == std::string::npos)
            port_rtp = transport.substr(pos + strlen("server_port="));
        else
            port_rtp = transport.substr(pos + strlen("server_port="), pos1 - pos - strlen("server_port="));
        if(std::string(url) == video_url_){
            sscanf(port_rtp.c_str(), "%d-%d", &rtp_port_video_server_, &rtcp_port_video_server_);
        }
        else{
            sscanf(port_rtp.c_str(), "%d-%d", &rtp_port_audio_server_, &rtcp_port_audio_server_);
        }
    }
    else{
        // tcp donothing
    }
    // printf("code:%d\n",parsed_message.code);
    // printf("stat:%s\n",parsed_message.message.c_str());
    // for (const auto& kvp : parsed_message.result) {
    //     std::cout << "Key: " << kvp.first << ", Value: " << kvp.second << std::endl;
    // }
    if(!audio_url_.empty() && !audio_setup_){
        audio_setup_ = true;
        if(SendSTEUP(audio_url_.c_str()) <= 0){
            return -1;
        }
        rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_STEUP_ADUIO;
        return 0;
    }
    if(!video_url_.empty() && !video_setup_){
        video_setup_ = true;
        if(SendSTEUP(video_url_.c_str()) <= 0){
            return -1;
        }
        rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_STEUP_VIDEO;
        return 0;
    }
    rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_PLAY;
    return 0;
}
int RtspClient::SendPLAY(const char *url){
    char result[512] = {0};
    sprintf(result, "PLAY %s RTSP/1.0\r\n"
                    "CSeq: %d\r\n"
                    "User-Agent: %s\r\n",
            url,
            cseq,
            USER_AGENT);
    std::string authenticate;
    if(!realm_.empty() && !nonce_.empty()){
        std::string response = GenerateAuthResponse(url_info_.username.c_str(), url_info_.password.c_str(), realm_.c_str(), nonce_.c_str(), url, "PLAY");
        authenticate = GenerateAuthHeader(url, response);
    }
    if(!authenticate.empty()){
        sprintf(result+strlen(result),"Authorization: %s\r\n",authenticate.c_str());
    }
    if(!session_.empty()){
        sprintf(result+strlen(result),"Session: %s\r\n",session_.c_str());
    }
    sprintf(result+strlen(result),"Range: npt=0.000-\r\n");
    sprintf(result+strlen(result),"\r\n");
    cseq++;
    int ret = send(rtsp_sd_, result, strlen(result), 0);
#ifdef RTSP_DEBUG
    std::cout <<  __FILE__ << __LINE__ << std::endl;
    std::cout <<  result << std::endl;
#endif
    return ret;
}
int RtspClient::DecodePLAY(const char *url, const char *buffer, int len){
    std::string str = buffer;
    struct ResponseMessage parsed_message;
    int used_bytes = ParseRTSPMessage(str, parsed_message);
    buffer_cmd_used_ += used_bytes;
    if(parsed_message.code < 0){ // internal error
        return -1;
    }
    std::string session = GetValueByKey(parsed_message.result, "Session");
    if(session.empty()){
        buffer_cmd_used_ -= used_bytes;
        return 0;
    }
    int pos = session.find("timeout=");
    if(pos != std::string::npos){
        int pos1 = session.find(';', pos);
        std::string timeout_str;
        if(pos1 == std::string::npos){
            timeout_str = session.substr(pos + strlen("timeout="));
        }
        else{
            timeout_str = session.substr(pos + strlen("timeout="), pos1 - pos - strlen("timeout="));
        }
        timeout_ = atoi(timeout_str.c_str());
    }
    rtsp_cmd_stat_ = RTSPCMDSTAT::RTSP_PLAYING;
    return 0;
}
int RtspClient::ReadPacketUdp(){
    int  bytes = 0;
    unsigned char buffer[READ_SOCK_DATA_LEN] = {0};
    fd_set read_fds;
    FD_ZERO(&read_fds);
    std::vector<int> array_fd;
    if(rtp_port_video_server_ != -1){ // video
        FD_SET(rtp_sd_video_, &read_fds);
        FD_SET(rtcp_sd_video_, &read_fds);
        array_fd.push_back(rtp_sd_video_);
        array_fd.push_back(rtcp_sd_video_);
    }
    if(rtp_port_audio_server_ != -1){ // audio
        FD_SET(rtp_sd_audio_, &read_fds);
        FD_SET(rtcp_sd_audio_, &read_fds);
        array_fd.push_back(rtp_sd_audio_);
        array_fd.push_back(rtcp_sd_audio_);
    }
    // Set timeout
    struct timeval timeout;
    timeout.tv_sec = recv_rtp_packet_timeout_;
    timeout.tv_usec = 0;
    int max_fd;
    auto maxElementIter = std::max_element(array_fd.begin(), array_fd.end());
    max_fd = *maxElementIter;
    int ret = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
    if(ret < 0){
        std::cout << rtsp_url_ << ":network error" << std::endl;
        return -1;
	}
    else if(ret == 0)
    {
        std::cout << rtsp_url_ << ":select time out" << std::endl;
        return -1;
    }
    for(int i = 0; i < array_fd.size(); i++){
        if(FD_ISSET(array_fd[i], &read_fds)){
            struct sockaddr_in sender_addr;
            socklen_t sender_addr_len = sizeof(sender_addr);
            bytes = recvfrom(array_fd[i], buffer, READ_SOCK_DATA_LEN, 0, (struct sockaddr*)&sender_addr, &sender_addr_len);
            if (bytes <= 0) {
                std::cout << rtsp_url_ << ":recvfrom error" << std::endl;
                return -1;
            }
            if(array_fd[i] == rtp_sd_video_){
                rtp_video_demuxer_->InputData((const uint8_t*)buffer, bytes);
            }
            if(array_fd[i] == rtp_sd_audio_){
                rtp_audio_demuxer_->InputData((const uint8_t*)buffer, bytes);
            }
            // ignore rtcp
        }

    }
    return bytes;
}
int RtspClient::ReadPacketTcp(){
    int bytes = 0;
    unsigned char buffer[READ_SOCK_DATA_LEN] = {0};
    struct pollfd poll_event;
    poll_event.fd = rtsp_sd_;
    poll_event.events = POLLIN | POLLPRI;
    poll_event.revents = 0;
    int ret = poll(&poll_event, 1, recv_rtp_packet_timeout_ * 1000);
    if(ret < 0){
        std::cout << rtsp_url_ << ":network error" << std::endl;
        return -1;
	}
    else if(ret == 0)
    {
        std::cout << rtsp_url_ << ":poll time out" << std::endl;
        return -1;
    }
    bytes = recv(rtsp_sd_, buffer, READ_SOCK_DATA_LEN, 0);
    if (bytes <= 0) {
        std::cout << rtsp_url_ << ":recv error" << std::endl;
        return -1;
    }
    buffer[bytes] = '\0';
    unsigned char *ptr = buffer;
    int pos_buffer_end = bytes;
    while(ptr < buffer + pos_buffer_end){
        if((stat_ == EMPTY_STATE) || (stat_ == RTSP_MESSAGE_STATE)){
            if(*ptr == '$'){
                buffer_header_[pos_buffer_header_++] = *ptr;
                ptr++;
                stat_ = RTP_TCP_HEADER_STATE;
            }
            else{
                stat_ = RTSP_MESSAGE_STATE;
            }
        }
        else if(stat_ == RTP_TCP_HEADER_STATE){
            buffer_header_[pos_buffer_header_++] = *ptr;
            ptr++;
            if(pos_buffer_header_ == 4){
                stat_ = RTP_TCP_CONTENT_STATE;
                header_.channel = buffer_header_[1];
                for(int i = 2; i <= 3; i++){
                    header_.rtp_len16 <<= 8;
                    header_.rtp_len16 |= buffer_header_[i];
                } 
                pos_buffer_header_ = 0;
            }
        }
        else if(stat_ == RTP_TCP_CONTENT_STATE){
            if(header_.rtp_len16 > pos_buffer_){
                buffer_[pos_buffer_++] = *ptr;
                ptr++;
            }
            else{
                if(header_.channel == sig0_video_){ // video
                    rtp_video_demuxer_->InputData(buffer_, pos_buffer_);
                }
                else if(header_.channel == sig0_audio_){ // audio
                    rtp_audio_demuxer_->InputData(buffer_, pos_buffer_);
                }
                pos_buffer_ = 0;
                header_.rtp_len16 = 0;
                stat_ = EMPTY_STATE;
            }
        }
        if(stat_ == RTSP_MESSAGE_STATE){
            // skip rtsp message
            ptr++;
        }
    }
    return pos_buffer_end;
}
static void * RtspClient::RecvPacketThd(void *arg){
    RtspClient *self = (RtspClient*)arg;
    self->run_tid_ = true;
    struct timeval pre_time;
    struct timeval now_time;
    gettimeofday(&now_time, 0);
    pre_time = now_time;
    int ret;
    self->header_.channel = 0;
    self->header_.rtp_len16 = 0;
    while(self->run_flag_){
        //心跳
        gettimeofday(&now_time, 0);
        int64_t time_gap = now_time.tv_sec - pre_time.tv_sec;
        if(time_gap >= self->timeout_){
            ret = self->SendOPTIONS(self->url_info_.url.c_str());
            if(ret <= 0 ){
                std::cout << "send heartbeat failure" << std::endl;
                self->connected_ = false;
            }
            pre_time = now_time;
        }
        if(self->rtp_transport_ == TRANSPORT::RTP_OVER_UDP){
            if(self->ReadPacketUdp() < 0){
                self->connected_ = false;
            }
        }
        else{
            if(self->ReadPacketTcp() < 0){
                self->connected_ = false;
            }
        }
    }
    self->run_tid_ = false;
    return NULL;
}
