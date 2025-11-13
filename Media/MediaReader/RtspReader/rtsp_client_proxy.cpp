#include "rtsp_client_proxy.h"
extern "C" {
    #include "h264-sps.h"
    #include "h265-sps.h"
}
RtspClientProxy::RtspClientProxy(char *rtsp_url){
    rtsp_url_ = rtsp_url;
    client_ =  new RtspClient(transport_); 
    client_->Connect(rtsp_url); 
    client_->SetCallBack(this);
    tid_ = std::thread(RtspClientProxy::ReconnectThread, this);
}
RtspClientProxy::~RtspClientProxy(){
    run_flag_ = false;
    tid_.join();
    delete client_;
    std::cout << "~RtspClientProxy" << std::endl;
}
void *RtspClientProxy::ReconnectThread(void *arg){
    RtspClientProxy *self = (RtspClientProxy*)arg;
    while(self->run_flag_){
        bool stat = self->client_->GetOpenStat();
        if(stat == false){
            if((self->probe_cnt_ < PROBEFRAME) && (self->fps_ < 0)){
                self->probe_cnt_ = 0;
                self->fps_ = -1;
            }
            std::cout << self->rtsp_url_ << " Reconnect" << std::endl;
            delete self->client_;
            self->client_ =  new RtspClient(self->transport_); 
            self->client_->Connect(self->rtsp_url_.c_str()); 
            self->client_->SetCallBack(self);

        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return NULL;
}
void RtspClientProxy::GetVideoCon(int &width, int &height, int &fps){
    width = width_;
    height = height_;
    fps = fps_;
    return;
}
void RtspClientProxy::GetAudioCon(int &sample_rate_index, int &channels, int &profile){
    client_->GetAudioInfo(sample_rate_index, channels, profile);
    return;
}
enum VideoType RtspClientProxy::GetVideoType(){
    if(client_->GetVideoType() == MediaEnum::H264){
        return VideoType::VIDEO_H264;
    }
    else if(client_->GetVideoType() == MediaEnum::H265){
        // std::cout << "video type:" << ((data[4] >> 1) & 0x3f) << std::endl;
        return VideoType::VIDEO_H265;
    }
    return VideoType::VIDEO_NONE;
}
enum AudioType RtspClientProxy::GetAudioType(){
    if(client_->GetAudioType() == MediaEnum::AAC){
        return AudioType::AUDIO_AAC;
    }
    else if(client_->GetAudioType() == MediaEnum::PCMA){
        return AudioType::AUDIO_PCMA;
    }
    return AudioType::AUDIO_NONE;

}
void RtspClientProxy::RtspVideoData(int64_t pts, const uint8_t* data, size_t size){
    if(fps_ == -1 && client_->GetFramerate() > 0){
        fps_ = client_->GetFramerate();
    }
    int type;
    if(client_->GetVideoType() == MediaEnum::H264){
        // std::cout << "video type:" << (data[4] & 0x1f) << std::endl;
        type = data[4] & 0x1f;
        if(type == 7){
            video_ready_ = true;
            struct h264_sps_t sps;
            h264_sps_parse(data + 4, size - 4, &sps);
            int x, y;
            h264_display_rect(&sps, &x, &y, &width_, &height_);
        }
    }
    else if(client_->GetVideoType() == MediaEnum::H265){
        // std::cout << "video type:" << ((data[4] >> 1) & 0x3f) << std::endl;
        type = (data[4] >> 1) & 0x3f;
        if(type == 33){
            struct h265_sps_t sps;
            h265_sps_parse(data + 4, size - 4, &sps);
            int x, y;
            h265_display_rect(&sps, &x, &y, &width_, &height_);
        }
        if((width_ != -1) && (type == 32)){
            video_ready_ = true;
        }  
    }
    // probe fps
    if(!(type == 6 || type == 7 || type == 8 ||type == 32 || type == 33 || type == 34) && 
        (probe_cnt_ < PROBEFRAME) && (fps_ < 0)){
        if(last_timestamp_ == -1){
            last_timestamp_ = pts;
        }
        else{
            int interval = pts - last_timestamp_;
            interval_sum_ += interval;
            last_timestamp_ = pts;
            probe_cnt_++;
            if(probe_cnt_ == PROBEFRAME){
                fps_ = 90000 / (interval_sum_ / probe_cnt_);
            }
        }
    }
    if(fps_ < 0){
        video_ready_ = false;
    }
    if(!video_ready_){
        return;
    }
    VideoData video_data;
    video_data.data = (unsigned char *)data;
    video_data.data_len = size;
    video_data.pts = 0;
    video_data.dts = 0;
    if (data_listner_) {
        data_listner_->OnVideoData(video_data);
    }
    else{
        video_ready_ = false;
    }
    return;
}
void RtspClientProxy::RtspAudioData(int64_t pts,  const uint8_t* data, size_t size){
    if(client_->GetAudioType() == MediaEnum::AAC){
        char adts_header_buf[7] = {0};
        int profile, sample_rate_index, channels;
        client_->GetAudioInfo(sample_rate_index, channels, profile);
        GenerateAdtsHeader(adts_header_buf, size,
                    profile,//AAC编码级别
                    sample_rate_index,//采样率 Hz
                    channels);
        AudioData audio_data;
        unsigned char buffer [4 * 1024] = {0};
        memcpy(buffer, adts_header_buf, 7);
        memcpy(buffer + 7, data, size);
        audio_data.data = buffer;
        audio_data.data_len = size + 7;
        audio_data.channels = channels;
        audio_data.profile = profile;
        int freq_arr[13] = {
            96000, 88200, 64000, 48000, 44100, 32000,
            24000, 22050, 16000, 12000, 11025, 8000, 7350
        };
        audio_data.samplerate = freq_arr[sample_rate_index];
        if(data_listner_){
            data_listner_->OnAudioData(audio_data);
        }
    }
    else if(client_->GetAudioType() == MediaEnum::PCMA){
    
    }
    return;
}