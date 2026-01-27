#include "RtmpPullClient.h"
extern "C" {
    #include "h264-sps.h"
    #include "h265-sps.h"
}
static uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_ms);
}
void audioCallBack(enum FLVAudioType type, int profile, int sample_rate_index, int channel, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg){
    if(type != FLV_AUDIO_AAC){
        return;
    }
    RtmpPullClient *client = (RtmpPullClient*)arg;
    client->audio_type_ = AudioType::AUDIO_AAC;
    client->profile_ = profile;
    client->sample_rate_index_ = sample_rate_index;
    client->channels_ = channel;
    if(!client->probe_over_flag_){
        return;
    }

    unsigned char *data_withadts= (unsigned char *)malloc(data_len + 7);
    char adts_header_buffer[7];
    GenerateAdtsHeader(adts_header_buffer, data_len, profile, sample_rate_index, channel);
    memcpy(data_withadts, adts_header_buffer, 7);
    memcpy(data_withadts + 7, data, data_len);
    
    AudioData audio_data;
    audio_data.data = data_withadts;
    audio_data.data_len = data_len + 7;
    audio_data.channels = channel;
    audio_data.profile = profile;
    int freq_arr[13] = {
        96000, 88200, 64000, 48000, 44100, 32000,
        24000, 22050, 16000, 12000, 11025, 8000, 7350
    };
    audio_data.samplerate = freq_arr[sample_rate_index];
    if(client->data_listner_){
        client->data_listner_->OnAudioData(audio_data);
    }
    free(data_withadts);
    if(client->type_ == FLVOutMode::FLV_FILE){     
        uint64_t now = GetCurrentTimeMs();
        uint64_t expected = 0;
        client->start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
        int64_t pts = now - client->start_timestamp_.load(std::memory_order_relaxed);
        if(timestamp > 0 && timestamp > pts){
            std::this_thread::sleep_for(std::chrono::milliseconds(timestamp - pts));
        }
    }
    return;
}
void videoCallBack(enum FLVVideoType type, int64_t timestamp, uint8_t* data, uint32_t data_len, void* arg){
    RtmpPullClient *client = (RtmpPullClient*)arg;
    int nalu_type;
    if(type == FLV_VIDEO_H264){
        nalu_type = data[0] & 0x1f;
        client->video_type_ = VideoType::VIDEO_H264;
        if(nalu_type == 7){
            struct h264_sps_t sps;
            h264_sps_parse(data, data_len, &sps);
            int x, y;
            h264_display_rect(&sps, &x, &y, &client->width_, &client->height_);
        }
    }
    else if(type == FLV_VIDEO_H265){
        nalu_type = (data[0] >> 1) & 0x3f;
        client->video_type_ = VideoType::VIDEO_H265;
        if(nalu_type == 33){
            struct h265_sps_t sps;
            h265_sps_parse(data, data_len, &sps);
            int x, y;
            h265_display_rect(&sps, &x, &y, &client->width_, &client->height_);
        }
    }
    else{
        return;
    }
    // probe fps
    if(!(nalu_type == 6 || nalu_type == 7 || nalu_type == 8 ||nalu_type == 32 || nalu_type == 33 || nalu_type == 34) && 
        (client->probe_cnt_ < client->probe_frame_num_) && (client->fps_ < 0)){
        if(client->last_timestamp_ == -1){
            client->last_timestamp_ = timestamp;
        }
        else{
            int interval = timestamp - client->last_timestamp_;
            client->interval_sum_ += interval;
            client->last_timestamp_ = timestamp;
            client->probe_cnt_++;
            if(client->probe_cnt_ == client->probe_frame_num_){
                client->fps_ = 1000 / (client->interval_sum_ / client->probe_cnt_);
            }
        }
    }
    if(!client->probe_over_flag_ && client->width_ > 0 && client->height_ > 0 && client->fps_ > 0){
        if(client->type_ == FLVOutMode::FLV_FILE){
            terminateDemuxerFLVFile(client->context_demuxer_);
        }
        client->probe_over_flag_ = true;
    }
    if(client->probe_over_flag_ && !client->video_ready_ && (nalu_type == 7/*h264 sps*/ || nalu_type == 32/*h265 vps*/)){
        client->video_ready_ = true;
    }
    if(!client->video_ready_){
        return;
    }
    char start_code[4] = {0, 0, 0, 1};
    unsigned char *data_withwtartcode = (unsigned char *)malloc(data_len + 4);
    memcpy(data_withwtartcode, start_code, 4);
    memcpy(data_withwtartcode + 4, data, data_len);
    VideoData video_data;
    video_data.data = data_withwtartcode;
    video_data.data_len = data_len + 4;
    video_data.pts = 0;
    video_data.dts = 0;
    if (client->data_listner_) {
        client->data_listner_->OnVideoData(video_data);
    }
    free(data_withwtartcode);
    if(client->type_ == FLVOutMode::FLV_FILE){ 
        uint64_t now = GetCurrentTimeMs();
        uint64_t expected = 0;
        client->start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
        int64_t pts = now - client->start_timestamp_.load(std::memory_order_relaxed);
        if(timestamp > 0 && timestamp > pts){
            std::this_thread::sleep_for(std::chrono::milliseconds(timestamp - pts));
        }
    }
    return; 
}
void scriptDataCallBack(AMFDict dict, void* arg){
    RtmpPullClient *client = (RtmpPullClient*)arg;
    // printAMFDict(dict);
    for(int i = 0; i < dict.key_value_len; i++){
        KV *k_v = &dict.key_value[i];
        switch (k_v->value_type)
        {
            case AMFNUMBER:
                if(memcmp("width", k_v->key, strlen("width")) == 0){
                    client->width_ = (int)k_v->value.number;
                }
                else if(memcmp("height", k_v->key, strlen("height")) == 0){
                    client->height_ = (int)k_v->value.number;
                }
                else if(memcmp("framerate", k_v->key, strlen("framerate")) == 0){
                    client->fps_ = (int)k_v->value.number;
                }
                break;
            case AMFBOOLEAN:
                break;
            case AMFSTRING:
                break;
            case AMFLONGSTRING:
                break;
            default:
                break;
        }
    }
    if(!client->probe_over_flag_ && client->width_ > 0 && client->height_ > 0 && client->fps_ > 0){
        if(client->type_ == FLVOutMode::FLV_FILE){
            terminateDemuxerFLVFile(client->context_demuxer_);
        }
        client->probe_over_flag_ = true;
    }
    return;
}
RtmpPullClient::RtmpPullClient(std::string url, FLVOutMode type){
    type_ = type;
    url_ = url;
    if(type == FLVOutMode::FLV_FILE){
        log_debug("flv stream");
    }
    else{
        log_debug("rtmp file");
    }
    // probe width height fps
    context_demuxer_ = createFLVContext();
    setReadCallBack(context_demuxer_, audioCallBack, videoCallBack, scriptDataCallBack, this);
    if(type == FLVOutMode::FLV_FILE){
        int ret = demuxerFLVFile(context_demuxer_, const_cast<char*>(url_.c_str()));
        if(ret < 0){
            log_error("demuxerFLVFile error");
        }
        destroyFLVContext(context_demuxer_);
        // start reading
        context_demuxer_ = createFLVContext();
        setReadCallBack(context_demuxer_, audioCallBack, videoCallBack, scriptDataCallBack, this);
    }
    else{
        ConnectServer();
        th_rtmp_reconnect_ = std::thread(&RtmpPullClient::RtmpReconnectThread, this);

    }
    th_flv_handle_ = std::thread(&RtmpPullClient::FLVStreamReadThread, this);
}
RtmpPullClient::~RtmpPullClient(){
    abort_ = true;
    terminateDemuxerFLVFile(context_demuxer_); // demuxerFLVFile blocks execution and requires calling terminateDemuxerFLVFile to stop parsing
    th_flv_handle_.join();
    if(type_ == FLVOutMode::FLV_RTMP){
        th_rtmp_reconnect_.join();
        CloseConnect();
    }
    destroyFLVContext(context_demuxer_);
    log_debug("~RtmpPullClient");
}
int RtmpPullClient::ConnectServer(){
    rtmp_ = RTMP_Alloc();
    RTMP_Init(rtmp_);
    rtmp_->Link.timeout = 5; // seconds
    rtmp_->Link.lFlags |= RTMP_LF_LIVE;

    if(!RTMP_SetupURL(rtmp_, const_cast<char*>(url_.c_str()))) {
        log_error("Couldn't set the specified url :{}", url_);
        return -1;
    }

    if(!RTMP_Connect(rtmp_, nullptr)) {
        log_error("RTMP_Connect error :{}", url_);
        return -1;
    }

    if(!RTMP_ConnectStream(rtmp_, 0)) {
        log_error("RTMP_ConnectStream error :{}", url_);
        return -1;
    }
    rtmp_connect_stat_ = true;
    return 0;
}
void RtmpPullClient::CloseConnect(){
    if(rtmp_) {
        RTMP_Close(rtmp_);
        RTMP_Free(rtmp_);
        rtmp_ = nullptr;
    }
}
void RtmpPullClient::FLVStreamReadThread(){
    while(!abort_){
        if(type_ == FLVOutMode::FLV_FILE){
            int ret = demuxerFLVFile(context_demuxer_, const_cast<char*>(url_.c_str()));
            if(ret < 0){
                log_error("demuxerFLVFile error");
            }
            if(colse_cb_ != nullptr){
                colse_cb_();
            }
        }
        else if(type_ == FLVOutMode::FLV_RTMP && rtmp_connect_stat_){
            
            int ret = RTMP_Read(rtmp_, (char *)recv_buffer_, sizeof(recv_buffer_));
            if(ret <= 0){
                rtmp_connect_stat_ = false;
                continue;
            }
            int pos = 0;
            int remain_bytes = ret;
            while(remain_bytes > 0 && !abort_) {
                if(!read_flv_header_) {
                    readFLVHeader(&context_demuxer_->flv_header, recv_buffer_ + pos, remain_bytes);
                    pos += FLV_HEADER_SIZE + FLV_PREVIOUS_SIZE;
                    read_flv_header_ = true;
                } else {
                    readTagHeader(&context_demuxer_->tag_header, recv_buffer_ + pos, remain_bytes);
                    pos += FLV_TAG_HEADER_SIZE;

                    switch (context_demuxer_->tag_header.flv_media_type) {
                        case FLV_VIDEO:
                            readVideoTagData(context_demuxer_, recv_buffer_ + pos, remain_bytes - (pos));
                            break;
                        case FLV_AUDIO:
                            readAudioTagData(context_demuxer_, recv_buffer_ + pos, remain_bytes - (pos));
                            break;
                        case FLV_SCRIPT_DATA:
                            readScriptDataTagData(context_demuxer_, recv_buffer_ + pos, remain_bytes - (pos));
                            break;
                        default:
                            break;
                    }

                    pos += context_demuxer_->tag_header.data_size + FLV_PREVIOUS_SIZE;
                }

                remain_bytes = ret - pos;
            }

        } 
    }
    log_debug("FLVStreamReadThread Finished");
}
void RtmpPullClient::RtmpReconnectThread(){
    int ret = 0;
    while (!abort_) {
        if(rtmp_connect_stat_ == false){
            log_debug("{} reconnecting ...", url_);
            video_ready_ = false;
            CloseConnect();
            ConnectServer();
            if(rtmp_connect_stat_){
                log_debug("{} connection successful", url_);
            }
            else{
                log_debug("{} connection failed", url_);
            }
            
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    log_debug("RtmpReconnectThread Finished");
}
