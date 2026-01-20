#include "RtmpPushClient.h"
static uint32_t find_start_code(uint8_t *buf, uint32_t zeros_in_startcode)
{
    uint32_t info;
    uint32_t i;

    info = 1;
    if ((info = (buf[zeros_in_startcode] != 1) ? 0 : 1) == 0)
        return 0;

    for (i = 0; i < zeros_in_startcode; i++)
        if (buf[i] != 0) {
            info = 0;
            break;
        };

    return info;
}
static uint8_t *get_nal(uint32_t *len, uint8_t **offset, uint8_t *start, uint32_t total, uint8_t *prefix_len)
{
    uint32_t info;
    uint8_t *q;
    uint8_t *p = *offset;
    uint8_t prefix_len_z = 0;
    *len = 0;
    *prefix_len = 0;
    while (1) {

        if (((p - start) + 3) >= total)
            return NULL;

        info = find_start_code(p, 2);
        if (info == 1) {
            prefix_len_z = 2;
            *prefix_len = prefix_len_z;
            break;
        }

        if (((p - start) + 4) >= total)
            return NULL;

        info = find_start_code(p, 3);
        if (info == 1) {
            prefix_len_z = 3;
            *prefix_len = prefix_len_z;
            break;
        }
        p++;
    }
    q = p;
    p = q + prefix_len_z + 1;
    prefix_len_z = 0;
    while (1) {
        if (((p - start) + 3) >= total) {
            *len = (start + total - q);
            *offset = start + total;
            return q;
        }

        info = find_start_code(p, 2);
        if (info == 1) {
            prefix_len_z = 2;
            break;
        }

        if (((p - start) + 4) >= total) {
            *len = (start + total - q);
            *offset = start + total;
            return q;
        }

        info = find_start_code(p, 3);
        if (info == 1) {
            prefix_len_z = 3;
            break;
        }

        p++;
    }

    *len = (p - q);
    *offset = p;
    return q;
}
static uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_ms);
}
void WriteCallBack(enum FLVWriteType type, uint8_t* data, uint32_t data_len, void* arg){
    RtmpPushClient *client = (RtmpPushClient*)arg;
    bool need_send = false;
    // WriteCallBack调用顺序 WRITE_FLV_HEADER-->WRITE_FLV_PREVIOUS_SIZE; WRITE_FLV_TAG_HEADER-->WRITE_FLV_*_TAG_DATA-->WRITE_FLV_PREVIOUS_SIZE ...
    // switch (type)
    // {
    //     case WRITE_FLV_HEADER:
    //         printf("WRITE_FLV_HEADER\n");
    //         break;
    //     case WRITE_FLV_PREVIOUS_SIZE:
    //         printf("WRITE_FLV_PREVIOUS_SIZE\n");
    //         break;
    //     case WRITE_FLV_TAG_HEADER:
    //         printf("WRITE_FLV_TAG_HEADER\n");
    //         break;
    //     case WRITE_FLV_AUDIO_CONFIG_TAG_DATA:
    //         printf("WRITE_FLV_AUDIO_CONFIG_TAG_DATA\n");
    //         break;
    //     case WRITE_FLV_AUDIO_TAG_DATA:
    //         printf("WRITE_FLV_AUDIO_TAG_DATA\n");
    //         break;
    //     case WRITE_FLV_VIDEO_CONFIG_TAG_DATA:
    //         printf("WRITE_FLV_VIDEO_CONFIG_TAG_DATA\n");
    //         break;
    //     case WRITE_FLV_VIDEO_TAG_DATA:
    //         printf("WRITE_FLV_VIDEO_TAG_DATA\n");
    //         break;
    //     case WRITE_FLV_SCRIPT_TAG_DATA:
    //         printf("WRITE_FLV_SCRIPT_TAG_DATA\n");
    //         break;
    //     default:
    //         break;
    // }
    switch (type)
    {
        case WRITE_FLV_HEADER:
            break;
        case WRITE_FLV_TAG_HEADER:
        case WRITE_FLV_AUDIO_CONFIG_TAG_DATA:
        case WRITE_FLV_AUDIO_TAG_DATA:
        case WRITE_FLV_VIDEO_CONFIG_TAG_DATA:
        case WRITE_FLV_VIDEO_TAG_DATA:
        case WRITE_FLV_SCRIPT_TAG_DATA:
            if(client->type_ == FLVOutMode::FLV_RTMP){
                memcpy(client->send_buffer_ + client->send_buffer_len_, data, data_len);
                client->send_buffer_len_ += data_len;
            }
            break;
        case WRITE_FLV_PREVIOUS_SIZE:
            if(client->type_ == FLVOutMode::FLV_RTMP){
                if(!client->skip_flv_header_){
                    client->skip_flv_header_ = true;
                }
                else{
                    memcpy(client->send_buffer_ + client->send_buffer_len_, data, data_len);
                    client->send_buffer_len_ += data_len;
                    need_send = true;
                }
            }
            break;
        default:
            break;
    }
    if(client->abort_){
        return;
    }
    if(client->type_ == FLVOutMode::FLV_FILE &&  client->flv_fd_){
        fwrite(data, 1, data_len, client->flv_fd_);
    }
    else if(client->type_ == FLVOutMode::FLV_RTMP && client->rtmp_connect_stat_ && need_send){
        int ret = RTMP_Write(client->rtmp_, (const char *)client->send_buffer_, client->send_buffer_len_);
        if(ret < 0) {
            client->rtmp_connect_stat_ = false;
            log_debug("RTMP_Write error");
        }
        client->send_buffer_len_ = 0;
        memset(client->send_buffer_, 0 , sizeof(client->send_buffer_));
        
    }
}

RtmpPushClient::RtmpPushClient(std::string url, FLVOutMode type){
    // rtmp flv
    url_ = url;
    type_ = type;
    if(type == FLVOutMode::FLV_RTMP){
        log_debug("rtmp stream");
        ConnectServer();
        th_rtmp_reconnect_ = std::thread(&RtmpPushClient::RtmpReconnectThread, this);
    }
    else{
        log_debug("flv file");
        flv_fd_ = fopen(url_.c_str(), "wb");
    }
    // flv
    OpencvFLVHandle();

    th_video_ = std::thread(&RtmpPushClient::VideoStreamThread, this);
    th_audio_ = std::thread(&RtmpPushClient::AudioStreamThread, this);
}
int RtmpPushClient::ConnectServer(){
    rtmp_ = RTMP_Alloc();
    RTMP_Init(rtmp_);
    rtmp_->Link.timeout = 5; // seconds
    rtmp_->Link.lFlags |= RTMP_LF_LIVE;

    if(!RTMP_SetupURL(rtmp_, const_cast<char*>(url_.c_str()))) {
        log_error("Couldn't set the specified url :{}", url_);
        return -1;
    }

    RTMP_EnableWrite(rtmp_);

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
void RtmpPushClient::CloseConnect(){
    if(rtmp_) {
        RTMP_Close(rtmp_);
        RTMP_Free(rtmp_);
        rtmp_ = nullptr;
    }
}
int RtmpPushClient::OpencvFLVHandle(){
    skip_flv_header_ = false;
    context_muxer_ = createFLVContext();
    setWriteCallBack(context_muxer_, WriteCallBack, this);
    int ret = writeFLVGlobalHeader(context_muxer_, 1, 1);
    if(ret < 0){
        log_warn("writeFLVGlobalHeader error");
        return -1;
    }
    AMFDict dict;
    memset((void*)&dict, 0, sizeof(AMFDict));
    setAMFDict(&dict, AMFNUMBER, (uint8_t*)"duration", strlen("duration"), 0.0, 0, nullptr, nullptr, 0);
    ret = writeScriptData(context_muxer_, 0, dict); // AMF1(onMetaData) + AMF2(duration width height,...)
    if(ret < 0){
        log_error("writeScriptData error");
        return -1;
    }
    return 0;
}
void RtmpPushClient::CloseFLVHandle(){
    if(context_muxer_){
        destroyFLVContext(context_muxer_);
        context_muxer_ = nullptr;
    }
}
RtmpPushClient::~RtmpPushClient(){
    abort_ = true;
    video_cond_.notify_all();
    audio_cond_.notify_all();
    th_video_.join();
    th_audio_.join();
    CloseFLVHandle();
    if(type_ == FLVOutMode::FLV_RTMP){
        th_rtmp_reconnect_.join();
        CloseConnect();
    }
    else{
        if(flv_fd_){
            fclose(flv_fd_);
        }
    }
    
    while (!video_list_.empty()) {
        MediaData packet = video_list_.front();
        video_list_.pop_front();
        if(packet.data)
            free(packet.data);
    }
    while (!audio_list_.empty()) {
        MediaData packet = audio_list_.front();
        audio_list_.pop_front();
        if(packet.data)
            free(packet.data);
    }
    if (vps_) {
        free(vps_);
        vps_ = nullptr;
    }
    if (sps_) {
        free(sps_);
        sps_ = nullptr;
    }
    if (pps_) {
        free(pps_);
        pps_ = nullptr;
    }
    log_debug("~RtmpPushClient");
}
// h264/h265
void RtmpPushClient::SetVideoInfo(enum VideoType type){
    video_type_ = type;
    if(video_type_ != VideoType::VIDEO_H264 && video_type_ != VideoType::VIDEO_H265){
        log_error("video type error");
    }
}
void RtmpPushClient::InputVideoData(uint8_t *data, int data_len, int64_t timestamp){
    MediaData packet;
    packet.data = (unsigned char *)malloc(data_len);
    memcpy(packet.data, data, data_len);
    packet.data_len = data_len;
    uint64_t now = GetCurrentTimeMs();
    uint64_t expected = 0;
    start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
    packet.dts = packet.pts = now - start_timestamp_.load(std::memory_order_relaxed);
    std::unique_lock<std::mutex> unique(video_mtx_);
    video_list_.push_back(packet);
    unique.unlock();
    video_cond_.notify_one();
}
// aac
void RtmpPushClient::SetAudioInfo(enum AudioType type){
    audio_type_ = type;
    if(type != AudioType::AUDIO_AAC){
        log_error("audio type error");
    }
}
void RtmpPushClient::InputAudioData(uint8_t *data, int data_len, int64_t timestamp){
    MediaData packet;
    packet.data = (unsigned char *)malloc(data_len);
    memcpy(packet.data, data, data_len);
    packet.data_len = data_len;
    uint64_t now = GetCurrentTimeMs();
    uint64_t expected = 0;
    start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
    packet.dts = packet.pts = now - start_timestamp_.load(std::memory_order_relaxed);
    std::unique_lock<std::mutex> unique(audio_mtx_);
    audio_list_.push_back(packet);
    unique.unlock();
    audio_cond_.notify_one();
}
void RtmpPushClient::VideoStreamThread(){
    int ret = 0;
    bool send_parameters_flag = false;
    while (!abort_) {
        std::unique_lock<std::mutex> unique(video_mtx_);
        if (!video_list_.empty()) {
            MediaData packet = video_list_.front();
            video_list_.pop_front();
            unique.unlock();
            uint8_t *data_nalus = packet.data;
            int len_nalus = packet.data_len;

            uint8_t *p_video = nullptr;
            uint32_t nal_len;
            uint8_t *buf_sffset = data_nalus;
            uint8_t prefix_len = 0;
            uint8_t *video_data = data_nalus;
            uint32_t video_len = len_nalus;
            p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
            while (p_video != nullptr && !abort_) {
                prefix_len = prefix_len + 1;
                uint8_t *data = p_video;
                int data_len = nal_len;
                int nalu_type;
                int start_code = 0;
                if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
                    start_code = 3;
                } else if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
                    start_code = 4;
                }
                if (video_type_ == VIDEO_H264) {
                    nalu_type = data[start_code] & 0x1f;
                    if (!video_ready_) {
                        if (nalu_type == 7) {
                            if (sps_ == nullptr || (sps_buffer_len_ < data_len - start_code)) {
                                sps_ = (uint8_t *)realloc(sps_, data_len - start_code);
                                sps_buffer_len_ = data_len - start_code;
                            }
                            memcpy(sps_, data + start_code, data_len - start_code);
                            sps_len_ = data_len - start_code;

                        } else if (nalu_type == 8) {
                            if (pps_ == nullptr || (pps_buffer_len_ < data_len - start_code)) {
                                pps_ = (uint8_t *)realloc(pps_, data_len - start_code);
                                pps_buffer_len_ = data_len - start_code;
                            }
                            memcpy(pps_, data + start_code, data_len - start_code);
                            pps_len_ = data_len - start_code;
                        }
                    }

                } else if (video_type_ == VIDEO_H265) {
                    nalu_type = (data[start_code] >> 1) & 0x3f;
                    if (!video_ready_) {
                        if (nalu_type == 32) {
                            if (vps_ == nullptr || (vps_buffer_len_ < data_len - start_code)) {
                                vps_ = (uint8_t *)realloc(vps_, data_len - start_code);
                                vps_buffer_len_ = data_len - start_code;
                            }
                            memcpy(vps_, data + start_code, data_len - start_code);
                            vps_len_ = data_len - start_code;
                        } else if (nalu_type == 33) {
                            if (sps_ == nullptr || (sps_buffer_len_ < data_len - start_code)) {
                                sps_ = (uint8_t *)realloc(sps_, data_len - start_code);
                                sps_buffer_len_ = data_len - start_code;
                            }
                            memcpy(sps_, data + start_code, data_len - start_code);
                            sps_len_ = data_len - start_code;

                        } else if (nalu_type == 34) {
                            if (pps_ == nullptr || (pps_buffer_len_ < data_len - start_code)) {
                                pps_ = (uint8_t *)realloc(pps_, data_len - start_code);
                                pps_buffer_len_ = data_len - start_code;
                            }
                            memcpy(pps_, data + start_code, data_len - start_code);
                            pps_len_ = data_len - start_code;
                        }
                    }
                }
                if (sps_len_ != 0 && pps_len_ != 0) {
                    video_ready_ = true;
                }
                if (!video_ready_) {
                    goto NEXT;
                }
                if(!(nalu_type == 6 || nalu_type == 7 || nalu_type == 8 || nalu_type == 32 || nalu_type == 33 || nalu_type == 34)){
                    std::unique_lock<std::mutex> unique_flv(flv_mtx_);
                    if(!send_parameters_flag){
                        if(video_type_ == VideoType::VIDEO_H264){
                            setVideoMediaType(context_muxer_, FLV_VIDEO_H264);
                        }
                        else if(video_type_ == VideoType::VIDEO_H265){
                            setVideoMediaType(context_muxer_, FLV_VIDEO_H265);
                        }
                        else{
                            
                        }
                        ret = setVideoParameters(context_muxer_, vps_, vps_len_, sps_, sps_len_, pps_, pps_len_);
                        if(ret < 0){
                            log_error("setVideoParameters error");
                        }
                        ret = writeVideoSpecificConfig(context_muxer_, 0);
                        if(ret < 0){
                            log_error("writeVideoSpecificConfig error");
                        }
                        send_parameters_flag = true;
                    }
                    ret = writeVideoData(context_muxer_, packet.pts, data + start_code, data_len - start_code);
                    if(ret < 0){
                        log_error("writeVideoData error");
                    }
                }
            NEXT:
                p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
            }
            free(packet.data);
        } else {
            auto now = std::chrono::system_clock::now();
            video_cond_.wait_until(unique, now + std::chrono::milliseconds(100));
            unique.unlock();
            continue;
        }
    }
    log_debug("VideoStreamThread Finished");
}
void RtmpPushClient::AudioStreamThread(){
    int ret = 0;
    auto last_config_time = std::chrono::steady_clock::now();
    while (!abort_) {
        std::unique_lock<std::mutex> unique(audio_mtx_);
        if (!audio_list_.empty()) {
            MediaData packet = audio_list_.front();
            audio_list_.pop_front();
            unique.unlock();
            struct AdtsHeader res;
            ret = ParseAdtsHeader(packet.data, &res);
            if(ret < 0){
                log_error("ParseAdtsHeader error");
                free(packet.data);
                continue;
            }
            uint8_t *data = nullptr;
            int data_len = 0;
            int adts_header_len = (res.protectionAbsent == 1) ? 7 : 9;

            data = packet.data + adts_header_len;
            data_len = packet.data_len - adts_header_len;
            std::unique_lock<std::mutex> unique_flv(flv_mtx_);
            if(!audio_ready_){
                setAudioMediaType(context_muxer_, FLV_AUDIO_AAC);
                ret = writeAudioSpecificConfig(context_muxer_, 0, res.profile, res.samplingFreqIndex, res.channelCfg);
                if(ret < 0){
                    log_error("writeAudioSpecificConfig error");
                }
                audio_ready_ = true;
            }
            ret = writeAudioData(context_muxer_, packet.pts, data, data_len);
            if(ret < 0){
                log_error("writeAudioData error");
            }
            free(packet.data);
        } else {
            auto now = std::chrono::system_clock::now();
            audio_cond_.wait_until(unique, now + std::chrono::milliseconds(100));
            unique.unlock();
            continue;
        }
    }
    log_debug("AudioStreamThread Finished");
}
void RtmpPushClient::RtmpReconnectThread(){
    int ret = 0;
    while (!abort_) {
        if(rtmp_connect_stat_ == false){
            video_ready_ = audio_ready_ = false;
            CloseConnect();
            ConnectServer();
            std::unique_lock<std::mutex> unique_flv(flv_mtx_);
            CloseFLVHandle();
            OpencvFLVHandle();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    log_debug("RtmpReconnectThread Finished");
}