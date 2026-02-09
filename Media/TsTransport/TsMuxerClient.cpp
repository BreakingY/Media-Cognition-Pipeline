#include "TsMuxerClient.h"
static uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_ms);
}
static int time_base_convert(int64_t timestamp_ms, int sampling_rate){
    return timestamp_ms * (sampling_rate / 1000);
} 
void media_write_callback(int program_number, int stream_pid, int stream_type, uint8_t *data, int data_len, void *arg){
    TsMuxerClient *client = (TsMuxerClient*)arg;
    switch (stream_type){
        case STREAM_TYPE_AUDIO_AAC:
            // printf("STREAM_TYPE_AUDIO_AAC\n");
            break;
        case STREAM_TYPE_AUDIO_MPEG1:
            // printf("STREAM_TYPE_AUDIO_MPEG1\n");
            break;
        case STREAM_TYPE_AUDIO_MP3:
            // printf("STREAM_TYPE_AUDIO_MP3\n");
            break;
        case STREAM_TYPE_AUDIO_AAC_LATM:
            // printf("STREAM_TYPE_AUDIO_AAC_LATM\n");
            break;
        case STREAM_TYPE_AUDIO_G711A:
            // printf("STREAM_TYPE_AUDIO_G711A\n");
            break;
        case STREAM_TYPE_AUDIO_G711U:
            // printf("STREAM_TYPE_AUDIO_G711U\n");
            break;
        case STREAM_TYPE_VIDEO_H264:
            // printf("STREAM_TYPE_VIDEO_H264\n");
            break;
        case STREAM_TYPE_VIDEO_HEVC:
            // printf("STREAM_TYPE_VIDEO_HEVC\n");
            break;
        default:
            // printf("PSI SI\n");
            // if(client->type_ == TSMode::TS_FILE && !client->file_rewrit_PSI_flag_ && client->stream_pid_video_ > 0 && client->stream_pid_audio_ > 0){
            //     client->file_rewrit_PSI_flag_ = true;
            // }
            break;
    }
    if(client->type_ == TSMode::TS_FILE){
        if(client->ts_fd_){
            fwrite(data, 1, data_len, client->ts_fd_);
        }
    }
    else if(client->type_ == TSMode::TS_SRT){

    }
    return;
}

TsMuxerClient::TsMuxerClient(std::string url, TSMode type){
    url_ = url;
    type_ = type;
    if(type == TSMode::TS_FILE){
        log_debug("ts flie");
        ts_fd_ = fopen(url_.c_str(), "wb");
    }
    else if(type == TSMode::TS_SRT){
        log_debug("srt stream");
        ConnectServer();
        th_reconnect_ = std::thread(&TsMuxerClient::ReconnectThread, this);
    }
    OpenTsHandle();

    th_video_ = std::thread(&TsMuxerClient::VideoStreamThread, this);
    th_audio_ = std::thread(&TsMuxerClient::AudioStreamThread, this);
}
int TsMuxerClient::ConnectServer(){
    connect_stat_ = true;
    return 0;
}
void TsMuxerClient::CloseConnect(){

}
int TsMuxerClient::OpenTsHandle(){
    context_muxer_ = create_ts_context();
    if(!context_muxer_){
        log_error("create_ts_context error");
        exit(0);
    }
    mpeg2_ts_set_write_callback(context_muxer_, media_write_callback, this);
    if(mpeg2_ts_add_program(context_muxer_, program_number_, nullptr, 0) < 0){
        log_error("mpeg2_ts_add_program error");
        exit(0);
    }
    return 0;
}
void TsMuxerClient::CloseTsHandle(){
    if(!context_muxer_){
        return;
    }
    if(mpeg2_ts_remove_program(context_muxer_, program_number_) < 0){
        log_error("mpeg2_ts_remove_program error\n");
        exit(0);
    }
    destroy_ts_context(context_muxer_);
}
TsMuxerClient::~TsMuxerClient(){
    abort_ = true;
    video_cond_.notify_all();
    audio_cond_.notify_all();
    th_video_.join();
    th_audio_.join();
    CloseTsHandle();
    if(type_ == TSMode::TS_FILE){
        if(ts_fd_){
            fclose(ts_fd_);
        }
    }
    else{
        th_reconnect_.join();
        CloseConnect();
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
    log_debug("~TsMuxerClient");
}
// h264/h265
void TsMuxerClient::SetVideoInfo(enum VideoType type){
    video_type_ = type;
    if(video_type_ == VideoType::VIDEO_H264){
        stream_type_video_ = STREAM_TYPE_VIDEO_H264;
    }
    else if(video_type_ == VideoType::VIDEO_H265){
        stream_type_video_ = STREAM_TYPE_VIDEO_H265;
    }
    else{
        log_error("video type error");
        return;
    }
}
void TsMuxerClient::InputVideoData(uint8_t *data, int data_len, int64_t timestamp){
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
// aac g711a g711u
void TsMuxerClient::SetAudioInfo(enum AudioType type){
    audio_type_ = type;
    if(audio_type_ == AudioType::AUDIO_AAC){
        stream_type_audio_ = STREAM_TYPE_AUDIO_AAC;
    }
    else if(audio_type_ == AudioType::AUDIO_PCMA){
        stream_type_audio_ = STREAM_TYPE_AUDIO_G711A;
    }
    else{
        log_error("audio type error");
        return;
    }
}
void TsMuxerClient::InputAudioData(uint8_t *data, int data_len, int64_t timestamp){
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
void TsMuxerClient::VideoStreamThread(){
    int ret = 0;
    bool send_parameters_flag = false;
    int64_t last_pts = -1;
    while (!abort_) {
        std::unique_lock<std::mutex> unique(video_mtx_);
        if (!video_list_.empty()) {
            if(stream_pid_video_ < 0){
                stream_pid_video_ = mpeg2_ts_add_program_stream(context_muxer_, program_number_, stream_type_video_, nullptr, 0);
                if(stream_pid_video_ < 0){
                    log_error("mpeg2_ts_add_program_stream error");
                    exit(0);
                }
            }
            MediaData packet = video_list_.front();
            video_list_.pop_front();
            unique.unlock();
            if(last_pts == -1){
                last_pts = packet.pts;
            }
            else if(last_pts == packet.pts){
                packet.pts = last_pts + 1;
            }
            last_pts = packet.pts;
            std::unique_lock<std::mutex> unique_flv(ts_mtx_);
            if(mpeg2_ts_packet_muxer(context_muxer_, stream_pid_video_, (uint8_t*)packet.data, packet.data_len, stream_type_video_, time_base_convert(packet.pts, 90000), time_base_convert(packet.dts, 90000)) < 0){
                log_error("mpeg2_ts_packet_muxer error\n");
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
void TsMuxerClient::AudioStreamThread(){
    int ret = 0;
    auto last_config_time = std::chrono::steady_clock::now();
    int64_t pts = 0;
    int64_t last_pts = -1;
    int64_t dts = 0;
    while (!abort_) {
        std::unique_lock<std::mutex> unique(audio_mtx_);
        if (!audio_list_.empty()) {
            if(stream_pid_audio_ < 0){
                stream_pid_audio_ = mpeg2_ts_add_program_stream(context_muxer_, program_number_, stream_type_audio_, nullptr, 0);
                if(stream_pid_audio_ < 0){
                    log_error("mpeg2_ts_add_program_stream error");
                    exit(0);
                }
            }
            if(stream_pid_video_ < 0){
                stream_pid_video_ = mpeg2_ts_add_program_stream(context_muxer_, program_number_, STREAM_TYPE_VIDEO_H264, nullptr, 0);
                if(stream_pid_video_ < 0){
                    log_error("mpeg2_ts_add_program_stream error");
                    exit(0);
                }
            }
            MediaData packet = audio_list_.front();
            audio_list_.pop_front();
            unique.unlock();
            if(stream_type_audio_ == STREAM_TYPE_AUDIO_AAC){
                struct AdtsHeader res;
                ret = ParseAdtsHeader(packet.data, &res);
                if(ret < 0){
                    log_error("ParseAdtsHeader error");
                    free(packet.data);
                    continue;
                }
                if(last_pts == -1){
                    last_pts = packet.pts;
                }
                else if(last_pts == packet.pts){
                    packet.pts = last_pts + ((1024 * 1000) / GetSampleRate(res.samplingFreqIndex));
                }
                pts = dts = time_base_convert(packet.pts, 90000);
            }
            else if(stream_type_audio_ == STREAM_TYPE_AUDIO_G711A || stream_type_audio_ == STREAM_TYPE_AUDIO_G711U){
                if(last_pts == packet.pts){
                    packet.pts = last_pts + ((160 * 1000) / 8000);
                }
                pts = dts = time_base_convert(packet.pts, 90000);
            }
            else{
                free(packet.data);
                continue;
            }
            last_pts = packet.pts;
            std::unique_lock<std::mutex> unique_flv(ts_mtx_);
            if(mpeg2_ts_packet_muxer(context_muxer_, stream_pid_audio_, (uint8_t*)packet.data, packet.data_len, stream_type_audio_, pts, dts) < 0){
                log_error("mpeg2_ts_packet_muxer error\n");
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
void TsMuxerClient::ReconnectThread(){
    int ret = 0;
    while (!abort_) {
        if(connect_stat_ == false){
            log_debug("{} reconnecting ...", url_);
            video_ready_ = audio_ready_ = false;
            CloseConnect();
            ConnectServer();
            CloseTsHandle();
            OpenTsHandle();
            std::unique_lock<std::mutex> unique_flv(ts_mtx_);
            if(connect_stat_){
                log_debug("{} connection successful", url_);
            }
            else{
                log_debug("{} connection failed", url_);
            }
            
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    log_debug("ReconnectThread Finished");
}