#include "TsDemuxerClient.h"
extern "C" {
    #include "h264-sps.h"
    #include "h265-sps.h"
}
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
// H264/H265 of MPEG2 will carry AUD NALU
void video_read_callback(int program_number, int stream_pid, int type, int64_t pts, int64_t dts, uint8_t *data, int data_len, void *arg){
    // printf("program_number:%d stream_pid:%d\n",program_number, stream_pid);
    TsDemuxerClient *client = (TsDemuxerClient*)arg;
    if(client->save_program_number_ == -1){
        client->save_program_number_ = program_number;
    }
    else if(client->save_program_number_ != program_number){
        return;
    }
    switch (type){
        case STREAM_TYPE_VIDEO_H264:
            // printf("STREAM_TYPE_VIDEO_H264\n");
            break;
        case STREAM_TYPE_VIDEO_HEVC:
            // printf("STREAM_TYPE_VIDEO_HEVC\n");
            break;
        default:
            return;
            break;
    }
    int64_t timestamp = pts;
    uint8_t *data_nalus = data;
    int len_nalus = data_len;

    uint8_t *p_video = nullptr;
    uint32_t nal_len;
    uint8_t *buf_sffset = data_nalus;
    uint8_t prefix_len = 0;
    uint8_t *video_data = data_nalus;
    uint32_t video_len = len_nalus;
    bool aud_flag = false;
    p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    while (p_video != nullptr && !client->abort_) {
        prefix_len = prefix_len + 1;
        uint8_t *data_nalu = p_video;
        int data_nalu_len = nal_len;
        int nalu_type;
        int start_code = 0;
        if (data_nalu[0] == 0 && data_nalu[1] == 0 && data_nalu[2] == 1) {
            start_code = 3;
        } else if (data_nalu[0] == 0 && data_nalu[1] == 0 && data_nalu[2] == 0 && data_nalu[3] == 1) {
            start_code = 4;
        }
        if (type == STREAM_TYPE_VIDEO_H264) {
            nalu_type = data_nalu[start_code] & 0x1f;
            client->video_type_ = VideoType::VIDEO_H264;
            if(nalu_type == 7){
                struct h264_sps_t sps;
                h264_sps_parse(data_nalu + start_code, data_nalu_len - start_code, &sps);
                int x, y;
                h264_display_rect(&sps, &x, &y, &client->width_, &client->height_);
            }

        } else if (type == STREAM_TYPE_VIDEO_HEVC) {
            nalu_type = (data_nalu[start_code] >> 1) & 0x3f;
            client->video_type_ = VideoType::VIDEO_H265;
            if(nalu_type == 33){
                struct h265_sps_t sps;
                h265_sps_parse(data_nalu + start_code, data_nalu_len - start_code, &sps);
                int x, y;
                h265_display_rect(&sps, &x, &y, &client->width_, &client->height_);
            }
        }
        // probe fps
        if(nalu_type == 9/*h264 AUD*/ || nalu_type == 35/*h265 AUD*/ && 
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
                    client->fps_ = 90000 / (client->interval_sum_ / client->probe_cnt_);
                }
            }
        }
        if(nalu_type == 9/*h264 AUD*/ || nalu_type == 35/*h265 AUD*/){
            aud_flag = true;
        }
        // if(!(nalu_type == 6 || nalu_type == 7 || nalu_type == 8 || nalu_type == 9/*AUD*/ || nalu_type == 32 || nalu_type == 33 || nalu_type == 34 || nalu_type == 35/*AUD*/) && 
        //     (client->probe_cnt_ < client->probe_frame_num_) && (client->fps_ < 0)){
        //     if(client->last_timestamp_ == -1){
        //         client->last_timestamp_ = timestamp;
        //     }
        //     else{
        //         int interval = timestamp - client->last_timestamp_;
        //         client->interval_sum_ += interval;
        //         client->last_timestamp_ = timestamp;
        //         client->probe_cnt_++;
        //         if(client->probe_cnt_ == client->probe_frame_num_){
        //             client->fps_ = 90000 / (client->interval_sum_ / client->probe_cnt_);
        //         }
        //     }
        // }
        if(!client->probe_over_flag_ && client->width_ > 0 && client->height_ > 0 && client->fps_ > 0){
            client->probe_over_flag_ = true;
        }
        if(client->probe_over_flag_ && !client->video_ready_ && (nalu_type == 7/*h264 sps*/ || nalu_type == 32/*h265 vps*/)){
            client->video_ready_ = true;
        }
        
    NEXT:
        p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    }
    if(!client->video_ready_ || !client->media_ready_){
        return;
    }
    VideoData video_data_packet;
    video_data_packet.data = (unsigned char *)data;
    video_data_packet.data_len = data_len;
    video_data_packet.pts = 0;
    video_data_packet.dts = 0;
    if (client->data_listner_) {
        client->data_listner_->OnVideoData(video_data_packet);
    }        
    if(client->type_ == TSMode::TS_FILE){   
        timestamp = pts / (90000 / 1000);
        if(client->start_timestamp_video_ == -1 || timestamp < client->last_timestamp_video_){
            client->start_timestamp_video_ = timestamp;
        }
        client->last_timestamp_video_ = timestamp;
        timestamp -= client->start_timestamp_video_;  
        uint64_t now = GetCurrentTimeMs();
        uint64_t expected = 0;
        client->start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
        int64_t pts_now = now - client->start_timestamp_.load(std::memory_order_relaxed);
        if(timestamp > 0 && timestamp > pts_now && aud_flag){
            std::this_thread::sleep_for(std::chrono::milliseconds(timestamp - pts_now));
        }
    }
    
    
    return; 
}
void audio_read_callback(int program_number, int stream_pid, int type, int64_t pts, int64_t dts, uint8_t *data, int data_len, void *arg){
    // printf("program_number:%d stream_pid:%d\n",program_number, stream_pid);
    TsDemuxerClient *client = (TsDemuxerClient*)arg;
    if(!client->media_ready_){
        return;
    }
    if(client->save_program_number_ == -1){
        client->save_program_number_ = program_number;
    }
    else if(client->save_program_number_ != program_number){
        return;
    }
    switch (type){
        case STREAM_TYPE_AUDIO_AAC:
            // printf("STREAM_TYPE_AUDIO_AAC\n");
            client->audio_type_ = AudioType::AUDIO_AAC;
            break;
        case STREAM_TYPE_AUDIO_MPEG1:
        case STREAM_TYPE_AUDIO_MP3:
        case STREAM_TYPE_AUDIO_AAC_LATM:
        case STREAM_TYPE_AUDIO_G711A:
        case STREAM_TYPE_AUDIO_G711U:
            return;
        default:
            return;
            break;
    }
    int64_t timestamp = pts;
    timestamp = pts / (90000 / 1000);
    if(client->start_timestamp_audio_ == -1 || timestamp < client->last_timestamp_audio_){
        client->start_timestamp_audio_ = timestamp;
    }
    client->last_timestamp_audio_ = timestamp;
    timestamp -= client->start_timestamp_audio_;
    int remain_bytes = data_len;
    int aac_frames_cnt = 0;
    uint8_t *data_aac = data;
    while(remain_bytes > 0){
        struct AdtsHeader res;
        int ret = ParseAdtsHeader(data_aac, &res);
        if(ret < 0){
            log_error("ParseAdtsHeader error");
            return;
        }
        AudioData audio_data_packet;
        audio_data_packet.data = (unsigned char *)data_aac;
        audio_data_packet.data_len = res.aacFrameLength;
        audio_data_packet.channels = res.channelCfg;
        audio_data_packet.profile = res.profile;
        audio_data_packet.samplerate = GetSampleRate(res.samplingFreqIndex);
        if(client->data_listner_){
            client->data_listner_->OnAudioData(audio_data_packet);
        }
        remain_bytes -= res.aacFrameLength;
        
        if(client->type_ == TSMode::TS_FILE){     
            uint64_t now = GetCurrentTimeMs();
            uint64_t expected = 0;
            client->start_timestamp_.compare_exchange_strong(expected, now, std::memory_order_relaxed);
            int64_t pts_now = now - client->start_timestamp_.load(std::memory_order_relaxed);
            int64_t now_frame_timestamp = timestamp + aac_frames_cnt * ( (1024 * 1000.0) / (audio_data_packet.samplerate) );
            if(now_frame_timestamp > 0 && now_frame_timestamp > pts_now){
                std::this_thread::sleep_for(std::chrono::milliseconds(now_frame_timestamp - pts_now));
            }
        }
        aac_frames_cnt++;
        data_aac += res.aacFrameLength;
    }
    return;
}
TsDemuxerClient::TsDemuxerClient(std::string url, TSMode type){
    type_ = type;
    url_ = url;
    if(type == TSMode::TS_FILE){
        log_debug("ts file");
    }
    else if(type == TSMode::TS_SRT){
        log_debug("srt stream");
    }
    OpenTsHandle();
    if(type == TSMode::TS_FILE){
        ts_fd_ = fopen(url_.c_str(), "rb");
    }
    else if(type == TSMode::TS_SRT){
        ConnectServer();
        th_reconnect_ = std::thread(&TsDemuxerClient::ReconnectThread, this);
    }
    th_ts_handle_ = std::thread(&TsDemuxerClient::TsStreamReadThread, this);
}
int TsDemuxerClient::OpenTsHandle(){
    context_demuxer_ = create_ts_context();
    if(!context_demuxer_){
        return -1;
    }
    mpeg2_ts_set_read_callback(context_demuxer_, video_read_callback, audio_read_callback, this);
    return 0;
}
void TsDemuxerClient::CloseTsHandle(){
    if(context_demuxer_){
        destroy_ts_context(context_demuxer_);
        context_demuxer_ = nullptr;
    }
}
TsDemuxerClient::~TsDemuxerClient(){
    abort_ = true;
    th_ts_handle_.join();
    if(type_ == TSMode::TS_FILE){
        if(ts_fd_){
            fclose(ts_fd_);
        }
    }
    else if(type_ == TSMode::TS_SRT){
        th_reconnect_.join();
        CloseConnect();
    }
    CloseTsHandle();
    log_debug("~TsDemuxerClient");
}
int TsDemuxerClient::ConnectServer(){
    
    connect_stat_ = true;
    return 0;
}
void TsDemuxerClient::CloseConnect(){
}
void TsDemuxerClient::TsStreamReadThread(){
    while(!abort_){
        if(type_ == TSMode::TS_FILE && !media_over_){
            int ts_packet_length = probe_ts_packet_length(ts_fd_);
            log_debug("ts_packet_length: {}", ts_packet_length);
            if(ts_packet_length < 0){
                exit(0);
            }
            unsigned char *buffer = (unsigned char *)malloc(ts_packet_length + 1);
            memset(buffer, 0, ts_packet_length + 1);
            int ret;
            // probe width height fps
            while(fread(buffer, 1, ts_packet_length, ts_fd_) == ts_packet_length && !probe_over_flag_){
                ret = mpeg2_ts_packet_demuxer(context_demuxer_, buffer, ts_packet_length);
                if(ret < 0){
                    log_error("mpeg2_ts_packet_demuxer error");
                    continue;
                }
                switch (context_demuxer_->ts_header.pid){
                    case PID_PAT:
                        // printf("PID_PAT\n");
                        // dump_section_header(context->section_header);
                        // dump_program(context->pat);
                        break;
                    case PID_CAT:
                        // printf("PID_CAT\n");
                        break;
                    case PID_SDT:
                        // printf("SDT、BAT、ST\n");
                        // dump_section_header(context->section_header);
                        // for(int i = 0; i < context->sdt.sdt_info_array_num; i++){
                        //     printf("%s\n", context->sdt.sdt_info_array[i].descriptor);
                        // }
                        break;
                    default:
                        break;
                }
                // if(context->ts_header.PCR){
                //     printf("PCR:%" PRIu64 "\n", context->ts_header.PCR);
                // }
                // dump_ts_header(context->ts_header);
                // dump_pmt_array(context->pmt_array, context->pmt_array_num);
                memset(buffer, 0, ts_packet_length + 1);
            }
            if(fps_ < 0){ // The media duration is too short, failing to meet the detection frame count requirement
                fps_ = 90000 / (interval_sum_ / probe_cnt_);
            }
            media_ready_ = true;
            start_timestamp_audio_ = -1;
            start_timestamp_video_ = -1;
            // demuxer
            CloseTsHandle();
            OpenTsHandle();
            fseek(ts_fd_, 0, SEEK_SET);
            while(!abort_ && fread(buffer, 1, ts_packet_length, ts_fd_) == ts_packet_length){
                ret = mpeg2_ts_packet_demuxer(context_demuxer_, buffer, ts_packet_length);
                if(ret < 0){
                    log_error("mpeg2_ts_packet_demuxer error");
                    continue;
                }
                memset(buffer, 0, ts_packet_length + 1);
            }
            free(buffer);
            media_over_ = true;
            if(colse_cb_ != nullptr){
                colse_cb_();
            }
        }
        else if(type_ == TSMode::TS_SRT && connect_stat_){
            media_ready_ = true;
        } 
        else{
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    log_debug("TsStreamReadThread Finished");
}
void TsDemuxerClient::ReconnectThread(){
    int ret = 0;
    while (!abort_) {
        if(connect_stat_ == false){
            log_debug("{} reconnecting ...", url_);
            video_ready_ = false;
            CloseConnect();
            ConnectServer();
            CloseTsHandle();
            OpenTsHandle();
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
