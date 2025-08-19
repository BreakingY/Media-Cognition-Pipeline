#include "MediaWrapper.h"
// #define MP4MUXER

#ifdef MP4MUXER
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
/**
 * 编码后音视频数据
 */
int MiedaWrapper::WriteVideo2File(uint8_t *data_nalus, int len_nalus)
{
    // 这里不使用编码器传过来的时间戳，根据当前时间重新生成
    time_now_ = std::chrono::steady_clock::now();
    nframe_counter_++;
    if (nframe_counter_ == 1) {
        time_pre_ = time_now_;
        time_ts_accum_ = 0;
    }
    uint64_t duration_t = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_ - time_pre_).count();
    time_ts_accum_ += duration_t;
    uint64_t pts_t = time_ts_accum_;
    time_pre_ = time_now_;

    uint8_t *p_video = NULL;
    uint32_t nal_len;
    uint8_t *buf_sffset = data_nalus;
    uint8_t prefix_len = 0;
    uint8_t *video_data = data_nalus;
    uint32_t video_len = len_nalus;
    p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    while (p_video != NULL) {
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
            if (!extra_ready_) {
                if (nalu_type == 7) {
                    if (sps_ == NULL || (sps_buffer_len_ < data_len - start_code)) {
                        sps_ = (uint8_t *)realloc(sps_, data_len - start_code);
                        sps_buffer_len_ = data_len - start_code;
                    }
                    memcpy(sps_, data + start_code, data_len - start_code);
                    sps_len_ = data_len - start_code;

                } else if (nalu_type == 8) {
                    if (pps_ == NULL || (pps_buffer_len_ < data_len - start_code)) {
                        pps_ = (uint8_t *)realloc(pps_, data_len - start_code);
                        pps_buffer_len_ = data_len - start_code;
                    }
                    memcpy(pps_, data + start_code, data_len - start_code);
                    pps_len_ = data_len - start_code;
                }
            }

        } else if (video_type_ == VIDEO_H265) {
            nalu_type = (data[start_code] >> 1) & 0x3f;
            if (!extra_ready_) {
                if (nalu_type == 32) {
                    if (vps_ == NULL || (vps_buffer_len_ < data_len - start_code)) {
                        vps_ = (uint8_t *)realloc(vps_, data_len - start_code);
                        vps_buffer_len_ = data_len - start_code;
                    }
                    memcpy(vps_, data + start_code, data_len - start_code);
                    vps_len_ = data_len - start_code;
                } else if (nalu_type == 33) {
                    if (sps_ == NULL || (sps_buffer_len_ < data_len - start_code)) {
                        sps_ = (uint8_t *)realloc(sps_, data_len - start_code);
                        sps_buffer_len_ = data_len - start_code;
                    }
                    memcpy(sps_, data + start_code, data_len - start_code);
                    sps_len_ = data_len - start_code;

                } else if (nalu_type == 34) {
                    if (pps_ == NULL || (pps_buffer_len_ < data_len - start_code)) {
                        pps_ = (uint8_t *)realloc(pps_, data_len - start_code);
                        pps_buffer_len_ = data_len - start_code;
                    }
                    memcpy(pps_, data + start_code, data_len - start_code);
                    pps_len_ = data_len - start_code;
                }
            }
        }
        if (sps_len_ != 0 && pps_len_ != 0) {
            extra_ready_ = true;
        }
        if (!extra_ready_) {
            goto NEXT;
        }
        if (video_stream_ == -1) {
            ExtraData extra;
            extra.vps = vps_;
            extra.vps_len = vps_len_;
            extra.sps = sps_;
            extra.sps_len = sps_len_;
            extra.pps = pps_;
            extra.pps_len = pps_len_;
            mp4_muxer_->AddVideo(90000, video_type_, extra, width_, height_, fps_);
        }
        // 音频
        if( ((rtsp_flag_ == true) && (rtsp_client_proxy_->GetAudioType() != AudioType::AUDIO_NONE))
            || (reader_->GetAudioType() != AudioType::AUDIO_NONE) ){
            if(audio_stream_ == -1){
                int channels;
                int samplerate;
                int profile;
                aac_encoder_->GetAudioCon(channels, samplerate, profile); // 获取AAC编码器输出信息
                mp4_muxer_->AddAudio(channels, samplerate, profile, AUDIO_AAC);
            }
        }
        if (video_stream_ == -1) {
            mp4_muxer_->Open();
            mp4_muxer_->SendHeader();
            audio_stream_ = mp4_muxer_->GetAudioStreamIndex();
            video_stream_ = mp4_muxer_->GetVideoStreamIndex();
        }
        AVRational time_base = mp4_muxer_->fmt_ctx_->streams[mp4_muxer_->video_index_]->time_base;
        AVRational time_base_q = {1, AV_TIME_BASE};                             // 微妙
        int64_t video_pts = av_rescale_q(pts_t * 1000, time_base_q, time_base); // 转换到ffmpeg时间基
        mp4_muxer_->SendPacket(data + start_code, data_len - start_code, video_pts, video_pts, video_stream_);
    NEXT:
        p_video = get_nal(&nal_len, &buf_sffset, video_data, video_len, &prefix_len);
    }

    return 0;
}
int MiedaWrapper::WriteAudio2File(uint8_t *data, int len)
{
    if (audio_stream_ == -1) {
        return 0;
    }
#if 0
    struct AdtsHeader res;
    ParseAdtsHeader((uint8_t*)data, &res);
    log_info("channelCfg:{} rate:{}",res.channelCfg,sampling_frequencies[res.samplingFreqIndex]);
#endif
    time_now_1_ = std::chrono::steady_clock::now();
    nframe_counter_1_++;
    if (nframe_counter_1_ == 1) {
        time_pre_1_ = time_now_1_;
        time_ts_accum_1_ = 0;
    }
    uint64_t duration_t = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_1_ - time_pre_1_).count();
    time_ts_accum_1_ += duration_t;
    uint64_t pts_t = time_ts_accum_1_;
    // 模拟实时流，所以这里面的pts重新生成
    time_pre_1_ = time_now_1_;

    AVRational time_base = mp4_muxer_->fmt_ctx_->streams[mp4_muxer_->audio_index_]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};                             // 微妙
    int64_t audio_pts = av_rescale_q(pts_t * 1000, time_base_q, time_base); // 转换到ffmpeg时间基
    mp4_muxer_->SendPacket(data + 7, len - 7, audio_pts, audio_pts, audio_stream_);
    return 0;
}
#endif
MiedaWrapper::MiedaWrapper(char *input, char *ouput)
{
#ifdef MP4MUXER
    mp4_muxer_ = new Muxer();
    mp4_muxer_->Init(ouput);
#endif
    if( memcmp("rtsp://", input, strlen("rtsp://")) == 0 ){ // rtsp
        rtsp_flag_ = true;
        rtsp_client_proxy_ = new RtspClientProxy(input);
        rtsp_client_proxy_->ProbeVideoFps(); // 必须在SetDataListner之前调用ProbeVideoFps,否则在RtspClientProxy::RtspVideoData调用data_listner_的时候会阻塞
        rtsp_client_proxy_->GetVideoCon(width_, height_, fps_);
        rtsp_client_proxy_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else{ // file
        reader_ = new MediaReader(input);
        reader_->GetVideoCon(width_, height_, fps_);
        reader_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
}
void MiedaWrapper::MediaOverhandle()
{
    over_flag_ = true;
    return;
}
/**
 * 音视频解封装、解码
 */
// with startcode
void MiedaWrapper::OnVideoData(VideoData data)
{
    if(rtsp_flag_ == true){ // rtsp
        video_type_ = rtsp_client_proxy_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
    }
    else{ // file 
        video_type_ = reader_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
    }
    if (!hard_decoder_) {
        log_debug("video_type:{} width:{} height:{} fps_:{}", video_type_ == VIDEO_H264 ? "VIDEO_H264" : "VIDEO_H265", width_, height_, fps_);
        hard_decoder_ = new HardVideoDecoder(video_type_ == VIDEO_H264 ? false : true);
        hard_decoder_->SetFrameFetchCallback(static_cast<DecDataCallListner *>(this));
#if defined(USE_DVPP_MPI) || defined(USE_NVIDIA_X86)
        hard_decoder_->Init(device_id_, width_, height_); // dvpp nvidia
#endif
    }
    // int type;
    // if(video_type_ == VIDEO_H264){
    //     type = data.data[4] & 0x1f;
    // }
    // else{
    //     type = (data.data[4] >> 1) & 0x3f;
    // }
    hard_decoder_->InputVideoData(data.data, data.data_len, 0, 0); // 实时解码，不需要传递pts
    return;
}
// width adts
void MiedaWrapper::OnAudioData(AudioData data)
{
    if(rtsp_flag_ == true){
        audio_type_ = rtsp_client_proxy_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    else{
        audio_type_ = reader_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    if (aac_decoder_ == NULL) {
        log_debug("audio_type:AAC profile:{} samplerate:{} channels:{}", data.profile, data.samplerate, data.channels);
        aac_decoder_ = new AACDecoder();
        aac_decoder_->SetResampleArg(AV_SAMPLE_FMT_S16, 2, 44100); // 重采样输出格式，解码器会把解码后的PCM数据重采样成设定的格式
        aac_decoder_->SetCallback(static_cast<DecDataCallListner *>(this));
    }
    aac_decoder_->InputAACData(data.data, data.data_len); // 实时解码，不需要传递pts
    return;
}

/**
 * 解码后音视频数据
 */
void MiedaWrapper::OnRGBData(cv::Mat frame)
{
    // 拿到解码后的图像就可以根据自己的业务需求进行处理，例如：AI识别、opencv检测、图像渲染等。
    // 之后再把处理后的图像进行编码
    if (!hard_encoder_) {
        hard_encoder_ = new HardVideoEncoder();
#if defined(USE_DVPP_MPI) || defined(USE_NVIDIA_X86)
        hard_encoder_->SetDevice(device_id_); // dvpp nvidia
#endif
        hard_encoder_->Init(frame, fps_);
        hard_encoder_->SetDataCallback(static_cast<EncDataCallListner *>(this));
    }
    hard_encoder_->AddVideoFrame(frame);
    return;
}
// FILE *fp_file = NULL;
// data_len是单通道当本个数 LC-AAC:1024 HE-AAC:2048
void MiedaWrapper::OnPCMData(unsigned char **data, int data_len)
{
    // 拿到解码后的PCM音频根据自己的业务需求进行处理，例如语音识别、语音合成等。
    // 之后再把处理后的音频进行编码
    if (aac_encoder_ == NULL) {
        aac_encoder_ = new AACEncoder();
        // aac编码模块只接受packed模式的pcm数据
        // 和 aac_decoder_->SetResampleArg(AV_SAMPLE_FMT_S16,2,44100)保持一致即可，但如果aac_decoder_->SetResampleArg中指定了AV_SAMPLE_FMT_S16P,这里使用AV_SAMPLE_FMT_S16，数据就要转换成packed模型在送入队列
        aac_encoder_->Init(AV_SAMPLE_FMT_S16, 2 , 44100, data_len); // 输入格式，编码器会把PCM数据重采样成AAC编码器需要的格式然后进行编码
        aac_encoder_->SetCallback(static_cast<EncDataCallListner *>(this));
    }
    
    // 转换成packed在传送给aac编码模块
    enum AVSampleFormat dst_sample_fmt = AV_SAMPLE_FMT_S16;
    int dst_nb_channels = 2;
    int out_spb = av_get_bytes_per_sample(dst_sample_fmt);
    int buf_len = data_len * out_spb * dst_nb_channels;
    if (buffer_pcm_ == NULL || (buffer_pcm_len_ < buf_len)) {
        buffer_pcm_ = (unsigned char *)realloc(buffer_pcm_, buf_len);
        buffer_pcm_len_ = buf_len;
    }
    int pos = 0;
    if (av_sample_fmt_is_planar(dst_sample_fmt)) { // plannar,dst_linesize=data_len*out_spb
        for (int i = 0; i < data_len; i++) {
            for (int c = 0; c < dst_nb_channels; c++)
                memcpy(buffer_pcm_ + pos, data[c] + i * out_spb, out_spb);
            pos += out_spb;
        }
    } else { // packed,dst_linesize=data_len*out_spb*out_channels
        memcpy(buffer_pcm_, data[0], data_len * out_spb * dst_nb_channels);
    }
    aac_encoder_->AddPCMFrame(buffer_pcm_, buf_len);
    // if (fp_file == NULL) {
    //     fp_file = fopen("test.pcm", "wb+");
    // }
    // fwrite(buffer_pcm_, 1, buf_len, fp_file); // ffplay -ar 44100 -ac 2 -f s16le -i test.pcm
    return;
}
static const char *enc_h264_filename = "out.h264";
static FILE *enc_h264_fd = NULL;
void MiedaWrapper::OnVideoEncData(unsigned char *data, int data_len, int64_t pts)
{
    if (enc_h264_fd == NULL) {
        enc_h264_fd = fopen(enc_h264_filename, "wb");
    }
    fwrite(data, 1, data_len, enc_h264_fd);
#ifdef MP4MUXER
    WriteVideo2File(data, data_len);
#endif
    return;
}
static const char *enc_aac_filename = "out.aac";
static FILE *enc_aac_fd = NULL;
void MiedaWrapper::OnAudioEncData(unsigned char *data, int data_len)
{
    if (enc_aac_fd == NULL) {
        enc_aac_fd = fopen(enc_aac_filename, "wb");
    }
    fwrite(data, 1, data_len, enc_aac_fd);
#ifdef MP4MUXER
    WriteAudio2File(data, data_len);
#endif
    return;
}
MiedaWrapper::~MiedaWrapper()
{

    if (reader_) {
        delete reader_;
        reader_ = NULL;
    }
    if(rtsp_client_proxy_){
        delete rtsp_client_proxy_;
        rtsp_client_proxy_ = NULL;
    }
    if (hard_decoder_) {
        delete hard_decoder_;
        hard_decoder_ = NULL;
    }
    if (hard_encoder_) {
        delete hard_encoder_;
        hard_encoder_ = NULL;
    }
    if (aac_decoder_) {
        delete aac_decoder_;
        aac_decoder_ = NULL;
    }
    if (aac_encoder_) {
        delete aac_encoder_;
        aac_encoder_ = NULL;
    }
    if (mp4_muxer_) {
        mp4_muxer_->SendTrailer();
        delete mp4_muxer_;
        mp4_muxer_ = NULL;
    }
    if (vps_) {
        free(vps_);
        vps_ = NULL;
    }
    if (sps_) {
        free(sps_);
        sps_ = NULL;
    }
    if (pps_) {
        free(pps_);
        pps_ = NULL;
    }
    if (buffer_pcm_) {
        free(buffer_pcm_);
        buffer_pcm_ = NULL;
    }
    log_debug("~MiedaWrapper");
}