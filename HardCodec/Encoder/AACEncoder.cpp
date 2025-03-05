#include "AACEncoder.h"
static const uint64 NANO_SECOND = UINT64_C(1000000000);
AACEncoder::AACEncoder()
{
    encode_swr_ctx_ = NULL;
    c_ctx_ = NULL;
    codec_ = NULL;
    time_inited_ = 0;
    abort_ = false;
    // av_init_packet(&pkt_enc_);
    memset(&pkt_enc_, 0, sizeof(pkt_enc_));
    encode_id_ = std::thread(AACEncoder::AACEncThread, this);
    scale_id_ = std::thread(AACEncoder::AACScaleThread, this);
}
AACEncoder::~AACEncoder()
{
    abort_ = true;
    encode_id_.join();
    scale_id_.join();
    if (encode_swr_ctx_) {
        swr_free(&encode_swr_ctx_);
        encode_swr_ctx_ = NULL;
    }
    if (c_ctx_ != NULL) {
        avcodec_close(c_ctx_);
        avcodec_free_context(&c_ctx_);
        c_ctx_ = NULL;
    }
    for (std::list<AACPCMNode *>::iterator it = pcm_frames_.begin(); it != pcm_frames_.end(); ++it) {
        AACPCMNode *node = *it;
        delete node;
    }
    pcm_frames_.clear();
    for (std::list<AVFrame *>::iterator it = dec_frames_.begin(); it != dec_frames_.end(); ++it) {
        AVFrame *frame = *it;
        // if(frame->data[0]){
        //     av_freep(&frame->data[0]);
        // }
        av_frame_free(&frame);
    }
    dec_frames_.clear();
    log_debug("~AACEncoder");
}
void AACEncoder::SetCallback(EncDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}
int AACEncoder::Init(enum AVSampleFormat fmt, int channels, int ratio, int nb_samples)
{
    src_sample_fmt_ = fmt;
    src_nb_channels_ = channels;
    src_ratio_ = ratio;

    dst_sample_fmt_ = AV_SAMPLE_FMT_S16;
    dst_nb_channels_ = 2;
    dst_ratio_ = 44100;
    if (!encode_swr_ctx_) {
        encode_swr_ctx_ = swr_alloc_set_opts(NULL, av_get_default_channel_layout(dst_nb_channels_), dst_sample_fmt_, dst_ratio_,
                                             av_get_default_channel_layout(src_nb_channels_), src_sample_fmt_, src_ratio_, 0, NULL);
        int ret = swr_init(encode_swr_ctx_);
        if (ret < 0) {
            swr_free(&encode_swr_ctx_);
            log_error("encode_swr_ctx_ alloc & set error");
            exit(1);
        }
        src_nb_samples_ = nb_samples;
        dst_nb_samples_ = av_rescale_rnd(src_nb_samples_, dst_ratio_, src_ratio_, AV_ROUND_UP); // 1024
    }
    if (!codec_) {
        // codec_ = avcodec_find_encoder(AV_CODEC_ID_AAC);//libfdk_aac和aac的参数不一样
        codec_ = avcodec_find_encoder_by_name("libfdk_aac");
        if (!codec_) {
            log_error("EnCodec not found");
            exit(1);
        }
        c_ctx_ = avcodec_alloc_context3(codec_);
        c_ctx_->sample_fmt = dst_sample_fmt_; // fdk_aac需要16位的音频输

        c_ctx_->channel_layout = AV_CH_LAYOUT_STEREO; // 输入音频的CHANNEL LAYOUT
        c_ctx_->channels = dst_nb_channels_;          // 输入音频的声道数
        c_ctx_->sample_rate = dst_ratio_;             // 输入音频的采样率
        c_ctx_->bit_rate = 0;                         // AAC : 128K   AAV_HE: 64K  AAC_HE_V2: 32K. bit_rate为0时才会查找profile属性值
        // FF_PROFILE_AAC_LOW == 1 FF_PROFILE_AAC_HE == 4 FF_PROFILE_AAC_HE_V2 == 28
        if(src_nb_samples_ == 2048){
            c_ctx_->profile = FF_PROFILE_AAC_HE;
        }
        else{
            c_ctx_->profile = FF_PROFILE_AAC_LOW; // 1024 
        }
        if (avcodec_open2(c_ctx_, codec_, NULL) < 0) {
            log_error("audio_encode_init error");
            exit(1);
        }
        log_info("audio_encode_init ok");
    }
    return 0;
}
void AACEncoder::GetAudioCon(int &channels, int &sample_rate, int &profile){
    if(c_ctx_->profile == FF_PROFILE_AAC_LOW){
        channels = 2;
        sample_rate = 44100;
        profile = 1;
    }
    else if(c_ctx_->profile == FF_PROFILE_AAC_HE){
        channels = 2;
        sample_rate = 44100 / 2;
        profile = 2;
    }
    
    return;
}
int AACEncoder::AddPCMFrame(unsigned char *data, int data_len)
{
    std::unique_lock<std::mutex> guard(pcm_mutex_);
#if 0
    // 丢帧处理
    while (pcm_frames_.size() > 5) {
        AACPCMNode *node = pcm_frames_.front();
        delete node;
        pcm_frames_.pop_front();
    }
#endif
    AACPCMNode *pcm_data = new AACPCMNode(data, data_len);
    pcm_frames_.push_back(pcm_data);
    guard.unlock();
    pcm_cond_.notify_one();

    if (!time_inited_) {
        time_inited_ = 1;
        time_now_ = std::chrono::steady_clock::now();
        time_pre_ = time_now_;
        now_frames_ = pre_frames_ = 0;
    } else {
        time_now_ = std::chrono::steady_clock::now();
        long tmp_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_ - time_pre_).count();
        if (tmp_time > 1000) {
            log_debug(" aac output frame rate {} ", (now_frames_ - pre_frames_ + 1) * 1000 / tmp_time);
            time_pre_ = time_now_;
            pre_frames_ = now_frames_;
        }
        now_frames_++;
    }
    return 1;
}
void *AACEncoder::AACScaleThread(void *arg)
{
    AACEncoder *self = (AACEncoder *)arg;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->pcm_mutex_);
        if (!self->pcm_frames_.empty()) {
            AACPCMNode *pcm_node = self->pcm_frames_.front();
            self->pcm_frames_.pop_front();
            guard.unlock();
#if 1
            /**
             * FFmpeg真正进行重采样的函数是swr_convert。它的返回值就是重采样输出的点数。
             * 使用FFmpeg进行重采样时内部是有缓存的，而内部缓存了多少个采样点，可以用函数swr_get_delay获取。
             * 也就是说调用函数swr_convert时你传递进去的第三个参数表示你希望输出的采样点数，
             * 但是函数swr_convert的返回值才是真正输出的采样点数，这个返回值一定是小于或等于你希望输出的采样点数。
             */
            int64_t delay = swr_get_delay(self->encode_swr_ctx_, self->src_ratio_);
            int64_t real_dst_nb_samples = av_rescale_rnd(delay + self->src_nb_samples_, self->dst_ratio_, self->src_ratio_, AV_ROUND_UP);
            if (real_dst_nb_samples > self->dst_nb_samples_) {
                log_debug("change dst_nb_samples_");
                self->dst_nb_samples_ = real_dst_nb_samples;
            }
#endif
            AVFrame *frame_enc = av_frame_alloc();
            frame_enc->nb_samples = self->dst_nb_samples_;
            frame_enc->format = self->dst_sample_fmt_;
            frame_enc->channels = self->dst_nb_channels_;
            frame_enc->channel_layout = av_get_default_channel_layout(self->dst_nb_channels_);
            av_frame_get_buffer(frame_enc, 1);
            int ret = swr_convert(self->encode_swr_ctx_, frame_enc->data, frame_enc->nb_samples, (const uint8_t **)&pcm_node->pcm_data, self->src_nb_samples_);

            std::unique_lock<std::mutex> guard(self->frame_mutex_);
            self->dec_frames_.push_back(frame_enc);
            guard.unlock();
            self->frame_cond_.notify_one();
            delete pcm_node;

        } else {
            auto now = std::chrono::system_clock::now();
            self->pcm_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    log_info("AACScaleThread exit");
    return NULL;
}
void *AACEncoder::AACEncThread(void *arg)
{
    AACEncoder *self = (AACEncoder *)arg;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->frame_mutex_);
        if (!self->dec_frames_.empty()) {
            AVFrame *frame = self->dec_frames_.front();
            self->dec_frames_.pop_front();
            guard.unlock();

            int ret;
            ret = avcodec_send_frame(self->c_ctx_, frame);
            if (ret < 0) {
                log_warn("Error sending the frame to the encoder\n");
                // uint8_t* p=frame->data[0];
                // if(p){
                //     av_freep(&p);
                // }

                av_frame_free(&frame);
                continue;
            }
            while (ret >= 0) {
                ret = avcodec_receive_packet(self->c_ctx_, &self->pkt_enc_);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0) {
                    av_frame_free(&frame);
                    av_packet_unref(&self->pkt_enc_);
                    continue;
                }
                // 解码后的数据已经带了adts
                if (self->callback_) {
                    self->callback_->OnAudioEncData(self->pkt_enc_.data, self->pkt_enc_.size);
                }
                av_packet_unref(&self->pkt_enc_);
            }
            // uint8_t* p=frame->data[0];
            // if(p){
            //     av_freep(&p);
            // }
            av_frame_free(&frame);

        } else {
            auto now = std::chrono::system_clock::now();
            self->frame_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    // 清空缓冲区
    int ret = avcodec_send_frame(self->c_ctx_, NULL);
    while (ret >= 0) {
        ret = avcodec_receive_packet(self->c_ctx_, &self->pkt_enc_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0) {
            av_packet_unref(&self->pkt_enc_);
            continue;
        }
        if (self->callback_) {
            self->callback_->OnAudioEncData(self->pkt_enc_.data, self->pkt_enc_.size);
        }
        av_packet_unref(&self->pkt_enc_);
    }
    log_info("AACEncThread exit");
    return NULL;
}