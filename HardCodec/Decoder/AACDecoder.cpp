#include "AACDecoder.h"
static const uint64_t NANO_SECOND = UINT64_C(1000000000);

AACDecoder::AACDecoder()
{

    audio_codec_ = avcodec_find_decoder(AV_CODEC_ID_AAC);
    if (!audio_codec_) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }
    audio_codec_ctx_ = avcodec_alloc_context3(audio_codec_);
    if (!audio_codec_ctx_) {
        fprintf(stderr, "Could not allocate audio codec context\n");
        exit(1);
    }
    int ret = avcodec_open2(audio_codec_ctx_, audio_codec_, NULL);
    if (ret < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }
    // av_init_packet(&packet_);
    memset(&packet_, 0, sizeof(packet_));
    frame_ = NULL;
    swr_ctx_ = NULL;
    aborted_ = false;
    time_inited_ = 0;
    dec_thread_id_ = std::thread(AACDecoder::AACDecodeThread, this);
    sws_thread_id_ = std::thread(AACDecoder::AACScaleThread, this);
}
AACDecoder::~AACDecoder()
{
    aborted_ = true;
    dec_thread_id_.join();
    sws_thread_id_.join();

    for (std::list<AVFrame *>::iterator it = yuv_frames_.begin(); it != yuv_frames_.end(); ++it) {
        AVFrame *frame = *it;
        // uint8_t* p=frame->data[0];
        // av_freep(&p);
        av_frame_free(&frame);
    }
    yuv_frames_.clear();
    for (std::list<AACDataNode *>::iterator it = es_packets_.begin(); it != es_packets_.end(); ++it) {
        AACDataNode *packet = *it;
        delete packet;
    }
    es_packets_.clear();

    if (audio_codec_ctx_) {
        if (audio_codec_ctx_->extradata) {
            av_free(audio_codec_ctx_->extradata);
            audio_codec_ctx_->extradata = NULL;
            audio_codec_ctx_->extradata_size = 0;
        }
        avcodec_close(audio_codec_ctx_);
        if (audio_codec_ctx_->opaque != NULL) {
            free(audio_codec_ctx_->opaque);
            audio_codec_ctx_->opaque = NULL;
        }
        avcodec_free_context(&audio_codec_ctx_);
    }
    if (swr_ctx_) {
        swr_free(&swr_ctx_);
    }
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = NULL;
    }
    av_packet_unref(&packet_);
    log_debug("~AACDecoder");
}

void AACDecoder::InputAACData(unsigned char *data, int data_len)
{

    AACDataNode *node = new AACDataNode();
    node->es_data = (unsigned char *)malloc(data_len);
    memcpy(node->es_data, data, data_len);
    node->es_data_len = data_len;
    std::unique_lock<std::mutex> guard(packet_mutex_);
    es_packets_.push_back(node);
    guard.unlock();
    packet_cond_.notify_one();
    return;
}
void AACDecoder::SetCallback(DecDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}
void AACDecoder::DecodeAudio(AACDataNode *data)
{
    packet_.data = data->es_data;
    packet_.size = data->es_data_len;
    int ret;
    ret = avcodec_send_packet(audio_codec_ctx_, &packet_);
    while (ret >= 0) {
        if (!frame_) {
            frame_ = av_frame_alloc();
        }
        ret = avcodec_receive_frame(audio_codec_ctx_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0) {
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = NULL;
            }
            av_packet_unref(&packet_);
            return;
        }
        src_sample_fmt_ = (enum AVSampleFormat)frame_->format;
        src_nb_channels_ = frame_->channels;
        src_ratio_ = frame_->sample_rate;
        src_nb_samples_ = frame_->nb_samples;

        std::unique_lock<std::mutex> guard(frame_mutex_);
        yuv_frames_.push_back(frame_);
        frame_ = NULL;
        guard.unlock();
        frame_cond_.notify_one();
    }
    av_packet_unref(&packet_);
    return;
}
void *AACDecoder::AACDecodeThread(void *arg)
{
    AACDecoder *self = (AACDecoder *)arg;
    while (!self->aborted_) {
        std::unique_lock<std::mutex> guard(self->packet_mutex_);
        if (!self->es_packets_.empty()) {
            AACDataNode *packet = self->es_packets_.front();
            self->es_packets_.pop_front();
            guard.unlock();
            self->DecodeAudio(packet);

            delete packet;
        } else {
            auto now = std::chrono::system_clock::now();
            self->packet_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    log_info("AACDecodeThread Finished ");
    // 刷新缓冲区
    AACDataNode *node = new AACDataNode();
    node->es_data = NULL;
    node->es_data_len = 0;
    self->DecodeAudio(node);
    delete node;
    return NULL;
}
void AACDecoder::SetResampleArg(enum AVSampleFormat fmt, int channels, int ratio)
{
    dst_sample_fmt_ = fmt;
    dst_nb_channels_ = channels;
    dst_ratio_ = ratio;
    return;
}

void AACDecoder::ScaleAudio(AVFrame *frame)
{

    if (!swr_ctx_) {
        // 解码后音频重采样
        swr_ctx_ = swr_alloc_set_opts(NULL, av_get_default_channel_layout(dst_nb_channels_), dst_sample_fmt_, dst_ratio_,
                                      av_get_default_channel_layout(src_nb_channels_), src_sample_fmt_, src_ratio_, 0, NULL);
        int ret = swr_init(swr_ctx_);
        if (ret != 0) {
            swr_free(&swr_ctx_);
            log_critical("swr_ctx_ alloc & set error");
            exit(1);
        }
        dst_nb_samples_ = av_rescale_rnd(src_nb_samples_, dst_ratio_, src_ratio_, AV_ROUND_UP);
    }
    /**
     * swr_convert Parameters
     * s	allocated Swr context, with parameters set
     * out	output buffers, 如果是packed模式音频，只需设置第一个
     * out_count	一个通道中可用的输出样本数-非空间大小
     * in	input buffers, 如果是packed模式音频，只需设置第一个
     * in_count	一个通道中可用的输入采样数
     * 返回值是单个通道的采样点个数
     */
#if 1
    /**
     * FFmpeg真正进行重采样的函数是swr_convert。它的返回值就是重采样输出的点数。
     * 使用FFmpeg进行重采样时内部是有缓存的，而内部缓存了多少个采样点，可以用函数swr_get_delay获取。
     * 也就是说调用函数swr_convert时你传递进去的第三个参数表示你希望输出的采样点数，
     * 但是函数swr_convert的返回值才是真正输出的采样点数，这个返回值一定是小于或等于你希望输出的采样点数。
     */
    int64_t delay = swr_get_delay(swr_ctx_, src_ratio_);
    int64_t real_dst_nb_samples = av_rescale_rnd(delay + src_nb_samples_, dst_ratio_, src_ratio_, AV_ROUND_UP);
    if (real_dst_nb_samples > dst_nb_samples_) {
        log_debug("change dst_nb_samples_");
        dst_nb_samples_ = real_dst_nb_samples;
    }
#endif

    AVFrame *frame_dec = av_frame_alloc();
    frame_dec->nb_samples = dst_nb_samples_;
    frame_dec->format = dst_sample_fmt_;
    frame_dec->channels = dst_nb_channels_;
    frame_dec->channel_layout = av_get_default_channel_layout(dst_nb_channels_);
    av_frame_get_buffer(frame_dec, 1);
    int ret = swr_convert(swr_ctx_, frame_dec->data, frame_dec->nb_samples, (const uint8_t **)frame->data, frame->nb_samples);
    if (callback_) {
        now_frames_++;
    
        if (!time_inited_) {
            time_inited_ = 1;
            time_now_ = std::chrono::steady_clock::now();
            time_pre_ = time_now_;
        } else {
            time_now_ = std::chrono::steady_clock::now();
            long tmp_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_ - time_pre_).count();
    
            if (tmp_time > 1000) { // 1s
                int tmp_frame_rate = (now_frames_ - pre_frames_ + 1) * 1000 / tmp_time;
                log_debug("AAC input frame rate {} src_sample_fmt_:{} src_nb_channels_:{} src_ratio_:{} src_nb_samples_:{} ",
                          tmp_frame_rate, av_get_sample_fmt_name(src_sample_fmt_), src_nb_channels_, src_ratio_, src_nb_samples_);
                time_pre_ = time_now_;
                pre_frames_ = now_frames_;
            }
        }
        callback_->OnPCMData(frame_dec->data, ret);
    }
    av_frame_free(&frame_dec);
    // uint8_t* p=frame->data[0];
    // av_freep(&p);
    av_frame_free(&frame);
    return;
}
void *AACDecoder::AACScaleThread(void *arg)
{
    AACDecoder *self = (AACDecoder *)arg;
    while (!self->aborted_) {
        std::unique_lock<std::mutex> guard(self->frame_mutex_);
        if (!self->yuv_frames_.empty()) {
            AVFrame *frame = self->yuv_frames_.front();
            self->yuv_frames_.pop_front();
            guard.unlock();
            self->ScaleAudio(frame);
        } else {
            auto now = std::chrono::system_clock::now();
            self->frame_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    log_info("AACScaleThread Finished ");
    return NULL;
}