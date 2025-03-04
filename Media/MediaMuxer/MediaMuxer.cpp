#include "MediaMuxer.h"
Muxer::Muxer()
{
}

Muxer::~Muxer()
{
    if (audio_index_ != -1 || video_index_ != -1)
        DeInit();
    for (int i = 0; i < vps_number_; i++) {
        free(vps_buf_[i]);
    }
    for (int i = 0; i < sps_number_; i++) {
        free(sps_buf_[i]);
    }
    for (int i = 0; i < pps_number_; i++) {
        free(pps_buf_[i]);
    }
}

int Muxer::Init(const char *url)
{
    int ret = avformat_alloc_output_context2(&fmt_ctx_, NULL, NULL, url);
    if (ret < 0) {
        char errbuf[1024] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf) - 1);
        log_error("avformat_alloc_output_context2 failed:{}", errbuf);
        return -1;
    }
    url_ = url;
    ofmt_ = fmt_ctx_->oformat;
    return 0;
}

void Muxer::DeInit()
{
    if (fmt_ctx_) {
        if (!(ofmt_->flags & AVFMT_NOFILE)) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_close_input(&fmt_ctx_);
    }
    url_ = "";

    aud_stream_ = NULL;
    audio_index_ = -1;

    vid_stream_ = NULL;
    video_index_ = -1;
    return;
}
void Muxer::H264WriteExtra(unsigned char *extra_data, int &extra_data_size)
{

    unsigned char *extra_data_start = extra_data;
    extra_data[0] = 0x01;
    extra_data[1] = sps_buf_[0][1];
    extra_data[2] = sps_buf_[0][2];
    extra_data[3] = sps_buf_[0][3];
    extra_data += 4;
    *extra_data++ = 0xff;
    // sps
    *extra_data++ = 0xe0 | (sps_number_ & 0x1f); // sps 个数
    for (int i = 0; i < sps_number_; i++) {
        // 两个字节表示sps
        extra_data[0] = (sps_len_[i]) >> 8;
        extra_data[1] = sps_len_[i];
        extra_data += 2;
        memcpy(extra_data, sps_buf_[i], sps_len_[i]);
        extra_data += sps_len_[i];
    }

    // pps

    *extra_data++ = pps_number_; // 1个sps
    for (int i = 0; i < pps_number_; i++) {
        extra_data[0] = (pps_len_[i]) >> 8;
        extra_data[1] = pps_len_[i];
        extra_data += 2;
        memcpy(extra_data, pps_buf_[i], pps_len_[i]);
        extra_data += pps_len_[i];
    }

    extra_data_size = extra_data - extra_data_start;
    return;
}
void Muxer::H265WriteExtra(unsigned char *extra_data, int &extra_data_size)
{
    int i = 0;
    unsigned char *buffer = extra_data;
    buffer[i++] = 0x01;

    // general_profile_idc 8bit
    buffer[i++] = 0x00;
    // general_profile_compatibility_flags 32 bit
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;

    // 48 bit NUll nothing deal in rtmp
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;

    // general_level_idc
    buffer[i++] = 0x00;

    // 48 bit NUll nothing deal in rtmp
    buffer[i++] = 0xf0;
    buffer[i++] = 0x00;
    buffer[i++] = 0xfc;
    buffer[i++] = 0xfc;
    buffer[i++] = 0xf8;
    buffer[i++] = 0xf8;

    // bit(16) avgFrameRate;
    buffer[i++] = 0x00;
    buffer[i++] = 0x00;

    /* bit(2) constantFrameRate; */
    /* bit(3) numTemporalLayers; */
    /* bit(1) temporalIdNested; */
    buffer[i++] = 0x03;

    /* unsigned int(8) numOfArrays; 03 */
    buffer[i++] = 3;

    // printf("HEVCDecoderConfigurationRecord data = %s\n", buffer);
    buffer[i++] = 0xa0; // vps 32
    buffer[i++] = (vps_number_ >> 8) & 0xff;
    buffer[i++] = vps_number_ & 0xff;
    for (int j = 0; j < vps_number_; j++) {
        buffer[i++] = (vps_len_[j] >> 8) & 0xff;
        buffer[i++] = (vps_len_[j]) & 0xff;
        memcpy(&buffer[i], vps_buf_[j], vps_len_[j]);
        i += vps_len_[j];
    }

    // sps
    buffer[i++] = 0xa1; // sps 33
    buffer[i++] = (sps_number_ >> 8) & 0xff;
    buffer[i++] = sps_number_ & 0xff;
    for (int j = 0; j < sps_number_; j++) {
        buffer[i++] = (sps_len_[j] >> 8) & 0xff;
        buffer[i++] = sps_len_[j] & 0xff;
        memcpy(&buffer[i], sps_buf_[j], sps_len_[j]);
        i += sps_len_[j];
    }

    // pps
    buffer[i++] = 0xa2; // pps 34
    buffer[i++] = (pps_number_ >> 8) & 0xff;
    buffer[i++] = pps_number_ & 0xff;
    for (int j = 0; j < pps_number_; j++) {
        buffer[i++] = (pps_len_[j] >> 8) & 0xff;
        buffer[i++] = pps_len_[j] & 0xff;
        memcpy(&buffer[i], pps_buf_[j], pps_len_[j]);
        i += pps_len_[j];
    }
    extra_data_size = i;
    return;
}
bool Muxer::ParametersChange(unsigned char *vps, int vps_len, unsigned char *sps, int sps_len, unsigned char *pps, int pps_len)
{
    int idx_vps, idx_sps, idx_pps;
    if (vps != NULL) {
        for (idx_vps = 0; idx_vps < vps_number_; idx_vps++) {
            if (memcmp(vps, vps_buf_[idx_vps], vps_len) == 0) {
                break;
            }
        }
        if (idx_vps == vps_number_) {
            log_warn("vps change");
            return true;
        }
    }
    if (sps != NULL) {
        for (idx_sps = 0; idx_sps < sps_number_; idx_sps++) {
            if (memcmp(sps, sps_buf_[idx_sps], sps_len) == 0) {
                break;
            }
        }
        if (idx_sps == sps_number_) {
            log_warn("sps change");
            return true;
        }
    }
    if (pps != NULL) {
        for (idx_pps = 0; idx_pps < pps_number_; idx_pps++) {
            if (memcmp(pps, pps_buf_[idx_pps], pps_len) == 0) {
                break;
            }
        }
        if (idx_pps == pps_number_) {
            log_warn("pps change");
            return true;
        }
    }
    return false;
}
int Muxer::AddVideo(int time_base, VideoType type, ExtraData &extra, int width, int height, int fps)
{
    if (!fmt_ctx_) {
        log_error("fmt ctx is NULL");
        return -1;
    }
    if (extra.sps_len < 0 || extra.pps_len < 0 || (type == VIDEO_H265 && extra.vps_len < 0)) {
        log_error("extra data error");
        return -1;
    }

    video_type_ = type;
    enum AVCodecID id;
    if (type == VIDEO_H264) {
        id = AV_CODEC_ID_H264;
    } else if (type == VIDEO_H265) {
        id = AV_CODEC_ID_HEVC;
    }
    AVCodec *codec = avcodec_find_encoder(ofmt_->video_codec);
    if (codec == NULL) {
        log_error("No encoder[{}] found\n", avcodec_get_name(ofmt_->video_codec));
        return -1;
    }

    AVStream *st = avformat_new_stream(fmt_ctx_, codec);
    if (!st) {
        log_error("avformat_new_stream failed");
        return -1;
    }
    st->time_base.num = 1;
    st->time_base.den = time_base;
    log_debug("width:{} height:{}", width, height);
    sps_buf_[0] = (uint8_t *)malloc(extra.sps_len);
    memcpy(sps_buf_[0], extra.sps, extra.sps_len);
    sps_len_[0] = extra.sps_len;
    sps_number_++;

    pps_buf_[0] = (uint8_t *)malloc(extra.pps_len);
    memcpy(pps_buf_[0], extra.pps, extra.pps_len);
    pps_len_[0] = extra.pps_len;
    pps_number_++;
    AVCodecParameters *out_codecpar = st->codecpar;
    out_codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    out_codecpar->codec_id = id;
    out_codecpar->codec_tag = 0;
    out_codecpar->format = AV_PIX_FMT_YUV420P;
    out_codecpar->width = width;
    out_codecpar->height = height;
    // st->codec_ctx->codec_tag = 0;
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        // st->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        global_header_ = true;
        out_codecpar->extradata = (uint8_t *)av_malloc(1024);
        out_codecpar->extradata_size = 0;
        memset(out_codecpar->extradata, 0, 1024);
        if (type == VIDEO_H264) {
            H264WriteExtra(out_codecpar->extradata, out_codecpar->extradata_size);
        } else {
            vps_buf_[0] = (uint8_t *)malloc(extra.vps_len);
            memcpy(vps_buf_[0], extra.vps, extra.vps_len);
            vps_len_[0] = extra.vps_len;
            vps_number_++;
            H265WriteExtra(out_codecpar->extradata, out_codecpar->extradata_size);
        }
    }
    av_dump_format(fmt_ctx_, 0, url_.c_str(), 1);
    video_index_ = st->index;
    vid_stream_ = st;

    return 0;
}
static uint8_t *generate_aac_specific_config(int channelConfig, int samplingFrequency, int profile, int *length)
{
    uint8_t *data = NULL;
    *length = 2;
    data = (uint8_t *)malloc(*length);
    if (data == NULL) {
        log_error("malloc error");
        return NULL;
    }
    if (profile < 1 || profile > 4) {
        return NULL;
    }

    if (samplingFrequency != 96000 && samplingFrequency != 88200 &&
        samplingFrequency != 64000 && samplingFrequency != 48000 &&
        samplingFrequency != 44100 && samplingFrequency != 32000 &&
        samplingFrequency != 24000 && samplingFrequency != 22050 &&
        samplingFrequency != 16000 && samplingFrequency != 12000 &&
        samplingFrequency != 11025 && samplingFrequency != 8000) {
        return NULL;
    }

    if (channelConfig < 1 || channelConfig > 7) {
        return NULL;
    }
    int freq_arr[13] = {
        96000, 88200, 64000, 48000, 44100, 32000,
        24000, 22050, 16000, 12000, 11025, 8000, 7350
    };
    int sample_index = 4;
    for(int i = 0; i < 13; i++){
        if(freq_arr[i] == samplingFrequency){
            sample_index = i;
            break;
        }
    }
    data[0] = (profile << 3) | ((sample_index >> 1) & 0x7);
    data[1] = ((sample_index & 0x1) << 7) | (channelConfig << 3);

    return data;
}
void Muxer::AACWriteExtra(int channels, int sample_rate, int profile, AVCodecParameters *params)
{
    int aac_extradata_size = -1;
    uint8_t *aac_extradata = generate_aac_specific_config(channels, sample_rate, profile, &aac_extradata_size);
    if (aac_extradata == NULL) {
        log_error("AACWriteExtra error");
        return;
    }
    // 音频特定配置信息（AudioSpecificConfig）的长度必须大于等于2字节
    if (aac_extradata_size < 2) {
        log_error("AACWriteExtra error");
        return;
    }

    uint8_t profile_dec = (aac_extradata[0] >> 3) & 0x1F;
    uint8_t sampling_frequency_fndex = ((aac_extradata[0] & 0x07) << 1) | ((aac_extradata[1] >> 7) & 0x01);
    log_debug("profile_dec:{} profile:{}", profile_dec, profile);
    log_debug("sampling_frequency_fndex:{},sample_rate:{}", sampling_frequency_fndex, sample_rate);

    params->extradata = (uint8_t *)av_mallocz(aac_extradata_size);
    if (!params->extradata) {
        log_error("malloc error");
        return;
    }
    params->extradata_size = aac_extradata_size;
    memcpy(params->extradata, aac_extradata, aac_extradata_size);
    if (aac_extradata) {
        free(aac_extradata);
    }
    return;
}
int Muxer::AddAudio(int channels, int sample_rate, int profile, AudioType type)
{
    if (!fmt_ctx_) {
        log_error("fmt ctx is NULL");
        return -1;
    }
    log_debug("channels:{} sample_rate:{} profile:{}", channels, sample_rate, profile);
    audio_type_ = type;
    enum AVCodecID id;
    if (type != AUDIO_AAC) {
        log_warn("only support AAC");
        return -1;
    }
    id = AV_CODEC_ID_AAC;
    AVCodec *codec = avcodec_find_encoder(ofmt_->audio_codec);
    if (codec == NULL) {
        log_error("No encoder[{}] found\n", avcodec_get_name(ofmt_->audio_codec));
        return -1;
    }

    AVStream *st = avformat_new_stream(fmt_ctx_, codec);
    if (!st) {
        log_error("avformat_new_stream failed");
        return -1;
    }
    st->time_base.num = 1;
    st->time_base.den = sample_rate;
    AVCodecParameters *out_codecpar = st->codecpar;

    out_codecpar->codec_id = id;
    out_codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    out_codecpar->format = AV_SAMPLE_FMT_FLTP;
    out_codecpar->channels = channels;
    out_codecpar->sample_rate = sample_rate;
    out_codecpar->bit_rate = 0;
    out_codecpar->profile = profile;
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        global_header_ = true;
        // st->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        AACWriteExtra(channels, sample_rate, profile, out_codecpar);
    }

    av_dump_format(fmt_ctx_, 0, url_.c_str(), 1);
    audio_index_ = st->index;
    aud_stream_ = st;
    return 0;
}
int Muxer::SendHeader()
{
    if (!fmt_ctx_) {
        log_error("fmt ctx is NULL");
        return -1;
    }
    int ret = avformat_write_header(fmt_ctx_, NULL);
    if (ret < 0) {
        char errbuf[1024] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf) - 1);
        log_error("avformat_write_header failed:{}", errbuf);
        return -1;
    }
    log_debug("{}:SendHeader ok", url_);
    return 0;
}
void Muxer::RewriteVideoExtraData()
{
    AVCodecParameters *out_codecpar = vid_stream_->codecpar;
    int alloc_size = out_codecpar->extradata_size + 1024;
    out_codecpar->extradata = (uint8_t *)av_realloc(out_codecpar->extradata, alloc_size);
    out_codecpar->extradata_size = 0;
    memset(out_codecpar->extradata, 0, alloc_size);
    if (video_type_ == VIDEO_H264) {
        H264WriteExtra(out_codecpar->extradata, out_codecpar->extradata_size);
    } else if (video_type_ == VIDEO_H265) {
        H265WriteExtra(out_codecpar->extradata, out_codecpar->extradata_size);
    } else {
        log_error("video_type_ error:{}", video_type_);
    }
    return;
}
// 音视频同时写入需要加锁
int Muxer::SendPacket(unsigned char *data, int size, int64_t pts, int64_t dts, int stream_index)
{
    std::lock_guard<std::mutex> guard(mtx_);
    pkt_.stream_index = stream_index;
    pkt_.duration = 0; // 0 if unknown.
    pkt_.pos = -1;     //-1 if unknown
    if (size <= 0) {
        log_warn("packet size:{}", size);
        av_packet_unref(&pkt_);
        return -1;
    }
    int nal_type;
    if (stream_index == video_index_) {

        if (video_type_ == VIDEO_H264) {
            nal_type = data[0] & 0x1f;
            if (nal_type == 7 || nal_type == 5) {
                found_idr_ = true;
            }
            if (global_header_ && (nal_type == 6 || nal_type == 7 || nal_type == 8)) { // annexb skip sei sps pps
                if (nal_type == 7) {
                    bool changeFlag = ParametersChange(NULL, -1, data, size, NULL, -1);
                    if (changeFlag) { // sps change
                        sps_buf_[sps_number_] = (uint8_t *)malloc(size);
                        memcpy(sps_buf_[sps_number_], data, size);
                        sps_len_[sps_number_] = size;
                        sps_number_++;
                        RewriteVideoExtraData();
                        log_debug("rewrite sps ok");
                    }
                }
                if (nal_type == 8) {
                    bool changeFlag = ParametersChange(NULL, -1, NULL, -1, data, size);
                    if (changeFlag) { // pps change
                        pps_buf_[pps_number_] = (uint8_t *)malloc(size);
                        memcpy(pps_buf_[pps_number_], data, size);
                        pps_len_[pps_number_] = size;
                        pps_number_++;
                        RewriteVideoExtraData();
                        log_debug("rewrite pps ok");
                    }
                }
                av_packet_unref(&pkt_);
                return 0;
            }
            auto data_copy = (uint8_t *)av_malloc(size + 4 + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(data_copy + 4, data, size);
            data_copy[0] = (size) >> 24;
            data_copy[1] = (size) >> 16;
            data_copy[2] = (size) >> 8;
            data_copy[3] = size & 0xff;
            av_packet_from_data(&pkt_, data_copy, size);
            pkt_.size = size + 4;
            if (nal_type == 5) {
                pkt_.flags |= AV_PKT_FLAG_KEY;
            }
        } else if (video_type_ == VIDEO_H265) {
            nal_type = (data[0] >> 1) & 0x3f;
            if (nal_type == 32 || nal_type == 19) {
                found_idr_ = true;
            }
            if (global_header_ && (nal_type == 32 || nal_type == 33 || nal_type == 34)) { // annexb skip vps sps pps
                if (nal_type == 32) {
                    bool changeFlag = ParametersChange(data, size, NULL, -1, NULL, -1);
                    if (changeFlag) { // vps change
                        vps_buf_[vps_number_] = (uint8_t *)malloc(size);
                        memcpy(vps_buf_[vps_number_], data, size);
                        vps_len_[vps_number_] = size;
                        vps_number_++;
                        RewriteVideoExtraData();
                        log_debug("rewrite vps ok");
                    }
                }
                if (nal_type == 33) {
                    bool changeFlag = ParametersChange(NULL, -1, data, size, NULL, -1);
                    if (changeFlag) { // sps change
                        sps_buf_[sps_number_] = (uint8_t *)malloc(size);
                        memcpy(sps_buf_[sps_number_], data, size);
                        sps_len_[sps_number_] = size;
                        sps_number_++;
                        RewriteVideoExtraData();
                        log_debug("rewrite sps ok");
                    }
                }
                if (nal_type == 34) {
                    bool changeFlag = ParametersChange(NULL, -1, NULL, -1, data, size);
                    if (changeFlag) { // pps change
                        pps_buf_[pps_number_] = (uint8_t *)malloc(size);
                        memcpy(pps_buf_[pps_number_], data, size);
                        pps_len_[pps_number_] = size;
                        pps_number_++;
                        RewriteVideoExtraData();
                        log_debug("rewrite pps ok");
                    }
                }
                av_packet_unref(&pkt_);
                return 0;
            }
            auto data_copy = (uint8_t *)av_malloc(size + 4 + AV_INPUT_BUFFER_PADDING_SIZE);
            memcpy(data_copy + 4, data, size);
            data_copy[0] = (size) >> 24;
            data_copy[1] = (size) >> 16;
            data_copy[2] = (size) >> 8;
            data_copy[3] = size & 0xff;
            av_packet_from_data(&pkt_, data_copy, size);
            pkt_.size = size + 4;
            if (nal_type == 19) {
                pkt_.flags |= AV_PKT_FLAG_KEY;
            }
        }
        if (!found_idr_) {
            log_warn("not found_idr_ nal_type:{}", nal_type);
            av_packet_unref(&pkt_);
            return 0;
        }
        if (frames_video_ == 0) {
            start_pts_video_ = pts;
            start_dts_video_ = dts;
        }
        frames_video_++;
        pkt_.pts = pts - start_pts_video_;
        pkt_.dts = dts - start_dts_video_;
        if (last_pts_video_ >= pkt_.pts) {
            // log_error("video pts error last_pts_video_:{} now pts:{}",last_pts_video_,pkt.pts);
            pkt_.pts = last_pts_video_ + 1;
        }
        if (last_dts_video_ >= pkt_.dts) {
            // log_error("video dts error last_dts_video_:{} now dts:{}",last_dts_video_,pkt.dts);
            pkt_.dts = last_dts_video_ + 1;
        }
        if (find_first_frame_) {
            pkt_.pts += start_media_pts_;
            pkt_.dts += start_media_pts_;
        }
        last_pts_video_ = pkt_.pts;
        last_dts_video_ = pkt_.dts;

        if (!find_first_frame_) {
            start_media_pts_ = pkt_.pts;
            find_first_frame_ = true;
        }
        int ret = av_interleaved_write_frame(fmt_ctx_, &pkt_);
        av_packet_unref(&pkt_);
        if (ret == 0) {
            return 0;
        } else {
            char errbuf[1024] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf) - 1);
            log_error("av_interleaved_write_frame failed:{} {} type:{} pts:{}", errbuf, ret, nal_type, last_pts_video_);
            return -1;
        }
    } else if (stream_index == audio_index_) {
        if (frames_audio_ == 0) {
            start_pts_audio_ = pts;
            start_dts_audio_ = dts;
        }
        frames_audio_++;
        pkt_.pts = pts - start_pts_audio_;
        pkt_.dts = dts - start_dts_audio_;
        if (last_pts_audio_ >= pkt_.pts) {
            // log_error("audio pts error last_pts_video_:{} now pts:{}",last_pts_audio_,pkt_.pts);
            pkt_.pts = last_pts_audio_ + 1;
        }
        if (last_dts_audio_ >= pkt_.dts) {
            // log_error("audio dts error last_dts_video_:{} now dts:{}",last_dts_audio_,pkt_.dts);
            pkt_.dts = last_dts_audio_ + 1;
        }
        if (find_first_frame_) {
            pkt_.pts += start_media_pts_;
            pkt_.dts += start_media_pts_;
        }
        last_pts_audio_ = pkt_.pts;
        last_dts_audio_ = pkt_.dts;
        if (!find_first_frame_) {
            start_media_pts_ = pkt_.pts;
            find_first_frame_ = true;
        }
        auto data_copy = (uint8_t *)av_malloc(size + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(data_copy, data, size);
        av_packet_from_data(&pkt_, data_copy, size);
        pkt_.size = size;
        int ret = av_interleaved_write_frame(fmt_ctx_, &pkt_);
        av_packet_unref(&pkt_);
        if (ret == 0) {
            return 0;
        } else {
            char errbuf[1024] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf) - 1);
            log_error("av_write_frame failed:{} ret:{}", errbuf, ret);
            return -1;
        }
    }
    return 0;
}

int Muxer::SendTrailer()
{
    if (!fmt_ctx_) {
        log_error("fmt ctx is NULL");
        return -1;
    }
    int ret = av_write_trailer(fmt_ctx_);
    if (ret != 0) {
        char errbuf[1024] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf) - 1);
        log_error("av_write_trailer failed:{}", errbuf);
        return -1;
    }
    log_debug("{}:SendTrailer ok", url_);
    return 0;
}

int Muxer::Open()
{
    if (!(ofmt_->flags & AVFMT_NOFILE)) {
        int ret = avio_open(&fmt_ctx_->pb, url_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            log_error("Could not open output file {}", url_);
            return -1;
        }
        log_debug("avio_open ok");
    }
    return 0;
}

int Muxer::GetAudioStreamIndex()
{
    return audio_index_;
}

int Muxer::GetVideoStreamIndex()
{
    return video_index_;
}
