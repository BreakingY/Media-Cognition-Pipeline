#include "MediaReader.h"


static inline int StartCode3(unsigned char *buf)
{
    if (buf[0] == 0 && buf[1] == 0 && buf[2] == 1)
        return 1;
    else
        return 0;
}
static inline int StartCode4(unsigned char *buf)
{
    if (buf[0] == 0 && buf[1] == 0 && buf[2] == 0 && buf[3] == 1)
        return 1;
    else
        return 0;
}

int GetNALUFromBuf(unsigned char **frame, struct BufSt *buf)
{
    int startcode;
    unsigned char *pstart;
    unsigned char *tmp;
    int frame_len;
    int bufoverflag = 1;

    if (buf->pos >= buf->buf_len) {
        buf->stat = WRITE;
        // printf("h264buf empty\n");
        return -1;
    }

    if (!StartCode3(buf->buf + buf->pos) && !StartCode4(buf->buf + buf->pos)) {
        printf("statrcode err\n");
        return -1;
    }

    if (StartCode3(buf->buf + buf->pos)) {
        startcode = 3;
    } else
        startcode = 4;

    pstart = buf->buf + buf->pos;

    tmp = pstart + startcode;

    for (int i = 0; i < buf->buf_len - buf->pos - 3; i++)
    {
        if (StartCode3(tmp) || StartCode4(tmp))
        {
            frame_len = tmp - pstart;
            bufoverflag = 0;
            break;
        }
        tmp++;
    }
    if (bufoverflag == 1) {
        frame_len = buf->buf_len - buf->pos;
    }

    *frame = buf->buf + buf->pos;

    buf->pos += frame_len;
    if (buf->pos >= buf->buf_len) {
        buf->stat = WRITE;
    }
    return frame_len;
}

void MediaReader::PraseFrame()
{
    if (frame_->stat == READ || buffer_->stat == WRITE) {
        printf("stat err\n");
        return;
    }

    frame_->frame_len = GetNALUFromBuf(&frame_->frame, buffer_);

    if (frame_->frame_len < 0) {
        printf("GetNALUFromBuf err");
        return;
    }

    if (StartCode3(frame_->frame))
        frame_->startcode = 3;
    else
        frame_->startcode = 4;
    frame_->stat = READ;
    return;
}
/*
#define FF_PROFILE_AAC_MAIN 0
#define FF_PROFILE_AAC_LOW  1
#define FF_PROFILE_AAC_SSR  2
#define FF_PROFILE_AAC_LTP  3
#define FF_PROFILE_AAC_HE   4
#define FF_PROFILE_AAC_HE_V2 28
#define FF_PROFILE_AAC_LD   22
#define FF_PROFILE_AAC_ELD  38
#define FF_PROFILE_MPEG2_AAC_LOW 128
#define FF_PROFILE_MPEG2_AAC_HE  131
*/
static int get_audio_obj_type(int aactype){
    //AAC HE V2 = AAC LC + SBR + PS
    //AAV HE = AAC LC + SBR
    //所以无论是 AAC_HEv2 还是 AAC_HE 都是 AAC_LC
    switch(aactype){
        case 0:
        case 2:
        case 3:
            return aactype + 1;
        case 1:
        case 4:
        case 28:
            return 2;
        default:
            return 2;

    }
    return 2;
}

static int get_sample_rate_index(int freq, int aactype){

    int i = 0;
    int freq_arr[13] = {
        96000, 88200, 64000, 48000, 44100, 32000,
        24000, 22050, 16000, 12000, 11025, 8000, 7350
    };

    //如果是 AAC HEv2 或 AAC HE, 则频率减半
    if(aactype == 28 || aactype == 4){
        freq /= 2;
    }

    for(i=0; i< 13; i++){
        if(freq == freq_arr[i]){
            return i;
        }
    }
    return 4; // 默认是44100
}

static int get_channel_config(int channels, int aactype){
    // 如果是 AAC HEv2 通道数减半
    if(aactype == 28){
        return (channels / 2);
    }
    return channels;
}
void MediaReader::VideoInit(const char *filename)
{
    int ret;
    char errors[1024];
    format_ctx_ = avformat_alloc_context();
    if ((ret = avformat_open_input(&format_ctx_, filename, nullptr, nullptr)) < 0) {
        av_strerror(ret, errors, 1024);
        log_error("Could not open source file: {}, {}({})", filename, ret, errors);
        exit(1);
    }

    if ((ret = avformat_find_stream_info(format_ctx_, nullptr)) < 0) {
        av_strerror(ret, errors, 1024);
        log_error("Could not find stream, file: {}, {}({})", filename, ret, errors);
        exit(1);
    }

    // av_dump_format(format_ctx_, 0, filename, 0);

    audio_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    video_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    AVCodecParameters *codec_parameters = format_ctx_->streams[video_index_]->codecpar;
    enum AVCodecID codec_id = codec_parameters->codec_id;
    is_mp4_ = true;
    std::string filetype(format_ctx_->iformat->name);
    log_debug("file type:{}", format_ctx_->iformat->name);
    if (filetype.find("mpeg") != filetype.npos) {
        is_mp4_ = false;
    }
    if (codec_id == AV_CODEC_ID_H264 && is_mp4_) {
        const AVBitStreamFilter *pfilter = av_bsf_get_by_name("h264_mp4toannexb");
        av_bsf_alloc(pfilter, &bsf_ctx_);
        avcodec_parameters_copy(bsf_ctx_->par_in, format_ctx_->streams[video_index_]->codecpar);
        av_bsf_init(bsf_ctx_);
    } else if ((codec_id == AV_CODEC_ID_H265 || codec_id == AV_CODEC_ID_HEVC) && is_mp4_) {
        const AVBitStreamFilter *pfilter = av_bsf_get_by_name("hevc_mp4toannexb");
        av_bsf_alloc(pfilter, &bsf_ctx_);
        avcodec_parameters_copy(bsf_ctx_->par_in, format_ctx_->streams[video_index_]->codecpar);
        av_bsf_init(bsf_ctx_);
    }
    av_init_packet(&packet_);
    return;
}
MediaReader::MediaReader(const char *file_path)
{
    file_ = file_path;

    buffer_ = (struct BufSt*)malloc(sizeof(struct BufSt));
    buffer_->buf_len = 0;
    buffer_->pos = 0;
    buffer_->stat = WRITE;

    frame_ = (struct FrameSt*)malloc(sizeof(struct FrameSt));
    frame_->stat = WRITE;
    VideoInit(file_path);

    file_finish_ = false;

    th_file_ = std::thread(MediaReaderThread, this);
}
enum VideoType MediaReader::GetVideoType()
{
    if(video_index_ < 0){
        return VIDEO_NONE;
    }
    AVCodecParameters *codec_parameters = format_ctx_->streams[video_index_]->codecpar;
    enum AVCodecID codec_id = codec_parameters->codec_id;
    if (codec_id == AV_CODEC_ID_H264) {
        return VIDEO_H264;
    } else if (codec_id == AV_CODEC_ID_H265 || codec_id == AV_CODEC_ID_HEVC) {
        return VIDEO_H265;
    }
    return VIDEO_NONE;
}
enum AudioType MediaReader::GetAudioType()
{
    if (audio_index_ < 0) {
        return AUDIO_NONE;
    }
    AVCodecParameters *codec_parameters = format_ctx_->streams[audio_index_]->codecpar;
    enum AVCodecID codecId = codec_parameters->codec_id;
    if (codecId == AV_CODEC_ID_AAC) {
        return AUDIO_AAC;
    }
    return AUDIO_NONE;
}
void MediaReader::GetVideoCon(int &width, int &height, int &fps){
    if (video_index_ < 0) {
        width = height = fps = -1;
        return;
    }
    width = format_ctx_->streams[video_index_]->codecpar->width;
    height = format_ctx_->streams[video_index_]->codecpar->height;
    fps = av_q2d(format_ctx_->streams[video_index_]->avg_frame_rate);
}
void MediaReader::GetAudioCon(int &channels, int &sample_rate, int &profile, int &bit_per_sample)
{
    if (audio_index_ < 0) {
        channels = sample_rate = profile = bit_per_sample = -1;
        return;
    }
    sample_rate = format_ctx_->streams[audio_index_]->codecpar->sample_rate;
    channels = format_ctx_->streams[audio_index_]->codecpar->channels;
    profile = format_ctx_->streams[audio_index_]->codecpar->profile;
    bit_per_sample = format_ctx_->streams[audio_index_]->codecpar->bits_per_coded_sample;
    return;
}
void MediaReader::SetDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb)
{
    data_listner_ = lisnter;
    colse_cb_ = cb;
    return;
}

void *MediaReader::MediaReaderThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int ret;

    int64_t start_timestamp_pts_video = -1;
    int64_t start_timestamp_dts_video = -1;
    int64_t last_timestamp_pts_video = -1;
    int64_t last_timestamp_dts_video = -1;
    

    int64_t start_timestamp_pts_audio = -1;
    int64_t start_timestamp_dts_audio = -1;
    int64_t last_timestamp_pts_audio = -1;
    int64_t last_timestamp_dts_audio = -1;
    bool have_report = false;
    int64_t start_time = av_gettime();
    while (!self->abort_) {
        if (self->file_finish_ == true) {
            if (self->colse_cb_ != nullptr && !have_report) {
                self->colse_cb_();
                have_report = true;
            }
            av_usleep(50 * 1000);
            continue;
        }
        have_report = false;
        ret = av_read_frame(self->format_ctx_, &self->packet_);
        if (ret < 0) {
            self->file_finish_ = true;
            log_debug("{} file over", self->format_ctx_->url);
            av_packet_unref(&self->packet_);
            continue;
        }
        int64_t now_time = av_gettime() - start_time;
        AVRational time_base_q = {1, AV_TIME_BASE};
        if (self->packet_.stream_index == self->audio_index_) {
            if(start_timestamp_pts_audio == -1){
                start_timestamp_pts_audio = self->packet_.pts;
                start_timestamp_dts_audio = self->packet_.dts;
            }
            last_timestamp_pts_audio = self->packet_.pts;
            last_timestamp_dts_audio = self->packet_.dts;
            self->packet_.pts -= start_timestamp_pts_audio;
            self->packet_.dts -= start_timestamp_dts_audio;

            AVRational time_base = self->format_ctx_->streams[self->audio_index_]->time_base;
            int64_t curtimestamp = av_rescale_q(self->packet_.pts, time_base, time_base_q);
            if (curtimestamp > now_time){
                av_usleep(curtimestamp - now_time);
            }
            AVPacket &audio_packet = self->packet_;
            AudioData audiodata;
            audiodata.pts = av_rescale_q(audio_packet.pts, time_base, time_base_q) / 1000;
            audiodata.dts = av_rescale_q(audio_packet.dts, time_base, time_base_q) / 1000;
            audiodata.data_len = audio_packet.size;
            audiodata.data = audio_packet.data;
            audiodata.channels = self->format_ctx_->streams[self->audio_index_]->codecpar->channels;
            audiodata.profile = self->format_ctx_->streams[self->audio_index_]->codecpar->profile;
            audiodata.samplerate = self->format_ctx_->streams[self->audio_index_]->codecpar->sample_rate;
            if (self->data_listner_) {
                // 添加adts
                int profile = get_audio_obj_type(audiodata.profile) - 1;
                int sampling_frequency_index = get_sample_rate_index(audiodata.samplerate, audiodata.profile);
                int channel_config = get_channel_config(audiodata.channels, audiodata.profile);

                char adts_header_buf[7] = {0};
                GenerateAdtsHeader(adts_header_buf, audiodata.data_len,
                                profile,    // AAC编码级别
                                sampling_frequency_index, // 采样率 Hz
                                channel_config);
                unsigned char buffer[4 * 1024] = {0};
                memcpy(buffer, adts_header_buf, 7);
                memcpy(buffer + 7, audiodata.data,  audiodata.data_len);

                audiodata.data = buffer;
                audiodata.data_len += 7;
                self->data_listner_->OnAudioData(audiodata);
            }

        } else if (self->packet_.stream_index == self->video_index_) {
            if(start_timestamp_pts_video == -1){
                start_timestamp_pts_video = self->packet_.pts;
                start_timestamp_dts_video = self->packet_.dts;
            }
            last_timestamp_pts_video = self->packet_.pts;
            last_timestamp_dts_video = self->packet_.dts;
            self->packet_.pts -= start_timestamp_pts_video;
            self->packet_.dts -= start_timestamp_dts_video;

            AVRational time_base = self->format_ctx_->streams[self->video_index_]->time_base;
            int64_t curtimestamp = av_rescale_q(self->packet_.pts, time_base, time_base_q);
            if (curtimestamp > now_time){
                av_usleep(curtimestamp - now_time);
            }
            AVPacket &video_packet = self->packet_;
            av_bsf_send_packet(self->bsf_ctx_, &video_packet);
            while (!self->abort_){
                av_packet_unref(&video_packet);
                if(self->is_mp4_){
                    ret = av_bsf_receive_packet(self->bsf_ctx_, &video_packet);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
                        break;
                    }
                    else if (ret < 0) {
                        log_error("av bsf receive pkt failed!");
                        break;
                    }
                }
                self->buffer_->buf = video_packet.data;
                self->buffer_->buf_len = video_packet.size;
                self->buffer_->stat = READ;
                self->buffer_->pos = 0;
                while (self->buffer_->stat == READ) {
                    self->PraseFrame();
                    if (self->frame_->stat == WRITE) {
                        continue;
                    }
                    VideoData data;
                    data.data = self->frame_->frame;         // + self->frame_->startcode;
                    data.data_len = self->frame_->frame_len; // - self->frame_->startcode;
                    data.pts = av_rescale_q(video_packet.pts, time_base, time_base_q) / 1000;
                    data.dts = av_rescale_q(video_packet.dts, time_base, time_base_q) / 1000;

                    int type = -1;
                    AVCodecParameters *codec_parameters = self->format_ctx_->streams[self->video_index_]->codecpar;
                    enum AVCodecID codecId = codec_parameters->codec_id;
                    if (codecId == AV_CODEC_ID_H264) {
                        type = data.data[0] & 0x1f;
                    } else if (codecId == AV_CODEC_ID_H265 || codecId == AV_CODEC_ID_HEVC) {
                        type = (data.data[0] >> 1) & 0x3f;
                    }
                    // type == 9为分隔符
                    if (type == 9 || self->frame_->frame_len <= self->frame_->startcode) {
                        self->frame_->stat = WRITE;
                        continue;
                    }

                    if (self->data_listner_) {
                        self->data_listner_->OnVideoData(data);
                    }
                    self->frame_->stat = WRITE;
                }
            }
        }
        av_packet_unref(&self->packet_);
    }
    av_packet_unref(&self->packet_);
    log_debug("MediaReaderThread Finished");
    return nullptr;
}
MediaReader::~MediaReader()
{
    int ret;
    abort_ = true;
    th_file_.join();
    avformat_close_input(&format_ctx_);
    avformat_free_context(format_ctx_);
    av_packet_unref(&packet_);
    if(bsf_ctx_){
        av_bsf_free(&bsf_ctx_);
    }
    if(buffer_){
        free(buffer_);
        buffer_ = nullptr;
    }
    if(frame_){
        free(frame_);
        frame_ = nullptr;
    }
    log_debug("~MediaReader");
}
