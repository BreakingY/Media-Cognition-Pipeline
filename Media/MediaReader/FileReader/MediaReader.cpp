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

/*通过指定的buf，读取一个NALU数据到frame中*/
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

    pstart = buf->buf + buf->pos; // pstart跳过之前已经读取的数据指向本次NALU的起始码

    tmp = pstart + startcode; // 指向NALU数据

    for (int i = 0; i < buf->buf_len - buf->pos - 3; i++) // pos表示起始码的位置，所以已经读取的数据长度也应该是pos个字节
    {
        if (StartCode3(tmp) || StartCode4(tmp)) // 此时tmp指向下一个起始码位置
        {
            frame_len = tmp - pstart; // 包含起始码的NALU长度
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
    /*buf中的数据全部读取完毕，设置buf状态为WREIT*/
    if (buf->pos >= buf->buf_len) {
        buf->stat = WRITE;
    }
    return frame_len;
}

/*从buf中解析NALU数据*/
void MediaReader::PraseFrame()
{
    /*frameREAD,bufferWRITE状态不可访问*/
    if (frame_->stat == READ || buffer_->stat == WRITE) {
        printf("stat err\n");
        return;
    }

    /*buf处于可读状态并且frame处于可写状态*/
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
static double R2d(AVRational r)
{
    return r.den == 0 ? 0 : (double)r.num / (double)r.den;
}
void MediaReader::VideoInit(char *filename)
{
    int ret;
    char errors[1024];
    format_ctx_ = avformat_alloc_context();
    if ((ret = avformat_open_input(&format_ctx_, filename, NULL, NULL)) < 0) {

        av_strerror(ret, errors, 1024);
        DEBUGPRINT("Could not open source file: %s, %d(%s)\n", filename, ret, errors);
        exit(1);
    }

    if ((ret = avformat_find_stream_info(format_ctx_, NULL)) < 0) {
        av_strerror(ret, errors, 1024);
        DEBUGPRINT("Could not open source file: %s, %d(%s)\n", filename, ret, errors);
        exit(1);
    }

    // av_dump_format(format_ctx_, 0, filename, 0);

    audio_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio_index_ < 0) {
        DEBUGPRINT("no audio\n");
    }
    video_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_index_ < 0) {
        DEBUGPRINT("no video\n");
        exit(1);
    }
    AVCodecParameters *codec_parameters = format_ctx_->streams[video_index_]->codecpar;
    enum AVCodecID codec_id = codec_parameters->codec_id;
    is_mp4_ = true;
    std::string filetype(format_ctx_->iformat->name);
    DEBUGPRINT("file type:%s codec_id:%d AV_CODEC_ID_H264:%d AV_CODEC_ID_HEVC:%d\n", format_ctx_->iformat->name, codec_id, AV_CODEC_ID_H264, AV_CODEC_ID_HEVC);
    if (filetype.find("mpeg") != filetype.npos) {
        is_mp4_ = false;
    }
    if (codec_id == AV_CODEC_ID_H264 && is_mp4_) {
        const AVBitStreamFilter *pfilter = av_bsf_get_by_name("h264_mp4toannexb");
        av_bsf_alloc(pfilter, &bsf_ctx_);
        avcodec_parameters_copy(bsf_ctx_->par_in, format_ctx_->streams[video_index_]->codecpar);
        av_bsf_init(bsf_ctx_);
        DEBUGPRINT("AV_CODEC_ID_H264\n");
    } else if ((codec_id == AV_CODEC_ID_H265 || codec_id == AV_CODEC_ID_HEVC) && is_mp4_) {
        const AVBitStreamFilter *pfilter = av_bsf_get_by_name("hevc_mp4toannexb");
        av_bsf_alloc(pfilter, &bsf_ctx_);
        avcodec_parameters_copy(bsf_ctx_->par_in, format_ctx_->streams[video_index_]->codecpar);
        av_bsf_init(bsf_ctx_);
        DEBUGPRINT("AV_CODEC_ID_H265\n");
    }
    // av_init_packet(&packet_);
    memset(&packet_, 0, sizeof(packet_));
    AVStream *as = format_ctx_->streams[video_index_];
    fps_ = R2d(as->avg_frame_rate);
    printf("%s:%d fps_:%d\n", __FILE__, __LINE__, fps_);
    return;
}
enum VideoType MediaReader::GetVideoType()
{
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
    if (!HaveAudio()) {
        return AUDIO_NONE;
    }
    AVCodecParameters *codec_parameters = format_ctx_->streams[audio_index_]->codecpar;
    enum AVCodecID codecId = codec_parameters->codec_id;
    if (codecId == AV_CODEC_ID_AAC) {
        return AUDIO_AAC;
    }
    return AUDIO_NONE;
}

MediaReader::MediaReader(char *file_path)
{
    file_ = file_path;

    buffer_ = (struct BufSt*)malloc(sizeof(struct BufSt));
    buffer_->buf_len = 0;
    buffer_->pos = 0;
    buffer_->stat = WRITE;

    frame_ = (struct FrameSt*)malloc(sizeof(struct FrameSt));
    frame_->stat = WRITE;
    VideoInit(file_path);

    video_finish_ = false;
    audio_finish_ = false;
    file_finish_ = false;

    th_file_ = std::thread(MediaReaderThread, this);
    th_video_ = std::thread(VideoSyncThread, this);
    if (audio_index_ > 0) {
        int video_time = 1000 * 1000 / fps_;
        AVCodecParameters *codecpar = format_ctx_->streams[audio_index_]->codecpar;
        int audio_time = 1000 * 1000 / (codecpar->sample_rate / codecpar->frame_size);
        sync_threshold_ = std::max(video_time, audio_time);
        th_audio_ = std::thread(AudioSyncThread, this);
    } else {
        audio_finish_ = true;
    }
    loop_ = true;
}
bool MediaReader::HaveAudio()
{
    if (audio_index_ >= 0) {
        return true;
    }
    return false;
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

    int video_time = 1000 * 1000 / self->fps_;
    int audio_time = 1000 * 1000;
    if (self->HaveAudio()) {
        AVCodecParameters *codecpar = self->format_ctx_->streams[self->audio_index_]->codecpar;
        audio_time = 1000 * 1000 / (codecpar->sample_rate / codecpar->frame_size);
        printf("%s:%d sample_rate:%d frame_size:%d audio_time:%d video_time:%d\n", __FILE__, __LINE__, codecpar->sample_rate, codecpar->frame_size, audio_time, video_time);
    }
    int min_sleep_time = (video_time < audio_time) ? video_time : audio_time; // 微妙
    int last_idx = -1;
    bool have_report = false;
    while (!self->abort_) {
        if (self->file_finish_ == true) {
            if (self->video_finish_ && self->audio_finish_) {
                if (self->colse_cb_ != NULL && !have_report) {
                    self->colse_cb_();
                    have_report = true;
                }
            }
            av_usleep(min_sleep_time / 2);
            continue;
        }
        have_report = false;
        ret = av_read_frame(self->format_ctx_, &self->packet_);
        if (ret < 0) {
            self->file_finish_ = true;
            DEBUGPRINT("%s:%d %s file over\n", __FILE__, __LINE__, self->format_ctx_->url);
            av_packet_unref(&self->packet_); // av_read_frame返回小于0得时候也对packet_分配了缓冲区，所以要释放
            av_usleep(min_sleep_time / 2);
            continue;
        }
        if (self->packet_.stream_index == self->audio_index_) {
            if (last_idx == self->packet_.stream_index) {
                av_usleep(audio_time / 2);
            }
            AVPacket audio_packet;
            av_packet_ref(&audio_packet, &self->packet_);
            std::unique_lock<std::mutex> guard(self->audio_mtx_);
            self->audio_list_.push_back(audio_packet);
            guard.unlock();
            self->audio_cond_.notify_one();

        } else if (self->packet_.stream_index == self->video_index_) {
            if (last_idx == self->packet_.stream_index) {
                av_usleep(video_time);
            }
            AVPacket video_packet;
            av_packet_ref(&video_packet, &self->packet_);
            std::unique_lock<std::mutex> guard(self->video_mtx_);
            self->video_list_.push_back(video_packet);
            guard.unlock();
            self->video_cond_.notify_one();
        }
        av_packet_unref(&self->packet_);
        last_idx = self->packet_.stream_index;
    }
    av_packet_unref(&self->packet_);
    DEBUGPRINT("%s:%d MediaReaderThread exit\n", __FILE__, __LINE__);
    return NULL;
}
void MediaReader::Reset()
{
    std::unique_lock<std::mutex> guard_video(video_mtx_);
    while (!video_list_.empty()) {
        AVPacket video_packet;
        video_packet = video_list_.front();
        video_list_.pop_front();
        av_packet_unref(&video_packet);
    }
    guard_video.unlock();
    std::unique_lock<std::mutex> guard_audio(audio_mtx_);
    while (!audio_list_.empty()) {
        AVPacket audio_packet;
        audio_packet = audio_list_.front();
        audio_list_.pop_front();
        av_packet_unref(&audio_packet);
    }
    guard_audio.unlock();
    audio_reset_ = true;
    video_reset_ = true;
    av_seek_frame(format_ctx_, -1, 0, AVSEEK_FLAG_BACKWARD);
    file_finish_ = false;
    video_finish_ = false;
    if (audio_index_ >= 0) {
        audio_finish_ = false;
    }
    DEBUGPRINT("%s:%d reset ok\n", __FILE__, __LINE__);
    return;
}
void *MediaReader::VideoSyncThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int64_t curtimestamp;
    AVRational time_base = self->format_ctx_->streams[self->video_index_]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};
    int64_t start_time = av_gettime();
    int64_t starttimestamp = -1;
    int ret;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->video_mtx_); // std::unique_lock<std::mutex> guard方式手动解锁的时候不要使用self->video_mtx_.unlock()的方式解锁(会概率性报错)，推荐使用guard.unlock();
        if (self->file_finish_ && self->video_list_.empty()) {
            self->video_finish_ = true;
            guard.unlock();
            av_usleep(1000 * 1000 / self->fps_);
            continue;
        }
        if (self->video_reset_) {
            start_time = av_gettime();
            starttimestamp = -1;
            self->video_reset_ = !self->video_reset_;
        }
        if (!self->video_list_.empty()) {
            AVPacket video_packet;
            video_packet = self->video_list_.front();
            self->video_list_.pop_front();
            guard.unlock();
            curtimestamp = av_rescale_q(video_packet.dts, time_base, time_base_q); // 没有B帧的时候pts==dts，有B帧的时候pts!=dts
            if (starttimestamp == -1) {
                starttimestamp = curtimestamp;
                self->video_start_timestamp_ = starttimestamp;
            }
            // if (self->is_mp4_) {
            //     self->Mp4ToAnnexb(video_packet);
            // }
            // self->buffer_->buf = video_packet.data;
            // self->buffer_->buf_len = video_packet.size;
            // self->buffer_->stat = READ;
            // self->buffer_->pos = 0;
            int pts = av_rescale_q(video_packet.pts, time_base, time_base_q);
            int dts = av_rescale_q(video_packet.dts, time_base, time_base_q);
            
            int64_t now_time = av_gettime() - start_time;
            if (self->HaveAudio()) {
                int diff = curtimestamp - self->audio_now_time_;
                if (std::abs(diff) <= self->sync_threshold_) {
                    // do nothing
                } else {
                    if (diff < 0) { // 视频落后音频,加快播放
                        curtimestamp -= std::abs(diff);
                    } else if (diff > 0) { // 视频快于音频，降低播放速度
                        curtimestamp += std::abs(diff) / 2;
                    }
                }
                if ((curtimestamp - starttimestamp) > now_time) {
                    int sleepTime = curtimestamp - starttimestamp - now_time;
                    // printf("%s:%d video time:%ld sleepTime:%ld\n",__FILE__, __LINE__,curtimestamp,sleepTime);
                    av_usleep(sleepTime);
                }
            } else {
                if ((curtimestamp - starttimestamp) > now_time) {
                    int sleepTime = curtimestamp - starttimestamp - now_time;
                    // printf("%s:%d video time:%ld sleepTime:%ld\n",__FILE__, __LINE__,curtimestamp,sleepTime);
                    av_usleep(sleepTime);
                }
            }
            av_bsf_send_packet(self->bsf_ctx_, &video_packet);
            while (!self->abort_){
                av_packet_unref(&video_packet);
                if(self->is_mp4_){
                    ret = av_bsf_receive_packet(self->bsf_ctx_, &video_packet);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF){
                        break;
                    }
                    else if (ret < 0) {
                        printf("av bsf receive pkt failed!\n");
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
                    data.data = self->frame_->frame;         //+self->frame_->startcode;
                    data.data_len = self->frame_->frame_len; //-self->frame_->startcode;
                    data.pts = pts;
                    data.dts = dts;

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
            av_packet_unref(&video_packet);
        } else {
            auto now = std::chrono::system_clock::now();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            self->video_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double duration_millsecond = std::chrono::duration<double, std::milli>(t2 - t1).count();
            guard.unlock();
            continue;
        }
    }
    DEBUGPRINT("%s:%d VideoSyncThread over\n", __FILE__, __LINE__);
    return NULL;
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
            return aactype+1;
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
    return 4;//默认是44100
}

static int get_channel_config(int channels, int aactype){
    //如果是 AAC HEv2 通道数减半
    if(aactype == 28){
        return (channels / 2);
    }
    return channels;
}
void *MediaReader::AudioSyncThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int64_t curtimestamp;
    int64_t starttimestamp = -1;
    AVRational time_base = self->format_ctx_->streams[self->audio_index_]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};
    int64_t start_time = av_gettime();
    AVCodecParameters *codecpar = self->format_ctx_->streams[self->audio_index_]->codecpar;
    int audio_time = 1000 * 1000 / (codecpar->sample_rate / codecpar->frame_size);
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->audio_mtx_);
        if (self->file_finish_ && self->audio_list_.empty()) {
            self->audio_finish_ = true;
            guard.unlock();
            av_usleep(audio_time);
            continue;
        }
        if (self->audio_reset_) {
            start_time = av_gettime();
            starttimestamp = -1;
            self->audio_reset_ = !self->audio_reset_;
        }
        if (!self->audio_list_.empty()) {
            AVPacket audio_packet;
            audio_packet = self->audio_list_.front();
            self->audio_list_.pop_front();
            guard.unlock();
            curtimestamp = av_rescale_q(audio_packet.pts, time_base, time_base_q);
            if (starttimestamp == -1) {
                starttimestamp = curtimestamp;
                self->audio_start_timestamp_ = starttimestamp;
            }
            self->audio_now_time_ = curtimestamp;
            int64_t now_time = av_gettime() - start_time;
            if ((curtimestamp - starttimestamp) > now_time) {
                int sleepTime = curtimestamp - starttimestamp - now_time;
                // printf("%s:%d audio time:%ld sleepTime:%ld\n",__FILE__, __LINE__,curtimestamp,sleepTime);
                av_usleep(sleepTime);
            }
            AudioData audiodata;
            audiodata.pts = av_rescale_q(audio_packet.pts, time_base, time_base_q);
            audiodata.dts = av_rescale_q(audio_packet.dts, time_base, time_base_q);
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
            av_packet_unref(&audio_packet);
        } else {
            auto now = std::chrono::system_clock::now();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            self->audio_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double duration_millsecond = std::chrono::duration<double, std::milli>(t2 - t1).count();
            guard.unlock();
            continue;
        }
    }
    DEBUGPRINT("%s:%d AudioSyncThread over\n", __FILE__, __LINE__);
    return NULL;
}
MediaReader::~MediaReader()
{
    int ret;
    abort_ = true;
    th_file_.join();
    th_video_.join();
    while (!video_list_.empty()) {
        AVPacket packet = video_list_.front();
        video_list_.pop_front();
        av_packet_unref(&packet);
    }
    if (audio_index_ > 0) {
        th_audio_.join();
        while (!audio_list_.empty()) {
            AVPacket packet = video_list_.front();
            video_list_.pop_front();
            av_packet_unref(&packet);
        }
    }
    avformat_close_input(&format_ctx_);
    avformat_free_context(format_ctx_);
    av_packet_unref(&packet_);
    if(bsf_ctx_){
        av_bsf_free(&bsf_ctx_);
    }
    if(buffer_){
        free(buffer_);
        buffer_ = NULL;
    }
    if(frame_){
        free(frame_);
        frame_ = NULL;
    }
    DEBUGPRINT("~MediaReader\n");
}
