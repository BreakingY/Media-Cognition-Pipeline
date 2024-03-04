#include "MediaReader.h"

static inline int startCode3(unsigned char *buf)
{
    if (buf[0] == 0 && buf[1] == 0 && buf[2] == 1)
        return 1;
    else
        return 0;
}
static inline int startCode4(unsigned char *buf)
{
    if (buf[0] == 0 && buf[1] == 0 && buf[2] == 0 && buf[3] == 1)
        return 1;
    else
        return 0;
}

/*通过指定的buf，读取一个NALU数据到frame中*/
int getNALUFromBuf(unsigned char *frame, struct buf_st *buf)
{
    int startCode;
    unsigned char *pstart;
    unsigned char *tmp;
    int frameSize;
    int bufoverflag = 1;

    if (buf->pos >= buf->bufsize) {
        buf->stat = WRITE;
        // printf("h264buf empty\n");
        return -1;
    }

    if (!startCode3(buf->buf + buf->pos) && !startCode4(buf->buf + buf->pos)) {
        printf("statrcode err\n");
        return -1;
    }

    if (startCode3(buf->buf + buf->pos)) {
        startCode = 3;
    } else
        startCode = 4;

    pstart = buf->buf + buf->pos; // pstart跳过之前已经读取的数据指向本次NALU的起始码

    tmp = pstart + startCode; // 指向NALU数据

    for (int i = 0; i < buf->bufsize - buf->pos - 3; i++) // pos表示起始码的位置，所以已经读取的数据长度也应该是pos个字节
    {
        if (startCode3(tmp) || startCode4(tmp)) // 此时tmp指向下一个起始码位置
        {
            frameSize = tmp - pstart; // 包含起始码的NALU长度
            bufoverflag = 0;
            break;
        }
        tmp++;
    }
    if (bufoverflag == 1) {
        frameSize = buf->bufsize - buf->pos;
    }

    memcpy(frame, buf->buf + buf->pos, frameSize);

    buf->pos += frameSize;
    /*buf中的数据全部读取完毕，设置buf状态为WREIT*/
    if (buf->pos >= buf->bufsize) {
        buf->stat = WRITE;
    }
    return frameSize;
}

/*从buf中解析NALU数据*/
void MediaReader::praseFrame()
{
    /*frameREAD,bufferWRITE状态不可访问*/
    if (frame.stat == READ || buffer.stat == WRITE) {
        printf("stat err\n");
        return;
    }

    /*buf处于可读状态并且frame处于可写状态*/
    frame.frameSize = getNALUFromBuf(frame.frame, &buffer);

    if (frame.frameSize < 0) {
        printf("getNALUFromBuf err");
        return;
    }

    if (startCode3(frame.frame))
        frame.startCode = 3;
    else
        frame.startCode = 4;
    frame.stat = READ;
    /*for(int i=0;i<frame.frameSize+frame.startCode;i++)
               printf("%0x",frame.frame[i]);*/
    return;
}
static double r2d(AVRational r)
{
    return r.den == 0 ? 0 : (double)r.num / (double)r.den;
}
void MediaReader::VideoInit(char *filename)
{
    int ret;
    char errors[1024];
    if ((ret = avformat_open_input(&m_pFormatCtx, filename, NULL, NULL)) < 0) {

        av_strerror(ret, errors, 1024);
        DEBUGPRINT("Could not open source file: %s, %d(%s)\n", filename, ret, errors);
        exit(1);
    }

    if ((ret = avformat_find_stream_info(m_pFormatCtx, NULL)) < 0) {
        av_strerror(ret, errors, 1024);
        DEBUGPRINT("Could not open source file: %s, %d(%s)\n", filename, ret, errors);
        exit(1);
    }

    // av_dump_format(m_pFormatCtx, 0, filename, 0);

    audio_index = av_find_best_stream(m_pFormatCtx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
    if (audio_index < 0) {
        DEBUGPRINT("no audio\n");
    }
    video_index = av_find_best_stream(m_pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (video_index < 0) {
        DEBUGPRINT("no video\n");
        exit(1);
    }
    AVCodecParameters *codecParameters = m_pFormatCtx->streams[video_index]->codecpar;
    enum AVCodecID codecId = codecParameters->codec_id;
    isMP4 = true;
    std::string filetype(m_pFormatCtx->iformat->name);
    DEBUGPRINT("file type:%s codecId:%d AV_CODEC_ID_H264:%d AV_CODEC_ID_HEVC:%d\n", m_pFormatCtx->iformat->name, codecId, AV_CODEC_ID_H264, AV_CODEC_ID_HEVC);
    if (filetype.find("mpeg") != filetype.npos) {
        isMP4 = false;
    }
    if (codecId == AV_CODEC_ID_H264 && isMP4) {
        m_ph26xbsfc = av_bitstream_filter_init("h264_mp4toannexb");
        DEBUGPRINT("AV_CODEC_ID_H264\n");
    } else if ((codecId == AV_CODEC_ID_H265 || codecId == AV_CODEC_ID_HEVC) && isMP4) {
        m_ph26xbsfc = av_bitstream_filter_init("hevc_mp4toannexb");
        DEBUGPRINT("AV_CODEC_ID_H265\n");
    }
    av_init_packet(&m_packet);
    m_packet.data = NULL;
    m_packet.size = 0;
    AVStream *as = m_pFormatCtx->streams[video_index];
    m_fps = r2d(as->avg_frame_rate);
    printf("%s:%d m_fps:%d\n", __FILE__, __LINE__, m_fps);
}
enum VideoType MediaReader::getVideoType()
{
    AVCodecParameters *codecParameters = m_pFormatCtx->streams[video_index]->codecpar;
    enum AVCodecID codecId = codecParameters->codec_id;
    if (codecId == AV_CODEC_ID_H264) {
        return VIDEO_H264;
    } else if (codecId == AV_CODEC_ID_H265 || codecId == AV_CODEC_ID_HEVC) {
        return VIDEO_H265;
    }
    return VIDEO_NONE;
}
enum AudioType MediaReader::getAudioType()
{
    if (!haveAudio()) {
        return AUDIO_NONE;
    }
    AVCodecParameters *codecParameters = m_pFormatCtx->streams[audio_index]->codecpar;
    enum AVCodecID codecId = codecParameters->codec_id;
    if (codecId == AV_CODEC_ID_AAC) {
        return AUDIO_AAC;
    }
    return AUDIO_NONE;
}

int MediaReader::mp4toannexb(AVPacket &v_packet)
{
    uint8_t *out_data = NULL;
    int out_size = 0;
    int ret = 0;
    AVRational time_base = m_pFormatCtx->streams[video_index]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};
    long curtimestamp = av_rescale_q(v_packet.pts, time_base, time_base_q);

    av_bitstream_filter_filter(m_ph26xbsfc, m_pFormatCtx->streams[video_index]->codec, NULL, &out_data, &out_size, v_packet.data, v_packet.size, v_packet.flags & AV_PKT_FLAG_KEY);

    AVPacket TmpPkt;
    av_init_packet(&TmpPkt);
    av_packet_copy_props(&TmpPkt, &v_packet);
    av_packet_from_data(&TmpPkt, out_data, out_size);
    TmpPkt.size = out_size;
    av_packet_unref(&v_packet);

    av_copy_packet(&v_packet, &TmpPkt);
    av_packet_unref(&TmpPkt);

    return curtimestamp;
}
MediaReader::MediaReader(char *file_path)
{
    file = file_path;

    memset(buffer.buf, 0, BUFF_MAX);
    buffer.bufsize = 0;
    buffer.pos = 0;
    buffer.stat = WRITE;

    memset(frame.frame, 0, BUFF_MAX);
    frame.stat = WRITE;
    VideoInit(file_path);

    videoFinish = false;
    audioFinish = false;
    fileFinish = false;

    th_file = std::thread(MediaReaderThread, this);
    th_video = std::thread(VideoSyncThread, this);
    if (audio_index > 0) {
        int videoTime = 1000 * 1000 / m_fps;
        AVCodecContext *aCodecCtx = m_pFormatCtx->streams[audio_index]->codec;
        int audioTime = 1000 * 1000 / (aCodecCtx->sample_rate / aCodecCtx->frame_size);
        sync_threshold = std::max(videoTime, audioTime);
        th_audio = std::thread(AudioSyncThread, this);
    } else {
        audioFinish = true;
    }
    Loop = true;
}
bool MediaReader::haveAudio()
{
    if (audio_index >= 0) {
        return true;
    }
    return false;
}
void MediaReader::getAudioCon(int &channels, int &sample_rate, int &audio_object_type, int &bit_per_sample)
{
    if (audio_index < 0) {
        channels = sample_rate = audio_object_type = bit_per_sample = -1;
        return;
    }
    sample_rate = m_pFormatCtx->streams[audio_index]->codecpar->sample_rate;
    channels = m_pFormatCtx->streams[audio_index]->codecpar->channels;
    // profile = MPEG-4 Audio Object Type - 1
    audio_object_type = m_pFormatCtx->streams[audio_index]->codecpar->profile + 1;
    bit_per_sample = m_pFormatCtx->streams[audio_index]->codecpar->bits_per_coded_sample;
    return;
}
void MediaReader::setDataListner(MediaDataListner *lisnter, CloseCallbackFunc cb)
{
    DataListner = lisnter;
    colseCb = cb;
}

void *MediaReader::MediaReaderThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int ret;

    int videoTime = 1000 * 1000 / self->m_fps;
    int audioTime = 1000 * 1000;
    if (self->haveAudio()) {
        AVCodecContext *aCodecCtx = self->m_pFormatCtx->streams[self->audio_index]->codec;
        audioTime = 1000 * 1000 / (aCodecCtx->sample_rate / aCodecCtx->frame_size);
        printf("%s:%d sample_rate:%d frame_size:%d audioTime:%d videoTime:%d\n", __FILE__, __LINE__, aCodecCtx->sample_rate, aCodecCtx->frame_size, audioTime, videoTime);
    }
    int minSleepTime = (videoTime < audioTime) ? videoTime : audioTime; // 微妙
    int last_idx = -1;
    bool haveReport = false;
    while (!self->abort) {
        if (self->fileFinish == true) {
            if (self->videoFinish && self->audioFinish) {
                if (self->colseCb != NULL && !haveReport) {
                    self->colseCb();
                    haveReport = true;
                }
            }
            av_usleep(minSleepTime / 2);
            continue;
        }
        haveReport = false;
        ret = av_read_frame(self->m_pFormatCtx, &self->m_packet);
        if (ret < 0) {
            self->fileFinish = true;
            DEBUGPRINT("%s:%d %s file over\n", __FILE__, __LINE__, self->m_pFormatCtx->filename);
            av_packet_unref(&self->m_packet); // av_read_frame返回小于0得时候也对m_packet分配了缓冲区，所以要释放
            av_usleep(minSleepTime / 2);
            continue;
        }
        if (self->m_packet.stream_index == self->audio_index) {
            if (last_idx == self->m_packet.stream_index) {
                av_usleep(audioTime / 2);
            }
            AVPacket audio_packet;
            av_packet_ref(&audio_packet, &self->m_packet);
            std::unique_lock<std::mutex> guard(self->audio_mtx);
            self->audio_list.push_back(audio_packet);
            guard.unlock();
            self->audio_cond.notify_one();

        } else if (self->m_packet.stream_index == self->video_index) {
            if (last_idx == self->m_packet.stream_index) {
                av_usleep(videoTime);
            }
            AVPacket video_packet;
            av_packet_ref(&video_packet, &self->m_packet);
            std::unique_lock<std::mutex> guard(self->video_mtx);
            self->video_list.push_back(video_packet);
            guard.unlock();
            self->video_cond.notify_one();
        }
        av_packet_unref(&self->m_packet);
        last_idx = self->m_packet.stream_index;
    }
    av_packet_unref(&self->m_packet);
    DEBUGPRINT("%s:%d MediaReaderThread exit\n", __FILE__, __LINE__);
    return NULL;
}
void MediaReader::reset()
{
    std::unique_lock<std::mutex> guard_video(video_mtx);
    while (!video_list.empty()) {
        AVPacket video_packet;
        video_packet = video_list.front();
        video_list.pop_front();
        av_packet_unref(&video_packet);
    }
    guard_video.unlock();
    std::unique_lock<std::mutex> guard_audio(audio_mtx);
    while (!audio_list.empty()) {
        AVPacket audio_packet;
        audio_packet = audio_list.front();
        audio_list.pop_front();
        av_packet_unref(&audio_packet);
    }
    guard_audio.unlock();
    audio_reset = true;
    video_reset = true;
    av_seek_frame(m_pFormatCtx, -1, 0, AVSEEK_FLAG_BACKWARD);
    fileFinish = false;
    videoFinish = false;
    if (audio_index >= 0) {
        audioFinish = false;
    }
    DEBUGPRINT("%s:%d reset ok\n", __FILE__, __LINE__);
}
void *MediaReader::VideoSyncThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int64_t curtimestamp;
    AVRational time_base = self->m_pFormatCtx->streams[self->video_index]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};
    int64_t start_time = av_gettime();
    int64_t starttimestamp = -1;
    while (!self->abort) {
        std::unique_lock<std::mutex> guard(self->video_mtx); // std::unique_lock<std::mutex> guard方式手动解锁的时候不要使用self->video_mtx.unlock()的方式解锁(会概率性报错)，推荐使用guard.unlock();
        if (self->fileFinish && self->video_list.empty()) {
            self->videoFinish = true;
            guard.unlock();
            av_usleep(1000 * 1000 / self->m_fps);
            continue;
        }
        if (self->video_reset) {
            start_time = av_gettime();
            starttimestamp = -1;
            self->video_reset = !self->video_reset;
        }
        if (!self->video_list.empty()) {
            AVPacket video_packet;
            video_packet = self->video_list.front();
            self->video_list.pop_front();
            guard.unlock();
            curtimestamp = av_rescale_q(video_packet.dts, time_base, time_base_q); // 没有B帧的时候pts==dts，有B帧的时候pts!=dts
            if (starttimestamp == -1) {
                starttimestamp = curtimestamp;
                self->video_start_timestamp = starttimestamp;
            }
            if (self->isMP4) {
                self->mp4toannexb(video_packet);
            }
            memcpy(self->buffer.buf, video_packet.data, video_packet.size);
            self->buffer.bufsize = video_packet.size;
            self->buffer.stat = READ;
            self->buffer.pos = 0;
            int pts = av_rescale_q(video_packet.pts, time_base, time_base_q);
            int dts = av_rescale_q(video_packet.dts, time_base, time_base_q);
            av_packet_unref(&video_packet);
            int64_t now_time = av_gettime() - start_time;
            if (self->haveAudio()) {
                int diff = curtimestamp - self->audio_now_time;
                if (std::abs(diff) <= self->sync_threshold) {
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

            while (self->buffer.stat == READ) {
                self->praseFrame();
                if (self->frame.stat == WRITE) {
                    continue;
                }
                VideoData data;
                data.data = self->frame.frame;         //+self->frame.startCode;
                data.data_len = self->frame.frameSize; //-self->frame.startCode;
                data.pts = pts;
                data.dts = dts;

                int type = -1;
                AVCodecParameters *codecParameters = self->m_pFormatCtx->streams[self->video_index]->codecpar;
                enum AVCodecID codecId = codecParameters->codec_id;
                if (codecId == AV_CODEC_ID_H264) {
                    type = data.data[0] & 0x1f;
                } else if (codecId == AV_CODEC_ID_H265 || codecId == AV_CODEC_ID_HEVC) {
                    type = (data.data[0] >> 1) & 0x3f;
                }
                // type == 9为分隔符
                if (type == 9 || self->frame.frameSize <= self->frame.startCode) {
                    self->frame.stat = WRITE;
                    continue;
                }

                if (self->DataListner) {
                    self->DataListner->OnVideoData(data);
                }
                self->frame.stat = WRITE;
            }
        } else {
            auto now = std::chrono::system_clock::now();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            self->video_cond.wait_until(guard, now + std::chrono::milliseconds(100));
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double duration_millsecond = std::chrono::duration<double, std::milli>(t2 - t1).count();
            guard.unlock();
            continue;
        }
    }
    DEBUGPRINT("%s:%d VideoSyncThread over\n", __FILE__, __LINE__);
    return NULL;
}
void *MediaReader::AudioSyncThread(void *arg)
{
    MediaReader *self = (MediaReader *)arg;
    int64_t curtimestamp;
    int64_t starttimestamp = -1;
    AVRational time_base = self->m_pFormatCtx->streams[self->audio_index]->time_base;
    AVRational time_base_q = {1, AV_TIME_BASE};
    int64_t start_time = av_gettime();
    AVCodecContext *aCodecCtx = self->m_pFormatCtx->streams[self->audio_index]->codec;
    int audioTime = 1000 * 1000 / (aCodecCtx->sample_rate / aCodecCtx->frame_size);
    while (!self->abort) {
        std::unique_lock<std::mutex> guard(self->audio_mtx);
        if (self->fileFinish && self->audio_list.empty()) {
            self->audioFinish = true;
            guard.unlock();
            av_usleep(audioTime);
            continue;
        }
        if (self->audio_reset) {
            start_time = av_gettime();
            starttimestamp = -1;
            self->audio_reset = !self->audio_reset;
        }
        if (!self->audio_list.empty()) {
            AVPacket audio_packet;
            audio_packet = self->audio_list.front();
            self->audio_list.pop_front();
            guard.unlock();
            curtimestamp = av_rescale_q(audio_packet.pts, time_base, time_base_q);
            if (starttimestamp == -1) {
                starttimestamp = curtimestamp;
                self->audio_start_timestamp = starttimestamp;
            }
            self->audio_now_time = curtimestamp;
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
            audiodata.channels = self->m_pFormatCtx->streams[self->audio_index]->codecpar->channels;
            audiodata.profile = self->m_pFormatCtx->streams[self->audio_index]->codecpar->profile;
            audiodata.samplerate = self->m_pFormatCtx->streams[self->audio_index]->codecpar->sample_rate;
            if (self->DataListner) {
                self->DataListner->OnAudioData(audiodata);
            }
            av_packet_unref(&audio_packet);
        } else {
            auto now = std::chrono::system_clock::now();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            self->audio_cond.wait_until(guard, now + std::chrono::milliseconds(100));
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
    abort = true;
    th_file.join();
    th_video.join();
    while (!video_list.empty()) {
        AVPacket packet = video_list.front();
        video_list.pop_front();
        av_packet_unref(&packet);
    }
    if (audio_index > 0) {
        th_audio.join();
        while (!audio_list.empty()) {
            AVPacket packet = video_list.front();
            video_list.pop_front();
            av_packet_unref(&packet);
        }
    }
    avformat_close_input(&m_pFormatCtx);
    av_packet_unref(&m_packet);
    if (isMP4) {
        av_bitstream_filter_close(m_ph26xbsfc);
    }
    DEBUGPRINT("~MediaReader\n");
}
