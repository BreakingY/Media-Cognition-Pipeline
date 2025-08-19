#ifdef USE_FFMPEG_SOFT
#include "H264HardEncoder.h"
#include "log_helpers.h"

static const uint64 NANO_SECOND = UINT64_C(1000000000);
HardVideoEncoder::HardVideoEncoder()
{
    h264_codec_ctx_ = NULL;
    h264_codec_ = NULL;
    sws_context_ = NULL;
    abort_ = false;
    callback_ = NULL;
    nframe_counter_ = 0;
    time_ts_accum_ = 0;
    time_inited_ = 0;
    scale_id_ = std::thread(HardVideoEncoder::VideoScaleThread, this);
    encode_id_ = std::thread(HardVideoEncoder::VideoEncThread, this);
}
void HardVideoEncoder::SetDataCallback(EncDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}
HardVideoEncoder::~HardVideoEncoder()
{
    abort_ = true;
    encode_id_.join();
    scale_id_.join();
    bgr_frames_.clear();

    for (std::list<AVFrame *>::iterator it = yuv_frames_.begin(); it != yuv_frames_.end(); ++it) {
        AVFrame *frame = *it;
        av_freep(&frame->data[0]);
        av_frame_free(&frame);
    }
    yuv_frames_.clear();

    if (h264_codec_ctx_ != NULL) {
        avcodec_close(h264_codec_ctx_);
        avcodec_free_context(&h264_codec_ctx_);
        h264_codec_ctx_ = NULL;
    }
    if (sws_context_ != NULL) {
        sws_freeContext(sws_context_);
        sws_context_ = NULL;
    }

    log_info("~HardVideoEncoder");
}
int HardVideoEncoder::SoftEncInit(int width, int height, int fps)
{
    if (h264_codec_) {
        log_warn("has been init Encoder...");
        return -1;
    }
    decodec_id_ = AV_CODEC_ID_H264;
    h264_codec_ = avcodec_find_encoder(decodec_id_);
    h264_codec_ctx_ = avcodec_alloc_context3(h264_codec_);
    h264_codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    h264_codec_ctx_->pix_fmt = sw_pix_format_;
    h264_codec_ctx_->width = width;
    h264_codec_ctx_->height = height;
    h264_codec_ctx_->time_base.num = 1;
    h264_codec_ctx_->time_base.den = fps;
    h264_codec_ctx_->bit_rate = 4000000;
    h264_codec_ctx_->gop_size = 2 * fps;
    h264_codec_ctx_->thread_count = 1;
    h264_codec_ctx_->slices = 1; // int slice_count; // slice数 int slices; // 切片数量。 表示图片细分的数量。 用于并行解码。
    /**
     * 遇到问题：编码得到的h264文件播放时提示"non-existing PPS 0 referenced"
     * 分析原因：未将pps sps 等信息写入
     * 解决方案：加入标记AV_CODEC_FLAG2_LOCAL_HEADER
     */
    h264_codec_ctx_->flags |= AV_CODEC_FLAG2_LOCAL_HEADER;
    h264_codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
    AVDictionary *param = 0;
    // priv_data  属于每个编码器特有的设置域，用av_opt_set 设置
    av_opt_set(h264_codec_ctx_->priv_data, "preset", "ultrafast", 0);
    av_opt_set(h264_codec_ctx_->priv_data, "tune", "zerolatency", 0);
    av_dict_set(&param, "preset", "ultrafast", 0);
    av_dict_set(&param, "profile", "baseline", 0);
    if (avcodec_open2(h264_codec_ctx_, h264_codec_, &param) < 0) {
        log_error("Failed to open encoder!");
        avcodec_close(h264_codec_ctx_);
        avcodec_free_context(&h264_codec_ctx_);
        h264_codec_ = NULL;
        h264_codec_ctx_ = NULL;
        log_error("no decodec can be used");
        exit(1);
    }
    log_info("using soft enc");
    return 1;
}
int HardVideoEncoder::Init(cv::Mat bgr_frame, int fps)
{
    if (!h264_codec_) {
        SoftEncInit(bgr_frame.cols, bgr_frame.rows, fps); 
    }
    if (!sws_context_) {
        sws_context_ = sws_getContext(h264_codec_ctx_->width, h264_codec_ctx_->height, AV_PIX_FMT_BGR24, h264_codec_ctx_->width, h264_codec_ctx_->height, sw_pix_format_,
                                      SWS_FAST_BILINEAR, NULL, NULL, NULL);
    }

    return 1;
}

void *HardVideoEncoder::VideoScaleThread(void *arg)
{

    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    int last_width;
    long local_cnt = 0;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->bgr_mutex_);
        if (!self->bgr_frames_.empty()) {
            cv::Mat bgr_frame = self->bgr_frames_.front();
            self->bgr_frames_.pop_front();
            guard.unlock();
            // 如果尺寸发生变化需要重新初始化
            if (local_cnt == 0) {
                last_width = self->h264_codec_ctx_->width;
            }
            local_cnt++;
            if (last_width != bgr_frame.cols) {
                sws_freeContext(self->sws_context_);
                self->sws_context_ = NULL;
                self->sws_context_ = sws_getContext(bgr_frame.cols, bgr_frame.rows, AV_PIX_FMT_BGR24, self->h264_codec_ctx_->width, self->h264_codec_ctx_->height, self->sw_pix_format_,
                                                    SWS_FAST_BILINEAR, NULL, NULL, NULL);
            }
            last_width = bgr_frame.cols;
            AVFrame mat_frame;

            // avpicture_fill((AVPicture *)&mat_frame, bgr_frame.data, AV_PIX_FMT_BGR24, bgr_frame.cols, bgr_frame.rows);
            mat_frame.width = bgr_frame.cols;
            mat_frame.height = bgr_frame.rows;
            mat_frame.format = AV_PIX_FMT_BGR24;
            av_image_fill_arrays(mat_frame.data, mat_frame.linesize, bgr_frame.data, 
                                (AVPixelFormat)mat_frame.format, mat_frame.width, mat_frame.height, 1);


            AVFrame *yuv_frame = av_frame_alloc();
            yuv_frame->width = self->h264_codec_ctx_->width;
            yuv_frame->height = self->h264_codec_ctx_->height;
            yuv_frame->format = self->sw_pix_format_;
            // av_image_alloc()函数的功能是按照指定的宽、高、像素格式来分析图像内存。
            int tmpRet = av_image_alloc(yuv_frame->data, yuv_frame->linesize, yuv_frame->width, yuv_frame->height, self->sw_pix_format_, 1);
            if (tmpRet < 0) {
                log_critical("Alloc image date buffer failed");
            }

            sws_scale(self->sws_context_, mat_frame.data, mat_frame.linesize, 0, mat_frame.height,
                      yuv_frame->data, yuv_frame->linesize);
            std::unique_lock<std::mutex> guard(self->yuv_mutex_);
#ifdef DROP_FRAME
			// 丢帧处理
            if (self->yuv_frames_.size() > 5) {
                for (std::list<AVFrame *>::iterator it = self->yuv_frames_.begin();
                     it != self->yuv_frames_.end(); ++it) {
                    AVFrame *frame = *it;
                    av_freep(&frame->data[0]);
                    av_frame_free(&frame);
                }
                self->yuv_frames_.clear();
            }
#endif
            self->yuv_frames_.push_back(yuv_frame);
            guard.unlock();
            self->yuv_cond_.notify_one();
        } else {
            auto now = std::chrono::system_clock::now();
            self->bgr_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
        }
    }
    log_info("VideoScaleThread exit");
    return NULL;
}

void *HardVideoEncoder::VideoEncThread(void *arg)
{
    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    int ret = 0;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->yuv_mutex_);
        if (!self->yuv_frames_.empty()) {
            AVFrame *yuv_frame = self->yuv_frames_.front();
            self->yuv_frames_.pop_front();
            guard.unlock();

            ret = avcodec_send_frame(self->h264_codec_ctx_, yuv_frame);
            if (ret < 0) {
                log_error("Error sending a frame for encoding");
                av_freep(&yuv_frame->data[0]);
                av_frame_free(&yuv_frame);
                continue;
            }
            AVPacket *packet = av_packet_alloc();
            packet->data = NULL;
            packet->size = 0;
            yuv_frame->pts = self->nframe_counter_;
            self->nframe_counter_++;
            if (self->nframe_counter_ % self->h264_codec_ctx_->gop_size == 0) {
                yuv_frame->key_frame = 1;
                yuv_frame->pict_type = AV_PICTURE_TYPE_I;
            }
            while (ret >= 0) {
                ret = avcodec_receive_packet(self->h264_codec_ctx_, packet);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    log_error("Error during encoding");
                    break;
                }
                self->time_now_ = std::chrono::steady_clock::now();
                if (self->nframe_counter_ - 1 == 0) {
                    self->time_pre_ = self->time_now_;
                    self->time_ts_accum_ = 0;
                }
                // 编码后的数据sps pps也直接在packet->data里面,并且包含了起始码
                packet->stream_index = 0;
                packet->duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
                self->time_ts_accum_ += packet->duration;
                packet->pts = self->time_ts_accum_;
                if (self->callback_) {
                    self->callback_->OnVideoEncData((unsigned char *)packet->data, packet->size, packet->pts);
                }
                av_packet_unref(packet);
                self->time_pre_ = self->time_now_;
            }
            av_freep(&yuv_frame->data[0]);
            av_frame_free(&yuv_frame);
            av_packet_free(&packet);
        } else {
            auto now = std::chrono::system_clock::now();
            self->yuv_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    // 清空缓冲区 TODO 代码优化，这部分代码有点重复了
    ret = avcodec_send_frame(self->h264_codec_ctx_, NULL);
    AVPacket *packet = av_packet_alloc();
    packet->data = NULL;
    packet->size = 0;
    while (ret >= 0) {
        ret = avcodec_receive_packet(self->h264_codec_ctx_, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            log_error("Error during encoding");
            break;
        }
        self->time_now_ = std::chrono::steady_clock::now();
        if (self->nframe_counter_ - 1 == 0) {
            self->time_pre_ = self->time_now_;
            self->time_ts_accum_ = 0;
        }
        // 编码后的数据sps pps也直接在packet->data里面,并且包含了起始码
        packet->stream_index = 0;
        packet->duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
        self->time_ts_accum_ += packet->duration;
        packet->pts = self->time_ts_accum_;
        if (self->callback_) {
            self->callback_->OnVideoEncData((unsigned char *)packet->data, packet->size, packet->pts);
        }
        av_packet_unref(packet);
        self->time_pre_ = self->time_now_;
    }
    av_packet_free(&packet);

    log_info("VideoEncThread exit");
    return NULL;
}

int HardVideoEncoder::AddVideoFrame(cv::Mat bgr_frame)
{
    std::unique_lock<std::mutex> guard(bgr_mutex_);
#ifdef DROP_FRAME
	// 丢帧处理
    if (bgr_frames_.size() > 5) {
        bgr_frames_.clear();
    }
#endif
    bgr_frames_.push_back(bgr_frame);
    guard.unlock();
    bgr_cond_.notify_one();
    if (!time_inited_) {
        time_inited_ = 1;
        time_now_1_ = std::chrono::steady_clock::now();
        time_pre_1_ = time_now_1_;
        now_frames_ = pre_frames_ = 0;
    } else {
        time_now_1_ = std::chrono::steady_clock::now();
        long tmp_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_1_ - time_pre_1_).count();
        if (tmp_time > 1000) {
            log_debug(" output frame rate {} ", (now_frames_ - pre_frames_ + 1) * 1000 / tmp_time);
            time_pre_1_ = time_now_1_;
            pre_frames_ = now_frames_;
        }
        now_frames_++;
    }

    return 1;
}
#endif
