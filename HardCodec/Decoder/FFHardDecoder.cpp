#ifdef USE_FFMPEG_NVIDIA
#include "HardDecoder.h"
/**
 * 判断硬件解码类型支不支持，上面是通过 AVCodec 来判断的，实际上 FFmpeg 都给出了硬件类型的定义，在 AVHWDeviceType 枚举变量中。
 * enum AVHWDeviceType {
 *     AV_HWDEVICE_TYPE_NONE,
 *     AV_HWDEVICE_TYPE_VDPAU,
 *     AV_HWDEVICE_TYPE_CUDA,
 *     AV_HWDEVICE_TYPE_VAAPI,
 *     AV_HWDEVICE_TYPE_DXVA2,
 *     AV_HWDEVICE_TYPE_QSV,
 *     AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
 *     AV_HWDEVICE_TYPE_D3D11VA,
 *     AV_HWDEVICE_TYPE_DRM,
 *     AV_HWDEVICE_TYPE_OPENCL,
 *     AV_HWDEVICE_TYPE_MEDIACODEC,
 *     AV_HWDEVICE_TYPE_VULKAN,
 * }
 * 通过 av_hwdevice_get_type_name 方法可以将这些枚举值转换成对应的字符串，比如 AV_HWDEVICE_TYPE_MEDIACODEC 对应的字符串就是 mediacodec ，其实在源码里面也是有的：
 *
 * static const char *const hw_type_names[] = {
 *     [AV_HWDEVICE_TYPE_CUDA]   = "cuda",
 *     [AV_HWDEVICE_TYPE_DRM]    = "drm",
 *     [AV_HWDEVICE_TYPE_DXVA2]  = "dxva2",
 *     [AV_HWDEVICE_TYPE_D3D11VA] = "d3d11va",
 *     [AV_HWDEVICE_TYPE_OPENCL] = "opencl",
 *     [AV_HWDEVICE_TYPE_QSV]    = "qsv",
 *     [AV_HWDEVICE_TYPE_VAAPI]  = "vaapi",
 *     [AV_HWDEVICE_TYPE_VDPAU]  = "vdpau",
 *     [AV_HWDEVICE_TYPE_VIDEOTOOLBOX] = "videotoolbox",
 *     [AV_HWDEVICE_TYPE_MEDIACODEC] = "mediacodec",
 *     [AV_HWDEVICE_TYPE_VULKAN] = "vulkan",
 * }
 * 新版本avcodec_send_packet一次，需要循环调用avcodec_receive_frame多次，返回EAGAIN后，结束当前这次的解码，音频解码也是一样
 * AV_PIX_FMT_QSV 英特尔的qsv
 * AV_PIX_FMT_CUDA 英伟达cuda
 */
static const uint64_t NANO_SECOND = UINT64_C(1000000000);
// 硬件加速初始化
int HardVideoDecoder::hwDecoderInit(AVCodecContext *ctx, const enum AVHWDeviceType type)
{
    int err = 0;
    // 创建一个硬件设备上下文
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx_, type, NULL, NULL, 0)) < 0) {
        log_error("Failed to create specified HW device.");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    return err;
}
/**
 * ffmpeg -codecs | grep 264
 *        (decoders: h264 h264_v4l2m2m h264_cuvid ) (encoders: libx264 libx264rgb h264_nvenc h264_v4l2m2m h264_vaapi nvenc nvenc_h264 )
 */
// try to open hard decodec cuda:AV_PIX_FMT_CUDA

int HardVideoDecoder::HardDecInit()
{
    if (codec_) {
        log_warn("has been init Decoder...");
        return -1;
    }
    std::string codec_name;
    if (codec_type_ ==  CODEC_TYPE::CODEC_H264) {
        codec_name = "h264_cuvid";
    } 
    else if (codec_type_ ==  CODEC_TYPE::CODEC_H265){
        codec_name = "hevc_cuvid";
    }
    else{
        log_error("codec type error");
        exit(1);
    }
    // 列举支持的硬解码
    log_info("Available device types:");
    while ((type_ = av_hwdevice_iterate_types(type_)) != AV_HWDEVICE_TYPE_NONE) {
        log_info("{}", av_hwdevice_get_type_name(type_));
    }
    // 使用英伟达硬解码
    type_ = av_hwdevice_find_type_by_name("cuda");
    if (type_ == AV_HWDEVICE_TYPE_NONE) {
        codec_ctx_ = NULL;
        codec_ = NULL;
        return -1;
    }
    codec_ = avcodec_find_decoder_by_name(codec_name.c_str());
    if (!codec_) {
        codec_ctx_ = NULL;
        codec_ = NULL;
        return -1;
    }
    // 获取该硬解码器的像素格式。cuda对应的hw_pix_fmt_是AV_PIX_FMT_CUDA
    for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec_, i);
        if (!config) {
            log_error("get config error");
            return -1;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == type_) {
            hw_pix_fmt_ = config->pix_fmt;
            break;
        }
    }
    const char *pixname = av_get_pix_fmt_name(AVPixelFormat(hw_pix_fmt_));
    log_debug("hw_pix_fmt_:{}", pixname);
#if 0
    codec_->capabilities |= AV_CODEC_CAP_DELAY;
#endif
    codec_ctx_ = avcodec_alloc_context3(codec_);
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
    if (!codec_ctx_) {
        codec_ctx_ = NULL;
        codec_ = NULL;
        return -1;
    }
    codec_ctx_->thread_count = 1;
    // 硬解码格式
    codec_ctx_->pix_fmt = hw_pix_fmt_;
    av_opt_set_int(codec_ctx_, "refcounted_frames", 1, 0);
    if (hwDecoderInit(codec_ctx_, type_) < 0) {
        log_error("hard dec init failed");
        avcodec_close(codec_ctx_);
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = NULL;
        codec_ = NULL;
        return -1;
    }
    if (avcodec_open2(codec_ctx_, codec_, NULL) < 0) {
        log_error("hard dec init failed");
        avcodec_close(codec_ctx_);
        avcodec_free_context(&codec_ctx_);
        av_buffer_unref(&hw_device_ctx_);
        codec_ctx_ = NULL;
        codec_ = NULL;
        hw_device_ctx_ = NULL;
        return -1;
    }
    log_info("open hard dec ok");
    is_hard_ = true;
    return 1;
}
int HardVideoDecoder::SoftDecInit()
{
    if (codec_) {
        log_warn("has been init Decoder...");
        return -1;
    }
    if (codec_type_ ==  CODEC_TYPE::CODEC_H264) {
        decodec_id_ = AV_CODEC_ID_H264;
    } 
    else if (codec_type_ ==  CODEC_TYPE::CODEC_H265){
        decodec_id_ = AV_CODEC_ID_H265;
    }
    else{
        log_error("codec type error");
        exit(1);
    }
    // switch to soft decodec
    codec_ = avcodec_find_decoder(decodec_id_);
#if 0
    codec_->capabilities |= AV_CODEC_CAP_DELAY;
#endif
    codec_ctx_ = avcodec_alloc_context3(codec_);
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;
    if (codec_type_ ==  CODEC_TYPE::CODEC_H265) {
        codec_ctx_->thread_count = 3;
    } else {
        codec_ctx_->thread_count = 1;
    }
    if (avcodec_open2(codec_ctx_, codec_, NULL) < 0) {
        log_error("no decodec can be used");
        avcodec_close(codec_ctx_);
        avcodec_free_context(&codec_ctx_);
        exit(1);
    }
    log_info("open soft dec ok");
    return 1;
}
HardVideoDecoder::HardVideoDecoder(CODEC_TYPE codec_type)
{
    codec_ctx_ = NULL;
    codec_ = NULL;
    codec_type_ = codec_type;
    if (HardDecInit() < 0) {
        SoftDecInit();
    }
    abort_ = false;
    // av_init_packet(&packet_);
    memset(&packet_, 0, sizeof(packet_));
    frame_ = NULL;
    sw_frame_ = NULL;
    img_convert_ctx_ = NULL;
    callback_ = NULL;
    time_inited_ = 0;
    now_frames_ = pre_frames_ = 0;
    dec_thread_id_ = std::thread(HardVideoDecoder::DecodeThread, this);
    sws_thread_id_ = std::thread(HardVideoDecoder::ScaleThread, this);
}
HardVideoDecoder::~HardVideoDecoder()
{
    abort_ = true;

    dec_thread_id_.join();
    sws_thread_id_.join();

    for (std::list<AVFrame *>::iterator it = yuv_frames_.begin(); it != yuv_frames_.end(); ++it) {
        AVFrame *frame = *it;
        uint8_t *p = frame->data[0];
        av_freep(&p);
        av_frame_free(&frame);
    }
    yuv_frames_.clear();
    for (std::list<HardDataNode *>::iterator it = es_packets_.begin(); it != es_packets_.end(); ++it) {
        HardDataNode *packet = *it;
        delete packet;
    }
    es_packets_.clear();

    if (codec_ctx_) {
        if (codec_ctx_->extradata) {
            av_free(codec_ctx_->extradata);
            codec_ctx_->extradata = NULL;
            codec_ctx_->extradata_size = 0;
        }
        avcodec_close(codec_ctx_);
        if (codec_ctx_->opaque != NULL) {
            free(codec_ctx_->opaque);
            codec_ctx_->opaque = NULL;
        }
        avcodec_free_context(&codec_ctx_);
    }
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = NULL;
    }
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
        sw_frame_ = NULL;
    }
    if (img_convert_ctx_ != NULL) {
        sws_freeContext(img_convert_ctx_);
        img_convert_ctx_ = NULL;
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    av_packet_unref(&packet_);
    if(image_ptr_){
        free(image_ptr_);
        image_ptr_ = NULL;
    }
    log_debug("~HardVideoDecoder");
}
void HardVideoDecoder::SetFrameFetchCallback(DecDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}

void HardVideoDecoder::InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts)
{

    HardDataNode *node = new HardDataNode();
    node->es_data = (unsigned char *)malloc(data_len);
    memcpy(node->es_data, data, data_len);
    node->es_data_len = data_len;

    std::unique_lock<std::mutex> guard(packet_mutex_);
    es_packets_.push_back(node);
    guard.unlock();
    packet_cond_.notify_one();
    return;
}
void HardVideoDecoder::DecodeVideo(HardDataNode *data)
{
    packet_.data = data->es_data;
    packet_.size = data->es_data_len;
    int ret = avcodec_send_packet(codec_ctx_, &packet_);
    if (ret != 0) {
        av_packet_unref(&packet_);
        return;
    }
    AVFrame *tmp_frame = NULL;
    uint8_t *buffer = NULL;
    int size;
    // 新版本avcodec_send_packet一次，需要循环调用avcodec_receive_frame多次，返回EAGAIN后，结束当前这次的解码
    int cnt = 0;
    while (1) {
        if (!frame_) {
            frame_ = av_frame_alloc();
        }
        if (!sw_frame_) {
            sw_frame_ = av_frame_alloc();
        }
        ret = avcodec_receive_frame(codec_ctx_, frame_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0) {
            // log_debug("avcodec_receive_frame ret cnt:{} type:{}",cnt,type);
            // if(ret < 0){
            //     log_error("Error while decoding");
            // }
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = NULL;
            }
            if (sw_frame_) {
                av_frame_free(&sw_frame_);
                sw_frame_ = NULL;
            }
            av_packet_unref(&packet_);
            return;
        }
        if (frame_->format == hw_pix_fmt_) { // 硬解码
            // 将解码后的数据从GPU内存存格式转为CPU内存格式，并完成GPU到CPU内存的拷贝
            if ((ret = av_hwframe_transfer_data(sw_frame_, frame_, 0)) < 0) { // av_hwframe_transfer_data只支持NV12的格式转换，不能直接转换成RGB24，需要sws_scale完成
                log_error("Error transferring the data to system memory");
                if (frame_) {
                    av_frame_free(&frame_);
                    frame_ = NULL;
                }
                if (sw_frame_) {
                    av_frame_free(&sw_frame_);
                    sw_frame_ = NULL;
                }
                av_packet_unref(&packet_);
                return;
            }
            tmp_frame = sw_frame_;
        } else { // 软解

            tmp_frame = frame_;
        }
        // 计算一张YUV图需要的内存 大小
        size = av_image_get_buffer_size((AVPixelFormat)tmp_frame->format, tmp_frame->width, tmp_frame->height, 1);
        if (out_pix_fmt_ == AV_PIX_FMT_NONE) {
            const char *pixname = av_get_pix_fmt_name(AVPixelFormat(tmp_frame->format));
            // frame_->format是再gpu上的像素类型(hw_pix_fmt_)，tmp_frame->format是拷贝到cpu之后的类型(AV_PIX_FMT_NV12-由av_hwframe_transfer_data转换而来)，两者不一样
            log_debug("out_pix_fmt_:{}", pixname);
            out_pix_fmt_ = (AVPixelFormat)tmp_frame->format; // AV_PIX_FMT_NV12
        }
        buffer = (uint8_t *)av_malloc(size);
        ret = av_image_copy_to_buffer(buffer, size, (const uint8_t *const *)tmp_frame->data, (const int *)tmp_frame->linesize, (AVPixelFormat)tmp_frame->format, tmp_frame->width, tmp_frame->height, 1);
        if (ret < 0) {
            log_error("Can not copy image to buffer");
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = NULL;
            }
            if (sw_frame_) {
                av_frame_free(&sw_frame_);
                sw_frame_ = NULL;
            }
            av_freep(&buffer);
            av_packet_unref(&packet_);
            return;
        }
        AVFrame *frame_nv12 = av_frame_alloc();
        // ret = avpicture_fill((AVPicture *)frame_nv12, buffer, out_pix_fmt_, codec_ctx_->width, codec_ctx_->height);
        frame_nv12->width = codec_ctx_->width;
        frame_nv12->height = codec_ctx_->height;
        frame_nv12->format = out_pix_fmt_;
        av_image_fill_arrays(frame_nv12->data, frame_nv12->linesize, buffer, 
                            (AVPixelFormat)frame_nv12->format, frame_nv12->width, frame_nv12->height, 1);
        std::unique_lock<std::mutex> guard(frame_mutex_);
        yuv_frames_.push_back(frame_nv12);
        guard.unlock();
        frame_cond_.notify_one();

        if (frame_) {
            av_frame_free(&frame_);
            frame_ = NULL;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = NULL;
        }
        cnt++;
        // av_freep(&buffer);//调用avpicture_fill之后就不需要再这里释放，由后续流程释放
    }
    return;
}
void *HardVideoDecoder::DecodeThread(void *arg)
{

    HardVideoDecoder *self = (HardVideoDecoder *)arg;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->packet_mutex_);
        if (!self->es_packets_.empty()) {
            HardDataNode *pVideoPacket = self->es_packets_.front();
            self->es_packets_.pop_front();
            guard.unlock();
            self->DecodeVideo(pVideoPacket);

            delete pVideoPacket;
        } else {
            auto now = std::chrono::system_clock::now();
            self->packet_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    log_info("DecodeThread Finished ");
    // 刷新缓冲区
    HardDataNode *node = new HardDataNode();
    node->es_data = NULL;
    node->es_data_len = 0;
    self->DecodeVideo(node);
    delete node;
    return NULL;
}

void HardVideoDecoder::ScaleVideo(AVFrame *frame)
{
    if (!img_convert_ctx_) {
        img_convert_ctx_ = sws_getContext(frame->width, frame->height, out_pix_fmt_, frame->width, frame->height, AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL); // YUV(NV12)-->RGB
    }
    int size = frame->width * frame->height * 3;
    if(!image_ptr_){
        image_ptr_ = (unsigned char *)malloc(size);
    }
    /**
     * linesize[]数组中保存的是对应通道的数据宽度 ， 输出BGR为packed格式，所以指定linesize[0]既可，如果是planar格式，例如YUV420P
     * linesize[0]——-Y分量的宽度
     * linesize[1]——-U分量的宽度
     * linesize[2]——-V分量的宽度
     * linesize[i]的值并不一定等于图片的宽度，有时候为了对齐各解码器的CPU，实际尺寸会大于图片的宽度
     */
    int linesize[4] = {3 * frame->width, 0, 0, 0};

    sws_scale(img_convert_ctx_, frame->data, frame->linesize, 0, frame->height, (uint8_t **)&image_ptr_, linesize); // 处理后的数据放到image_ptr_中
    cv::Mat frame_mat(frame->height, frame->width, CV_8UC3, image_ptr_);
    cv::Mat frame_ret = frame_mat.clone();
    if (callback_ != NULL) {
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
                log_debug("input frame rate {} ", tmp_frame_rate);
                time_pre_ = time_now_;
                pre_frames_ = now_frames_;
            }
        }
        callback_->OnRGBData(frame_ret);
    }
    uint8_t *p = frame->data[0];
    av_freep(&p);
    av_frame_free(&frame);
    return;
}

void *HardVideoDecoder::ScaleThread(void *arg)
{
    HardVideoDecoder *self = (HardVideoDecoder *)arg;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->frame_mutex_);
        if (!self->yuv_frames_.empty()) {
            AVFrame *frame = self->yuv_frames_.front();
            self->yuv_frames_.pop_front();
            guard.unlock();
            self->ScaleVideo(frame);
        } else {
            auto now = std::chrono::system_clock::now();
            self->frame_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }

    log_info("ScaleThread Finished");
    return NULL;
}
#endif
