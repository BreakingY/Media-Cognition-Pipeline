#ifdef USE_MPP
#include "HardDecoder.h"
#define ALIGN(a, n) (((a) + (n) - 1) / (n) * (n))
#ifndef CHECK_MPP
#define CHECK_MPP(callstr)\
    {\
        MPP_RET error_code = callstr;\
        if (error_code != MPP_OK) {\
            std::cerr << "mpp error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif
HardVideoDecoder::HardVideoDecoder(CODEC_TYPE codec_type)
{
    codec_type_ = codec_type;
    if (codec_type_ ==  CODEC_TYPE::CODEC_H264) {
        type_ = MPP_VIDEO_CodingAVC;
    } 
    else if (codec_type_ ==  CODEC_TYPE::CODEC_H265){
        type_ = MPP_VIDEO_CodingHEVC;
    }
    else{
        log_error("codec type error");
        exit(1);
    }
    abort_ = false;
    callback_ = NULL;
    time_inited_ = 0;
    now_frames_ = pre_frames_ = 0;
}
HardVideoDecoder::~HardVideoDecoder()
{
    abort_ = true;
    dec_thread_id_.join();
    CHECK_MPP(mpi_->reset(ctx_));
    if(packet_){
        CHECK_MPP(mpp_packet_deinit(&packet_));
        packet_ = NULL;
    }
    if(ctx_){
        CHECK_MPP(mpp_destroy(ctx_));
        ctx_ = NULL;
    }
    if (group_) {
        mpp_buffer_group_put(group_);
        group_ = NULL;
    }
    if(cfg_){
        CHECK_MPP(mpp_dec_cfg_deinit(cfg_));
        cfg_ = NULL;
    }
    if(image_ptr_){
        free(image_ptr_);
        image_ptr_ = NULL;
    }
    for(auto &kv : src_handles_){
        releasebuffer_handle(kv.second);
    }
    src_handles_.clear();
    if(dst_handle_){
        releasebuffer_handle(dst_handle_);
        dst_handle_ = 0;
    }
    for (std::list<HardDataNode *>::iterator it = es_packets_.begin(); it != es_packets_.end(); ++it){
        HardDataNode *packet = *it;
        delete packet;
    }
    es_packets_.clear();
    log_debug("~HardVideoDecoder");
}
void HardVideoDecoder::Init(int32_t device_id, int width, int height){
    device_id_ = device_id;
    width_ = width;
    height_ = height;
    CHECK_MPP(mpp_packet_init(&packet_, NULL, 0));
    CHECK_MPP(mpp_create(&ctx_, &mpi_));
    CHECK_MPP(mpp_init(ctx_, MPP_CTX_DEC, type_));

    CHECK_MPP(mpp_dec_cfg_init(&cfg_));
    CHECK_MPP(mpi_->control(ctx_, MPP_DEC_GET_CFG, cfg_));
    RK_U32 need_split = 1;
    CHECK_MPP(mpp_dec_cfg_set_u32(cfg_, "base:split_parse", need_split));
    CHECK_MPP(mpi_->control(ctx_, MPP_DEC_SET_CFG, cfg_));

    dec_thread_id_ = std::thread(HardVideoDecoder::DecodeThread, this);
    return;
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
static uint64_t GetCurrentTimeMs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_ms);
}
void HardVideoDecoder::DecodeVideo(HardDataNode *data)
{
    RK_U32 pkt_done = 0;
    RK_U32 pkt_eos = 0;
    MPP_RET ret;
    mpp_packet_set_pts(packet_, GetCurrentTimeMs());
    mpp_packet_set_data(packet_, data->es_data);
    mpp_packet_set_size(packet_, data->es_data_len);
    mpp_packet_set_pos(packet_, data->es_data);
    mpp_packet_set_length(packet_, data->es_data_len);
    // setup eos flag
    if (data->es_data == NULL){
        pkt_eos = 1;
        CHECK_MPP(mpp_packet_set_eos(packet_));
    }
    while(true){
        RK_U32 frm_eos = 0;
        RK_S32 times = 30;
        if(!pkt_done){
            ret = mpi_->decode_put_packet(ctx_, packet_);
            if(MPP_OK == ret){
                pkt_done = 1;
            }
        }
        while(true){
            RK_S32 get_frm = 0;
            MppFrame frame = NULL;
            ret = mpi_->decode_get_frame(ctx_, &frame);
            if (MPP_ERR_TIMEOUT == ret) {
                if (times > 0) {
                    times--;
                    usleep(1 * 1000);
                    continue;
                }
                log_warn("decode_get_frame failed too much time");
            }
            if(ret != MPP_OK){
                log_error("decode_get_frame failed ret:{}", (int)ret);
            }
            if(frame){
                if(mpp_frame_get_info_change(frame)){
                    width_frame_ = mpp_frame_get_width(frame);
                    height_frame_ = mpp_frame_get_height(frame);
                    hor_stride_frame_ = mpp_frame_get_hor_stride(frame);
                    ver_stride_frame_ = mpp_frame_get_ver_stride(frame);
                    if(width_frame_ != width_ || height_frame_ != height_){
                        log_warn("width_:{} height_:{} width_frame_:{} height_frame_:{}", width_, height_, width_frame_, height_frame_);
                    }
                    for(auto &kv : src_handles_){
                        releasebuffer_handle(kv.second);
                    }
                    src_handles_.clear();
                    if(dst_handle_){
                        releasebuffer_handle(dst_handle_);
                        dst_handle_ = 0;
                    }
                    if(image_ptr_){
                        free(image_ptr_);
                        image_ptr_ = NULL; 
                    }
                    width_ = width_frame_;
                    height_ = height_frame_;
                    int dst_w_align = ALIGN(width_, 16); 
                    int out_size = dst_w_align * height_ * 3;
                    image_ptr_ = (unsigned char *)malloc(out_size);

                    buf_size_ = mpp_frame_get_buf_size(frame);
                    if(group_){
                        CHECK_MPP(mpp_buffer_group_clear(group_));
                    }
                    else{
                        CHECK_MPP(mpp_buffer_group_get_internal(&group_, MPP_BUFFER_TYPE_ION));
                    }
                    CHECK_MPP(mpp_buffer_group_limit_config(group_, buf_size_, buf_count_));
                    CHECK_MPP(mpi_->control(ctx_, MPP_DEC_SET_EXT_BUF_GROUP, group_));
                    CHECK_MPP(mpi_->control(ctx_, MPP_DEC_SET_INFO_CHANGE_READY, NULL));
                }
                else{
                    RK_U32 err_info = mpp_frame_get_errinfo(frame);
                    RK_U32 discard = mpp_frame_get_discard(frame);
                    MppBuffer buffer = NULL;
                    RK_U8 *base = NULL;
                    if(!err_info){
                        MppFrameFormat fmt = mpp_frame_get_fmt(frame);
                        if(fmt != MPP_FMT_YUV420SP){
                            log_error("fmt:{} MPP_FMT_YUV420SP:{}", (int)fmt, (int)MPP_FMT_YUV420SP);
                        }
                        buffer = mpp_frame_get_buffer(frame);
                        base = (RK_U8 *)mpp_buffer_get_ptr(buffer);
                        uint64_t t1 = GetCurrentTimeMs();
                        rga_buffer_handle_t src_handle;
                        int dst_w_align = ALIGN(width_, 16); 
                        int out_size = dst_w_align * height_ * 3;
                        int fd = mpp_buffer_get_fd(buffer);
                        auto it = src_handles_.find(fd);
                        if (it == src_handles_.end()) {
                            src_handle = (fd >= 0) ? importbuffer_fd(fd, buf_size_) : importbuffer_virtualaddr(base, buf_size_);
                            src_handles_[fd] = src_handle;
                        } else {
                            src_handle = it->second;
                        }
                        if(!dst_handle_){
                            dst_handle_ = importbuffer_virtualaddr(image_ptr_, out_size);
                        }
                        rga_buffer_t src_img = wrapbuffer_handle(src_handle, width_, height_, RK_FORMAT_YCbCr_420_SP, hor_stride_frame_, ver_stride_frame_);
                        rga_buffer_t dst_img = wrapbuffer_handle(dst_handle_, width_, height_, RK_FORMAT_BGR_888, dst_w_align, height_);

                        // imsetColorSpace(&src_img, IM_YUV_BT709_LIMIT_RANGE);
                        // imsetColorSpace(&dst_img, IM_RGB_FULL_RANGE);

                        // IM_STATUS ret_rga = imcvtcolor(src_img, dst_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
                        IM_STATUS ret_rga = imcvtcolor(src_img, dst_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888, IM_YUV_TO_RGB_BT709_LIMIT);
                        if(ret_rga != IM_STATUS_SUCCESS){ 
                            log_error("rga failed: {}", imStrError(ret_rga)); 
                        }

                        VideoFrame frame_cb;
                        frame_cb.timestamp = mpp_frame_get_pts(frame);
                        if(!use_device_ptr_){
                            cv::Mat strided(height_, width_, CV_8UC3, image_ptr_, (size_t)dst_w_align * 3);
                            cv::Mat frame_mat;
                            strided.copyTo(frame_mat); // frame_mat.contiguous()==true，step = width*3，stride已被去掉
                            frame_cb.frame = std::move(frame_mat);
                            frame_cb.device_ptr = nullptr;
                            frame_cb.data_flag = 0;
                        }
                        else{
                            frame_cb.device_ptr = image_ptr_;
                            frame_cb.data_flag = 1;
                        }
                        uint64_t t2 = GetCurrentTimeMs();
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
                                    log_debug("input frame rate {} fps, image processing latency {} ms", tmp_frame_rate, t2 - t1);
                                    time_pre_ = time_now_;
                                    pre_frames_ = now_frames_;
                                }
                            }
                            callback_->OnRGBData(frame_cb);
                        }
                    }
                }
                frm_eos = mpp_frame_get_eos(frame);
                mpp_frame_deinit(&frame);
                get_frm = 1;
            }
            if(pkt_eos && pkt_done && !frm_eos){
                usleep(1 * 1000);
                continue;
            }
            if(frm_eos){
                break;
            }
            if(get_frm){
                continue;
            }
            break;
        }
        if(pkt_done){
            break;
        }
        usleep(1 * 1000);
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
    log_debug("DecodeThread Finished ");
    // 刷新缓冲区
    // HardDataNode *node = new HardDataNode();
    // node->es_data = NULL;
    // node->es_data_len = 0;
    // self->DecodeVideo(node);
    // delete node;
    return NULL;
}
#endif
