#ifdef USE_MPP
#include "H264HardEncoder.h"
#include "log_helpers.h"
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
HardVideoEncoder::HardVideoEncoder()
{
    abort_ = false;
    callback_ = NULL;
    nframe_counter_ = 0;
    time_ts_accum_ = 0;
    time_inited_ = 0;
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
    bgr_frames_.clear();
    CHECK_MPP(mpi_->reset(ctx_));
    CHECK_MPP(mpp_destroy(ctx_));
    CHECK_MPP(mpp_buffer_put(frm_buf_));
    CHECK_MPP(mpp_buffer_put(pkt_buf_));
    CHECK_MPP(mpp_buffer_group_put(buf_grp_));
    CHECK_MPP(mpp_enc_cfg_deinit(cfg_));
    log_debug("~HardVideoEncoder");
}

int HardVideoEncoder::Init(cv::Mat bgr_frame, int fps)
{
    width_ = bgr_frame.cols;
    height_= bgr_frame.rows;

    hor_stride_ = ALIGN(width_, 8) * 3;
    ver_stride_ = ALIGN(height_, 2);
    size_t frame_size = (size_t)ALIGN(hor_stride_, 64) * ALIGN(ver_stride_, 64);
    RK_S32 bps = width_ * height_ / 8 * fps;

    int ret = -1;

    CHECK_MPP(mpp_create(&ctx_, &mpi_));
    MppPollType timeout = MPP_POLL_BLOCK;
    CHECK_MPP(mpi_->control(ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout));
    CHECK_MPP(mpp_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC));

    CHECK_MPP(mpp_enc_cfg_init(&cfg_));
    CHECK_MPP(mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:width",  width_));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:height", height_));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", hor_stride_));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", ver_stride_));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:format", MPP_FMT_BGR888));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:mode", MPP_ENC_RC_MODE_CBR));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:bps_target", bps));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:bps_max", bps * 17 / 16));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:bps_min", bps * 15 / 16));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_flex", 1));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_num",  fps));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denorm", 1));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_flex", 1));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_num",  fps));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denorm", 1));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:gop", fps * 2));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "codec:type", MPP_VIDEO_CodingAVC));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "prep:range", MPP_FRAME_RANGE_JPEG));
    CHECK_MPP(mpp_enc_cfg_set_u32(cfg_, "rc:max_reenc_times", 0));
    CHECK_MPP(mpp_enc_cfg_set_u32(cfg_, "rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED));
    CHECK_MPP(mpp_enc_cfg_set_s32(cfg_, "rc:qp_max", 45));
    CHECK_MPP(mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_));

    MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
    CHECK_MPP(mpi_->control(ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode));

    CHECK_MPP(mpp_buffer_group_get_internal(&buf_grp_, MPP_BUFFER_TYPE_DRM));
    CHECK_MPP(mpp_buffer_get(buf_grp_, &frm_buf_, frame_size));
    CHECK_MPP(mpp_buffer_get(buf_grp_, &pkt_buf_, frame_size));
    return 1;
}
void *HardVideoEncoder::VideoEncThread(void *arg)
{
    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->bgr_mutex_);
        if (!self->bgr_frames_.empty()) {
            cv::Mat bgr_frame = self->bgr_frames_.front();
            self->bgr_frames_.pop_front();
            guard.unlock();
            self->nframe_counter_++;
            RK_U8 *dst = (RK_U8 *)mpp_buffer_get_ptr(self->frm_buf_);
            // CHECK_MPP(mpp_buffer_sync_begin(self->frm_buf_));
            for(RK_S32 y = 0; y < self->height_; y++){
                memcpy(dst + (size_t)y * self->hor_stride_, bgr_frame.data + (size_t)y * bgr_frame.step, self->width_ * 3);
            }
            // CHECK_MPP(mpp_buffer_sync_end(self->frm_buf_));
            MppFrame frame = NULL;
            CHECK_MPP(mpp_frame_init(&frame));
            mpp_frame_set_width(frame, self->width_);
            mpp_frame_set_height(frame, self->height_);
            mpp_frame_set_hor_stride(frame, self->hor_stride_);
            mpp_frame_set_ver_stride(frame, self->ver_stride_);
            mpp_frame_set_fmt(frame, MPP_FMT_BGR888);
            mpp_frame_set_buffer(frame, self->frm_buf_);
            mpp_frame_set_pts(frame, self->nframe_counter_);

            MppPacket packet = NULL;
            CHECK_MPP(mpp_packet_init_with_buffer(&packet, self->pkt_buf_));
            mpp_packet_set_length(packet, 0);
            mpp_meta_set_packet(mpp_frame_get_meta(frame), KEY_OUTPUT_PACKET, packet);

            if(self->mpi_->encode_put_frame(self->ctx_, frame)){ 
                mpp_frame_deinit(&frame);
                log_error("encode_put_frame error");
                continue;
            }
            mpp_frame_deinit(&frame);
            MppPacket out = NULL;
            if(self->mpi_->encode_get_packet(self->ctx_, &out)){
                mpp_packet_deinit(&out);
                log_error("encode_get_packet error");
                continue;
            }
            if(!out){
                continue;
            }

            self->time_now_ = std::chrono::steady_clock::now();
            if (self->nframe_counter_ - 1 == 0) {
                self->time_pre_ = self->time_now_;
                self->time_ts_accum_ = 0;
            }
            uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
            self->time_ts_accum_ += duration;
            if (self->callback_) {
                self->callback_->OnVideoEncData((unsigned char *)mpp_packet_get_pos(out), mpp_packet_get_length(out), self->time_ts_accum_);
            }
            self->time_pre_ = self->time_now_;
            
            mpp_packet_deinit(&out);
        } else {
            auto now = std::chrono::system_clock::now();
            self->bgr_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
        }
    }

    MppFrame frame = NULL;
    CHECK_MPP(mpp_frame_init(&frame));
    mpp_frame_set_width(frame, self->width_);
    mpp_frame_set_height(frame, self->height_);
    mpp_frame_set_hor_stride(frame, self->hor_stride_);
    mpp_frame_set_ver_stride(frame, self->ver_stride_);
    mpp_frame_set_fmt(frame, MPP_FMT_BGR888);
    mpp_frame_set_eos(frame, 1);
    if(self->mpi_->encode_put_frame(self->ctx_, frame)){ 
        mpp_frame_deinit(&frame);
        log_debug("VideoEncThread exit");
        return NULL;
    }
    mpp_frame_deinit(&frame);
    do {
        MppPacket out = NULL;
        if (self->mpi_->encode_get_packet(self->ctx_, &out)){
            break;
        }
        if(out){
            RK_U32 is_eos = mpp_packet_get_eos(out);
            if (mpp_packet_get_length(out)){
                self->time_now_ = std::chrono::steady_clock::now();
                if (self->nframe_counter_ - 1 == 0) {
                    self->time_pre_ = self->time_now_;
                    self->time_ts_accum_ = 0;
                }
                uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
                self->time_ts_accum_ += duration;
                if (self->callback_) {
                    self->callback_->OnVideoEncData((unsigned char *)mpp_packet_get_pos(out), mpp_packet_get_length(out), self->time_ts_accum_);
                }
                self->time_pre_ = self->time_now_;
            }
            mpp_packet_deinit(&out);
            if(is_eos){
                break;
            }
        }
    } while (1);
    log_debug("VideoEncThread exit");
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
