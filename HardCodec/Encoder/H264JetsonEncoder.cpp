#ifdef USE_NVIDIA_ARM
#include "H264HardEncoder.h"
#include "log_helpers.h"
#include <npp.h>
#ifndef CHECK_CUDA
#define CHECK_CUDA(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif
HardVideoEncoder::HardVideoEncoder()
{
    callback_ = nullptr;
    nframe_counter_ = 0;
    time_ts_accum_ = 0;
    time_inited_ = 0;
}
void HardVideoEncoder::SetDataCallback(EncDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}
HardVideoEncoder::~HardVideoEncoder()
{
    if(jetson_enc_obj_){
        delete jetson_enc_obj_;
        jetson_enc_obj_ = nullptr;
    }
    CHECK_CUDA(cudaFree(d_bgr_));
    CHECK_CUDA(cudaFree(d_rgb_));
    CHECK_CUDA(cudaFree(d_yuv_));

    log_info("~HardVideoEncoder");
}
int HardVideoEncoder::Init(cv::Mat bgr_frame, int fps)
{
    CHECK_CUDA(cudaSetDevice(device_id_));
    width_ = bgr_frame.cols;
    height_ = bgr_frame.rows;
    int frame_size_bgr = width_ * height_ * 3;
    CHECK_CUDA(cudaMalloc(&d_bgr_, frame_size_bgr));
    CHECK_CUDA(cudaMalloc(&d_rgb_, frame_size_bgr));
    int frame_size_yuv = width_ * height_ * 3 / 2;
    CHECK_CUDA(cudaMalloc(&d_yuv_, frame_size_yuv));
    jetson_enc_obj_ = new JetsonEnc(width_, height_, fps);
    jetson_enc_obj_->SetDecCallBack(static_cast<JetsonEncListner *>(this));
    return 0;
}
int HardVideoEncoder::AddVideoFrame(cv::Mat bgr_frame)
{
#ifdef DROP_FRAME
	// 丢帧处理
    if(jetson_enc_obj_->GetQueueSize() >= 5){
        return 0;
    }
#endif
    int frame_size_bgr = width_ * height_ * 3;
    CHECK_CUDA(cudaMemcpy(d_bgr_, bgr_frame.data, frame_size_bgr, cudaMemcpyHostToDevice));
    NppiSize roi = {width_, height_};
    int frame_size_yuv = width_ * height_ * 3 / 2;
    // BGR -> RGB
    const int order[3] = {2, 1, 0};
    NppStatus status = nppiSwapChannels_8u_C3R((const Npp8u *)d_bgr_,  width_ * 3, (Npp8u *)d_rgb_, width_ * 3, roi, order);
    if (status != NPP_SUCCESS) {
        log_error("NPP BGR->RGB failed: {}", (int)status);
        return -1;
    }
    // RGB -> NV12
    int dst_uv_step = width_;
    int frame_size_y = width_ * height_;
    int frame_size_uv = (width_ / 2) * (height_ / 2);
    Npp8u* dst_planes[3] = {d_yuv_, d_yuv_ + frame_size_y, d_yuv_ + frame_size_y + frame_size_uv};
    int dst_step[3] = {width_, width_/2, width_/2};
    status = nppiRGBToYUV420_8u_C3P3R((const Npp8u *)d_rgb_,  width_ * 3, dst_planes, dst_step, roi);
    if (status != NPP_SUCCESS) {
        log_error("NPP RGB->NV12 failed: {}", (int)status);
        return -1;
    }
    void *buffer = malloc(frame_size_yuv);
    CHECK_CUDA(cudaMemcpy(buffer, d_yuv_, frame_size_yuv, cudaMemcpyDeviceToHost));
    jetson_enc_obj_->AddFrame((unsigned char *)buffer, frame_size_yuv);
    // JetsonEnc中编码完会释放buffer，此处无需释放，这样做的目的是减少拷贝
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

    return 0;
}
void HardVideoEncoder::OnJetsonEncData(unsigned char *data, int data_len){
    time_now_ = std::chrono::steady_clock::now();
    if (nframe_counter_ - 1 == 0) {
        time_pre_ = time_now_;
        time_ts_accum_ = 0;
    }
    uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_now_ - time_pre_).count();
    time_ts_accum_ += duration;
    if (callback_) {
        callback_->OnVideoEncData(data, data_len, time_ts_accum_);
    }
    time_pre_ = time_now_;
}
#endif
