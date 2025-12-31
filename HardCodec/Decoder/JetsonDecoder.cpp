#ifdef USE_NVIDIA_ARM
#include "HardDecoder.h"
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
HardVideoDecoder::HardVideoDecoder(CODEC_TYPE codec_type)
{
    codec_type_ = codec_type;
    callback_ = nullptr;
    time_inited_ = 0;
    now_frames_ = pre_frames_ = 0;
    if(codec_type == CODEC_H264){
        decoder_pixfmt_ = V4L2_PIX_FMT_H264;
    }
    else if(codec_type == CODEC_H265){
        decoder_pixfmt_ = V4L2_PIX_FMT_H265;
    }
}
HardVideoDecoder::~HardVideoDecoder()
{
    if (jetson_dec_obj_) {
        delete jetson_dec_obj_;
        jetson_dec_obj_ = nullptr;
    }
    CHECK_CUDA(cudaFree(device_color_frame_));
    CHECK_CUDA(cudaFree(device_frame_));
    if(jetson_addr_){
        free(jetson_addr_);
        jetson_addr_ = nullptr;
    }
    if(host_frame_){
        free(host_frame_);
        host_frame_ = nullptr;
    }
    log_debug("~HardVideoDecoder");
}
void HardVideoDecoder::Init(int32_t device_id, int width, int height){
    device_id_ = device_id;
    width_ = width;
    height_ = height;
    CHECK_CUDA(cudaSetDevice(device_id_));
    CHECK_CUDA(cudaMalloc(&device_color_frame_, width_ * height_ * 3));
    host_frame_ = malloc(width_ * height_ * 3);
    jetson_addr_ = (unsigned char *)malloc(width_ * height_ * 3 / 2);
    CHECK_CUDA(cudaMalloc(&device_frame_, width_ * height_ * 3 / 2));
    jetson_dec_obj_ = new JetsonDec(decoder_pixfmt_, width_, height_, jetson_addr_);
    jetson_dec_obj_->SetDecCallBack(static_cast<JetsonDecListner *>(this));
    return;
}
void HardVideoDecoder::SetFrameFetchCallback(DecDataCallListner *call_func)
{
    callback_ = call_func;
    return;
}
static uint64_t GetCurrentTimeUs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_us);
}
void HardVideoDecoder::InputVideoData(unsigned char *data, int data_len, int64_t duration, int64_t pts)
{
    jetson_dec_obj_->AddEsData(data, data_len, GetCurrentTimeUs());
    return;
}
void HardVideoDecoder::OnJetsonDecData(unsigned char *data, int data_len, uint64_t timestamp){
    CHECK_CUDA(cudaMemcpy(device_frame_, jetson_addr_, width_ * height_ * 3 / 2, cudaMemcpyHostToDevice));
    const Npp8u* src_planes[2];
    src_planes[0] = (const Npp8u*)device_frame_;                     // Y
    src_planes[1] = (const Npp8u*)device_frame_ + width_ * height_;  // UV

    NppiSize roi_size;
    roi_size.width  = width_;
    roi_size.height = height_;

    NppStatus status = nppiNV12ToBGR_8u_P2C3R(src_planes, width_, (Npp8u*)device_color_frame_, width_ * 3, roi_size);
    if (status != NPP_SUCCESS) {
        log_error("NPP NV12->BGR failed: {}", (int)status);
        return;
    }
    CHECK_CUDA(cudaMemcpy(host_frame_, device_color_frame_, width_ * height_ * 3, cudaMemcpyDeviceToHost));
    cv::Mat frame_mat(height_, width_, CV_8UC3, host_frame_);
    cv::Mat frame_ret = frame_mat.clone();
    if (callback_ != nullptr) {
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
}
#endif
