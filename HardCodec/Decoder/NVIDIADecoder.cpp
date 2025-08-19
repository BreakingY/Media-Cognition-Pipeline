#ifdef USE_NVIDIA_X86
#include "HardDecoder.h"
#include <npp.h>
#include <atomic>
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
static CUcontext cuContext = NULL;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();
static void CreateCudaContext(CUcontext *cuContext, int iGpu, unsigned int flags)
{
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    log_debug("GPU in use: {}", szDeviceName);
    ck(cuCtxCreate(cuContext, flags, cuDevice));
    return;
}

HardVideoDecoder::HardVideoDecoder(bool is_h265)
{
    if(is_h265){
        type_ = cudaVideoCodec_HEVC;
    }
    else{
        type_ = cudaVideoCodec_H264;
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
    if(dec_){
        delete dec_;
        dec_ = NULL;
    }
    CHECK_CUDA(cudaFree(device_frame_));
    CHECK_CUDA(cudaFree(device_color_frame_));
    if(host_frame_){
        free(host_frame_);
        host_frame_ = NULL;
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
    CHECK_CUDA(cudaSetDevice(device_id_));
    static std::once_flag flag;
    std::call_once(flag, [this] {
        CreateCudaContext(&cuContext, this->device_id_, 0);
    });
    dec_ = new NvDecoder(cuContext, true, type_, true);
    CHECK_CUDA(cudaMalloc(&device_frame_, width_ * height_ * 4));
    CHECK_CUDA(cudaMalloc(&device_color_frame_, width_ * height_ * 3));
    host_frame_ = malloc(width_ * height_ * 3);
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
static uint64_t GetCurrentTimeUs()
{
    auto now = std::chrono::steady_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(time_us);
}
void HardVideoDecoder::DecodeVideo(HardDataNode *data)
{
    uint8_t *p_frame;
    int n_frame_returned = 0;
    n_frame_returned = dec_->Decode(data->es_data, data->es_data_len, CUVID_PKT_ENDOFPICTURE | CUVID_PKT_TIMESTAMP, GetCurrentTimeUs()); // CUVID_PKT_ENDOFPICTURE解码器立即输出，没有缓存，没有解码缓存时延;CUVID_PKT_TIMESTAMP返回原始时间戳
    int i_matrix = dec_->GetVideoFormatInfo().video_signal_description.matrix_coefficients;
    for (int i = 0; i < n_frame_returned; i++) {
        int64_t timestamp;
        p_frame = dec_->GetFrame(&timestamp);
        Nv12ToColor32<BGRA32>(p_frame, width_, (uint8_t *)device_frame_, width_ * 4, width_, height_, i_matrix);
        NppiSize roi_size = {width_, height_};
        const int order[3] = {0, 1, 2};
        NppStatus status = nppiSwapChannels_8u_C4C3R((const Npp8u*)device_frame_, width_ * 4, (Npp8u*)device_color_frame_, width_ * 3, roi_size, order);
        if(status != NPP_SUCCESS){
            log_error("NPP BGRA->BGR failed: {}", status);
        }
        CHECK_CUDA(cudaMemcpy(host_frame_, device_color_frame_, width_ * height_ * 3, cudaMemcpyDeviceToHost));
        cv::Mat frame_mat(height_, width_, CV_8UC3, host_frame_);
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
    }
    return;
}
void *HardVideoDecoder::DecodeThread(void *arg)
{
    HardVideoDecoder *self = (HardVideoDecoder *)arg;
    CHECK_CUDA(cudaSetDevice(self->device_id_));
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
#endif
