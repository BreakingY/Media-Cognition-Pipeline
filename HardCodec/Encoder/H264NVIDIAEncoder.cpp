#ifdef USE_NVIDIA_X86
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
static CUcontext cuContext = NULL;
// simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();
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
HardVideoEncoder::HardVideoEncoder()
{
    abort_ = false;
    callback_ = NULL;
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
    abort_ = true;
    encode_id_.join();
    bgr_frames_.clear();
    if(enc_){
        enc_->DestroyEncoder();
        delete enc_;
        enc_ = NULL;
    }
    CHECK_CUDA(cudaFree(ptr_image_bgr_device_));
    CHECK_CUDA(cudaFree(ptr_image_bgra_device_));
    log_info("~HardVideoEncoder");
}
void HardVideoEncoder::SetDevice(int device_id){
    device_id_ = device_id;
    return;
}

int HardVideoEncoder::Init(cv::Mat bgr_frame, int fps)
{
    CHECK_CUDA(cudaSetDevice(device_id_));
    static std::once_flag flag;
    std::call_once(flag, [this] {
        CreateCudaContext(&cuContext, this->device_id_, 0);
    });
    width_ = bgr_frame.cols;
    height_ = bgr_frame.rows;
    fps_ = fps;
    std::string param1 = "-codec h264 -preset p4 -profile baseline -tuninginfo ultralowlatency -bf 0 "; // 编码参数，根据需求自行修改
    std::string param2 = "-fps " + std::to_string(fps_) + " -gop " + std::to_string(2 * fps_) + " -bitrate " + std::to_string(4000000);
    std::string sz_param = param1 + param2;
    log_debug("nvidia enc params:{}", sz_param);
    CHECK_CUDA(cudaMalloc(&ptr_image_bgr_device_, width_ * height_ * 3));
    CHECK_CUDA(cudaMalloc(&ptr_image_bgra_device_, width_ * height_ * 4));
    eformat_ = NV_ENC_BUFFER_FORMAT_ARGB;
    init_param_ = NvEncoderInitParam(sz_param.c_str());
    enc_ = new NvEncoderCuda(cuContext, width_, height_, eformat_);
    NV_ENC_INITIALIZE_PARAMS initialize_params = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encode_config = {NV_ENC_CONFIG_VER};
    initialize_params.encodeConfig = &encode_config;
    enc_->CreateDefaultEncoderParams(&initialize_params, init_param_.GetEncodeGUID(), init_param_.GetPresetGUID(), init_param_.GetTuningInfo());
    init_param_.SetInitParams(&initialize_params, eformat_);
    enc_->CreateEncoder(&initialize_params);
    encode_id_ = std::thread(HardVideoEncoder::VideoEncThread, this);
    return 1;
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
void *HardVideoEncoder::VideoEncThread(void *arg)
{
    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    CHECK_CUDA(cudaSetDevice(self->device_id_));
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->bgr_mutex_);
        if (!self->bgr_frames_.empty()) {
            cv::Mat bgr_frame = self->bgr_frames_.front();
            self->bgr_frames_.pop_front();
            guard.unlock();
            self->nframe_counter_++;
            CHECK_CUDA(cudaMemcpy(self->ptr_image_bgr_device_, bgr_frame.data, self->width_ * self->height_ * 3, cudaMemcpyHostToDevice));
            NppiSize roi_size = {self->width_, self->height_};
            const Npp8u alpha_val = {255};  // alpha 通道填充值，一般取 255
            const int order[3] = {0, 1, 2};
            NppStatus status = nppiSwapChannels_8u_C3C4R((const Npp8u*)self->ptr_image_bgr_device_, self->width_ * 3, (Npp8u*)self->ptr_image_bgra_device_, self->width_ * 4, roi_size, order, alpha_val);
            if(status != NPP_SUCCESS){
                log_error("NPP BGR->BGRA failed: {}", status);
            }
            std::vector<std::vector<uint8_t>> vPacket;
            const NvEncInputFrame *encoder_input_frame = self->enc_->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, self->ptr_image_bgra_device_, 0, (CUdeviceptr)encoder_input_frame->inputPtr,
                                            (int)encoder_input_frame->pitch,
                                            self->enc_->GetEncodeWidth(),
                                            self->enc_->GetEncodeHeight(),
                                            CU_MEMORYTYPE_DEVICE, // CU_MEMORYTYPE_HOST,CU_MEMORYTYPE_DEVICE
                                            encoder_input_frame->bufferFormat,
                                            encoder_input_frame->chromaOffsets,
                                            encoder_input_frame->numChromaPlanes);
            self->enc_->EncodeFrame(vPacket);
            for (std::vector<uint8_t> &packet : vPacket) {
                self->time_now_ = std::chrono::steady_clock::now();
                if (self->nframe_counter_ - 1 == 0) {
                    self->time_pre_ = self->time_now_;
                    self->time_ts_accum_ = 0;
                }
                uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
                self->time_ts_accum_ += duration;
                if (self->callback_) {
                    self->callback_->OnVideoEncData((unsigned char *)packet.data(), packet.size(), self->time_ts_accum_);
                }
                self->time_pre_ = self->time_now_;
            }
            
        } else {
            auto now = std::chrono::system_clock::now();
            self->bgr_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
        }
    }
    std::vector<std::vector<uint8_t>> vPacket;
    self->enc_->EndEncode(vPacket);
    for (std::vector<uint8_t> &packet : vPacket) {
        self->time_now_ = std::chrono::steady_clock::now();
        if (self->nframe_counter_ - 1 == 0) {
            self->time_pre_ = self->time_now_;
            self->time_ts_accum_ = 0;
        }
        uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
        self->time_ts_accum_ += duration;
        if (self->callback_) {
            self->callback_->OnVideoEncData((unsigned char *)packet.data(), packet.size(), self->time_ts_accum_);
        }
        self->time_pre_ = self->time_now_;
    }
    log_info("VideoEncThread exit");
    return NULL;
}
#endif
