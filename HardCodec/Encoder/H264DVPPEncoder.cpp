#ifdef USE_DVPP_MPI
#include "H264HardEncoder.h"
#include "log_helpers.h"

static const uint64 NANO_SECOND = UINT64_C(1000000000);
static std::atomic<int32_t> channel_id = {-1};
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %d in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %d in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
static int32_t GetChannedId(){
    if(channel_id >= MAX_ENC_CHANNEL_NUM){ // MAX_ENC_CHANNEL_NUM == 256
        channel_id = -1;
    }
    channel_id++;
    return channel_id;
}
HardVideoEncoder::HardVideoEncoder()
{
    abort_ = false;
    callback_ = NULL;
    nframe_counter_ = 0;
    nframe_counter_recv_ = 0;
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
    scale_id_.join();
    bgr_frames_.clear();
    yuv_frames_.clear();
    while (!out_buffer_pool_.empty()) {
        void* out_buffer = out_buffer_pool_.front();
        out_buffer_pool_.pop_front();
        CHECK_DVPP_MPI(hi_mpi_dvpp_free(out_buffer));
    }
    out_buffer_pool_.clear();
    CHECK_DVPP_MPI(hi_mpi_vpc_destroy_chn(channel_id_color_));
    if(in_img_buffer_){
        CHECK_DVPP_MPI(hi_mpi_dvpp_free(in_img_buffer_));
        in_img_buffer_ = NULL;
    }
    venc_mng_delete(enc_handle_);
    if(image_ptr_){
        free(image_ptr_);
        image_ptr_ = NULL;
    }
    log_info("~HardVideoEncoder");
}
void HardVideoEncoder::SetDevice(int device_id){
    device_id_ = device_id;
    return;
}
static int32_t CalcBitrate(int width, int height, int fps){
    int32_t bit_rate;
    int32_t frame_size = width * height;
    if (frame_size <= FRAME_SIZE_360P) {
        bit_rate = CALC_K * 1 + 1 * CALC_K * fps / DEFAULT_FRAME_RATE;
    } else if (frame_size <= FRAME_SIZE_720P) {
        bit_rate = CALC_K * 2 + 1 * CALC_K * fps / DEFAULT_FRAME_RATE; // 2:use to calculate bitrate
    } else if (frame_size <= FRAME_SIZE_1080P) {
        bit_rate = CALC_K * 2 + 2 * CALC_K * fps / DEFAULT_FRAME_RATE; // 2:use to calculate bitrate
    } else if (frame_size <= FRAME_SIZE_3K) {
        bit_rate = CALC_K * 3 + 3 * CALC_K * fps / DEFAULT_FRAME_RATE; // 3:use to calculate bitrate
    } else if (frame_size <= FRAME_SIZE_4K) {
        bit_rate = CALC_K * 5 + 5 * CALC_K * fps / DEFAULT_FRAME_RATE; // 5:use to calculate bitrate
    } else {
        bit_rate = CALC_K * 10 + 5 * CALC_K * fps / DEFAULT_FRAME_RATE; // 5,10:use to calculate bitrate
    }
    return bit_rate;
}
void vencStreamOut(uint32_t channelId, void* buffer, void *arg){
    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    hi_venc_stream* venc_stream = (hi_venc_stream*)buffer;
    int ret;
    for (int i = 0; i < venc_stream->pack_cnt; i++) {
        uint64_t data_len = venc_stream->pack[i].len - venc_stream->pack[i].offset;
        if(self->image_ptr_ == NULL){
            self->image_ptr_ = (unsigned char *)malloc(self->image_ptr_size_);
        }
        if(self->image_ptr_size_ < data_len){
            free(self->image_ptr_);
            self->image_ptr_size_ = data_len;
            self->image_ptr_ = (unsigned char *)malloc(self->image_ptr_size_);
        }
        ret = aclrtMemcpy(self->image_ptr_, data_len, venc_stream->pack[i].addr + venc_stream->pack[i].offset, data_len, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != HI_SUCCESS) {
            HMEV_HISDK_PRT(ERROR, "memcpy i:%u fail ret:%d pcData:%p,dataLen:%lu,addr:%p,offset:%d",
                i, ret, self->image_ptr_, data_len,
                venc_stream->pack[i].addr, venc_stream->pack[i].offset);
            return;
        }
        self->nframe_counter_recv_++;
        self->time_now_ = std::chrono::steady_clock::now();
        if (self->nframe_counter_recv_ - 1 == 0) {
            self->time_pre_ = self->time_now_;
            self->time_ts_accum_ = 0;
        }
        uint64_t duration = = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
        self->time_ts_accum_ += duration;
        if (self->callback_) {
            self->callback_->OnVideoEncData(self->image_ptr_, data_len, self->time_ts_accum_);
        }
        self->time_pre_ = self->time_now_;
    }
    return;
}
void HardVideoEncoder::InitEncParams(VencParam* encParam)
{
    encParam->codecType = codec_type_;
    encParam->encCallback.encDateOutProcess = vencStreamOut;
    encParam->encCallback.arg = this;
    encParam->rcMode = HMEV_RC_VBR;
    encParam->frameNum = -1;
    encParam->fixQp = FIX_QP;
    encParam->inputFrameLen = 1; // INPUT_FRAME_LEN; // 编码器缓存帧数
    if(codec_type_ == VENC_CODEC_TYPE_H264){
        encParam->profile = 0; // h.264 -- 0:Base 1:Main 2:High
    }
    else if(codec_type_ == VENC_CODEC_TYPE_H265){
        encParam->profile = 0; // h.265 -- 0
    }
    encParam->maxQp = 51;
    encParam->minQp = 20;
    encParam->maxIQp = 51;
    encParam->minIQp = 20;
    encParam->frameSize.width = width_;
    encParam->frameSize.height = height_;
    encParam->bitRate = 4 * 1000;
    encParam->frameRate = fps_;
    encParam->userSetDisplayRate = false;
    encParam->displayRate = fps_;
    encParam->frameGop = 2 * fps_;
    return;
}
int HardVideoEncoder::Init(cv::Mat bgr_frame, int fps)
{
    CHECK_ACL(aclrtSetDevice(device_id_));
    width_ = bgr_frame.cols;
    height_ = bgr_frame.rows;
    fps_ = fps;
    // color
    in_img_buffer_size_ = bgr_frame.cols * bgr_frame.rows * 3;
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(device_id_, &in_img_buffer_, in_img_buffer_size_));
    out_buffer_size_ = width_ * height_ * 3 / 2; // YUV420P
    for (uint32_t i = 0; i < pool_num_; i++) {
        void* out_buffer = NULL;
        CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(device_id_, &out_buffer, out_buffer_size_));
        out_buffer_pool_.push_back(out_buffer);
    }
    hi_vpc_chn_attr st_chn_attr {};
    st_chn_attr.attr = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_sys_create_chn(&channel_id_color_, &st_chn_attr));
    input_pic_.picture_width = width_;
    input_pic_.picture_height = height_;
    input_pic_.picture_format = in_format_;
    input_pic_.picture_width_stride = width_ * 3;
    input_pic_.picture_height_stride = height_;
    input_pic_.picture_buffer_size = width_ * height_ * 3;

    output_pic_.picture_width = width_;
    output_pic_.picture_height = height_;
    output_pic_.picture_format = out_format_;
    output_pic_.picture_width_stride = width_;
    output_pic_.picture_height_stride = height_;
    output_pic_.picture_buffer_size = width_ * height_ * 3 / 2;

    // enc
    bit_rate_ = CalcBitrate(width_, height_, fps_);
    enc_channel_ = GetChannedId();
    codec_type_ = VENC_CODEC_TYPE_H264;
    InitEncParams(&enc_param_);
    enc_param_.channelId = enc_channel_;
    enc_param_.highPriority = 0; // 0:普通、非0：高优先级
    int32_t ret = venc_mng_create(&enc_handle_, &enc_param_, device_id_);
    HMEV_HISDK_CHECK_RET_EXPRESS(ret != HMEV_SUCCESS, "venc_mng_create fail!");

    scale_id_ = std::thread(HardVideoEncoder::VideoScaleThread, this);
    encode_id_ = std::thread(HardVideoEncoder::VideoEncThread, this);
    return 1;
}
void *HardVideoEncoder::GetColorAddr(){
    while(!abort_){
        std::unique_lock<std::mutex> guard(out_buffer_pool_mutex_);
        if (!out_buffer_pool_.empty()) {
            void *addr = out_buffer_pool_.front();
            out_buffer_pool_.pop_front();
            guard.unlock();
            return addr;
        } else {
            auto now = std::chrono::system_clock::now();
            out_buffer_pool_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    return NULL;
}
void HardVideoEncoder::PutColorAddr(void *addr){
    if(!addr){
        return;
    }
    std::unique_lock<std::mutex> guard(out_buffer_pool_mutex_);
    out_buffer_pool_.push_back(addr);
    guard.unlock();
    out_buffer_pool_cond_.notify_one();
    return;
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
void *HardVideoEncoder::VideoScaleThread(void *arg)
{
    HardVideoEncoder *self = (HardVideoEncoder *)arg;
    CHECK_ACL(aclrtSetDevice(self->device_id_));
    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->bgr_mutex_);
        if (!self->bgr_frames_.empty()) {
            cv::Mat bgr_frame = self->bgr_frames_.front();
            self->bgr_frames_.pop_front();
            guard.unlock();
            void *addr = self->GetColorAddr();
            CHECK_ACL(aclrtMemcpy(self->in_img_buffer_ , self->in_img_buffer_size_, bgr_frame.data, self->in_img_buffer_size_, ACL_MEMCPY_HOST_TO_DEVICE));
            self->input_pic_.picture_address = self->in_img_buffer_;
            self->output_pic_.picture_address = addr;
            uint32_t task_id;
            CHECK_DVPP_MPI(hi_mpi_vpc_convert_color(self->channel_id_color_, &self->input_pic_, &self->output_pic_, &task_id, -1));
            CHECK_DVPP_MPI(hi_mpi_vpc_get_process_result(self->channel_id_color_, task_id, -1));

            std::unique_lock<std::mutex> guard(self->yuv_mutex_);
#ifdef DROP_FRAME
			// 丢帧处理
            if (self->yuv_frames_.size() > 5) {
                for (std::list<void *>::iterator it = self->yuv_frames_.begin();
                    it != self->yuv_frames_.end(); ++it) {
                    void *frame = *it;
                    self->PutColorAddr(frame);
                }
                self->yuv_frames_.clear();
            }
#endif
            self->yuv_frames_.push_back(addr);
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
    CHECK_ACL(aclrtSetDevice(self->device_id_));
    hi_char ac_thread_name[32] = {0};
    snprintf(ac_thread_name, sizeof(ac_thread_name), "HmevVencSendFrame");
    prctl(PR_SET_NAME, (unsigned long)ac_thread_name);

    EncoderHandle* enc_handle = venc_mng_get_handle(self->enc_channel_);
    std::shared_ptr<HmevEncoder> enc(enc_handle->encoder);
    hi_pixel_format pixel_format = self->out_format_;
    hi_data_bit_width bit_width = HI_DATA_BIT_WIDTH_8; 
    hi_compress_mode cmp_mode = HI_COMPRESS_MODE_NONE;
    hi_u32 align = DEFAULT_ALIGN;
    hi_video_frame_info* video_frame_info = NULL;

    while (!self->abort_) {
        std::unique_lock<std::mutex> guard(self->yuv_mutex_);
        if (!self->yuv_frames_.empty()) {
            void *yuv_frame = self->yuv_frames_.front();
            self->yuv_frames_.pop_front();
            guard.unlock();
            int ret = enc->dequeue_input_buffer(self->width_, self->height_, pixel_format, bit_width, cmp_mode, align, &video_frame_info);
            if (ret != HMEV_SUCCESS) {
                HMEV_HISDK_PRT(DEBUG, "dequeue_input_buffer fail");
                std::this_thread::sleep_for(std::chrono::microseconds(2000)); // sleep 2000 us
                continue;
            }
            CHECK_ACL(aclrtMemcpy(video_frame_info->v_frame.virt_addr[0] , self->out_buffer_size_, yuv_frame, self->out_buffer_size_, ACL_MEMCPY_DEVICE_TO_DEVICE));
            video_frame_info->v_frame.time_ref = self->nframe_counter_ * 2;
            ret = venc_mng_process_buffer((IHWCODEC_HANDLE)enc_handle, (void*)video_frame_info);
            if (ret != HMEV_SUCCESS) {
                HMEV_HISDK_PRT(ERROR, "Chn[%d] hi_mpi_venc_send_frame failed, s32Ret:0x%x\n", self->enc_channel_, ret);
                enc->cancel_frame(video_frame_info);
                continue;
            }
            self->nframe_counter_++;
            self->PutColorAddr(yuv_frame);

        } else {
            auto now = std::chrono::system_clock::now();
            self->yuv_cond_.wait_until(guard, now + std::chrono::milliseconds(100));
            guard.unlock();
            continue;
        }
    }
    log_info("VideoEncThread exit");
    return NULL;
}
#endif
