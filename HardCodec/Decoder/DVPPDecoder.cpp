#ifdef USE_DVPP_MPI
#include "HardDecoder.h"
#include <atomic>
static const uint64_t NANO_SECOND = UINT64_C(1000000000);
static std::atomic<int32_t> channel_id = {-1};
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
static int32_t GetChannedId(){
    if(channel_id >= VDEC_MAX_CHN_NUM){ // VDEC_MAX_CHN_NUM == 256
        channel_id = -1;
    }
    channel_id++;
    return channel_id;
}
HardVideoDecoder::HardVideoDecoder(bool is_h265)
{
    if(is_h265){
        chn_attr_.type = HI_PT_H265;
    }
    else{
        chn_attr_.type = HI_PT_H264;
    }
    callback_ = NULL;
    time_inited_ = 0;
    now_frames_ = pre_frames_ = 0;
}
HardVideoDecoder::~HardVideoDecoder()
{
    Stop();
    for (std::list<HardDataNode *>::iterator it = es_packets_.begin(); it != es_packets_.end(); ++it){
        HardDataNode *packet = *it;
        delete packet;
    }
    es_packets_.clear();
    log_debug("~HardVideoDecoder");
}
void HardVideoDecoder::Stop(){
    CHECK_ACL(aclrtSetDevice(device_id_));
    abort_ = true;
    send_stream_thread_id_.join();
    get_pic_thread_id_.join();
    CHECK_DVPP_MPI(hi_mpi_vdec_stop_recv_stream(channel_id_));
    CHECK_DVPP_MPI(hi_mpi_vdec_destroy_chn(channel_id_));
    while (!out_buffer_pool_.empty()) {
        void* out_buffer = out_buffer_pool_.front();
        out_buffer_pool_.pop_front();
        CHECK_DVPP_MPI(hi_mpi_dvpp_free(out_buffer));
    }
    out_buffer_pool_.clear();
    if(in_es_buffer_){
        CHECK_DVPP_MPI(hi_mpi_dvpp_free(in_es_buffer_));
        in_es_buffer_ = NULL;
    }
    CHECK_DVPP_MPI(hi_mpi_vpc_destroy_chn(channel_id_color_));
    if(output_pic_.picture_address){
        CHECK_DVPP_MPI(hi_mpi_dvpp_free(output_pic_.picture_address));
        output_pic_.picture_address = NULL;
    }
    if(image_ptr_){
        free(image_ptr_);
        image_ptr_ = NULL;
    }
    return;

}
void HardVideoDecoder::Init(int32_t device_id, int width, int height){
    device_id_ = device_id;
    width_ = width;
    height_ = height;
    CHECK_ACL(aclrtSetDevice(device_id_));
    CHECK_DVPP_MPI(hi_mpi_sys_init());
    chn_attr_.mode = HI_VDEC_SEND_MODE_FRAME; // Only support frame mode
    chn_attr_.pic_width = width;
    chn_attr_.pic_height = height;
    chn_attr_.stream_buf_size = width * height * 3 / 2;
    chn_attr_.frame_buf_cnt = 1;
    hi_pic_buf_attr buf_attr{(hi_u32)width, (hi_u32)height, 0, bit_width_, out_format_, HI_COMPRESS_MODE_NONE};
    chn_attr_.frame_buf_size = hi_vdec_get_pic_buf_size(chn_attr_.type, &buf_attr);
    chn_attr_.video_attr.ref_frame_num = 1;
    chn_attr_.video_attr.temporal_mvp_en = HI_TRUE;
    chn_attr_.video_attr.tmv_buf_size = hi_vdec_get_tmv_buf_size(chn_attr_.type, width, height);
    channel_id_ = GetChannedId();
    CHECK_DVPP_MPI(hi_mpi_vdec_create_chn(channel_id_, &chn_attr_));

    hi_vdec_chn_param chn_param;
    CHECK_DVPP_MPI(hi_mpi_vdec_get_chn_param(channel_id_, &chn_param));
    chn_param.video_param.dec_mode = HI_VIDEO_DEC_MODE_IPB;
    chn_param.video_param.compress_mode = HI_COMPRESS_MODE_HFBC;
    chn_param.video_param.video_format = HI_VIDEO_FORMAT_TILE_64x16;
    chn_param.display_frame_num = 1; 
    chn_param.video_param.out_order = HI_VIDEO_OUT_ORDER_DISPLAY; // Display sequence
    CHECK_DVPP_MPI(hi_mpi_vdec_set_chn_param(channel_id_, &chn_param));
    CHECK_DVPP_MPI(hi_mpi_vdec_start_recv_stream(channel_id_));

    out_buffer_size_ = width * height * 3 / 2; // YUV420P
    for (uint32_t i = 0; i < pool_num_; i++) {
        void* out_buffer = NULL;
        CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(device_id_, &out_buffer, out_buffer_size_));
        out_buffer_pool_.push_back(out_buffer);
    }

    // color convert
    hi_vpc_chn_attr st_chn_attr {};
    st_chn_attr.attr = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_sys_create_chn(&channel_id_color_, &st_chn_attr));
    input_pic_.picture_width = width_;
    input_pic_.picture_height = height_;
    input_pic_.picture_format = out_format_;
    input_pic_.picture_width_stride = width_;
    input_pic_.picture_height_stride = height_;
    input_pic_.picture_buffer_size = width_ * height_ * 3 / 2;

    output_pic_.picture_width = width_;
    output_pic_.picture_height = height_;
    output_pic_.picture_format = out_format_color_;
    output_pic_.picture_width_stride = width_ * 3;
    output_pic_.picture_height_stride = height_;
    output_pic_.picture_buffer_size = width_ * height_ * 3;
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(device_id_, &output_pic_.picture_address, output_pic_.picture_buffer_size));

    
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(device_id_, &in_es_buffer_, in_es_buffer_size_));
    send_stream_thread_id_ = std::thread(HardVideoDecoder::SendStream, this);
    get_pic_thread_id_ = std::thread(HardVideoDecoder::GetPic, this);
    return;
}
void *HardVideoDecoder::GetOutAddr(){
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
void HardVideoDecoder::PutOutAddr(void *addr){
    if(!addr){
        return;
    }
    std::unique_lock<std::mutex> guard(out_buffer_pool_mutex_);
    out_buffer_pool_.push_back(addr);
    guard.unlock();
    out_buffer_pool_cond_.notify_one();
    return;
}
void HardVideoDecoder::VdecResetChn(){
    CHECK_DVPP_MPI(hi_mpi_vdec_stop_recv_stream(channel_id_));
    CHECK_DVPP_MPI(hi_mpi_vdec_reset_chn(channel_id_));
    CHECK_DVPP_MPI(hi_mpi_vdec_start_recv_stream(channel_id_));
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

void HardVideoDecoder::DecodeVideo(HardDataNode *data){
    hi_vdec_stream stream;
    hi_vdec_pic_info out_pic_info;
    if(data->es_data == NULL){
        stream.addr = NULL;
        stream.len = 0;
        stream.end_of_frame = HI_FALSE;
        stream.end_of_stream = HI_TRUE; // Stream end flage
        out_pic_info.vir_addr = 0;
        out_pic_info.buffer_size = 0;
        hi_mpi_vdec_send_stream(channel_id_, &stream, &out_pic_info, -1);
        return;
    }
    stream.pts = GetCurrentTimeUs();
    stream.addr = (hi_u8*)in_es_buffer_; // Configure input stream address
    CHECK_ACL(aclrtMemcpy(stream.addr, in_es_buffer_size_, data->es_data, data->es_data_len, ACL_MEMCPY_HOST_TO_DEVICE));
    stream.len = data->es_data_len; // Configure input stream size
    stream.end_of_frame = HI_TRUE; // Configure flage of frame end
    stream.end_of_stream = HI_FALSE; // Configure flage of stream end

    out_pic_info.width = width_; // Output image width, supports resize, set 0 means no resize
    out_pic_info.height = height_; // Output image height, supports resize, set 0 means no resize
    out_pic_info.width_stride = width_; // Output memory width stride
    out_pic_info.height_stride = height_; // Output memory height stride
    out_pic_info.pixel_format = out_format_; // Configure output format

    stream.need_display = HI_TRUE;
    out_pic_info.vir_addr = (hi_u64)GetOutAddr();
    out_pic_info.buffer_size = out_buffer_size_;
    int ret = hi_mpi_vdec_send_stream(channel_id_, &stream, &out_pic_info, -1);
    if(ret != HI_SUCCESS){
        log_error("hi_mpi_vdec_send_stream error");
    }
    return;
}
void *HardVideoDecoder::SendStream(void *arg){
    HardVideoDecoder *self = (HardVideoDecoder*)arg;
    CHECK_ACL(aclrtSetDevice(self->device_id_));
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
    // 刷新缓冲区
    HardDataNode *node = new HardDataNode();
    node->es_data = NULL;
    node->es_data_len = 0;
    self->DecodeVideo(node);
    delete node;
    log_info("SendStream Finished");
    return NULL;
}
void *HardVideoDecoder::GetPic(void *arg){
    HardVideoDecoder *self = (HardVideoDecoder*)arg;
    CHECK_ACL(aclrtSetDevice(self->device_id_));
    hi_video_frame_info frame;
    hi_vdec_stream stream;
    hi_vdec_supplement_info st_supplement{};
    int ret;
    while (!self->abort_) {
        ret = hi_mpi_vdec_get_frame(self->channel_id_, &frame, &st_supplement, &stream, 1000);
        void *output_buffer = NULL;
        if(ret == HI_SUCCESS){       
            output_buffer = (void*)frame.v_frame.virt_addr[0];
            int32_t dec_result = frame.v_frame.frame_flag;
            if((dec_result == 0) && (output_buffer != NULL)){ // get frame
                uint64_t pts = frame.v_frame.pts; // stream.pts
                // color convert
                self->input_pic_.picture_address = output_buffer;
                uint32_t task_id;
                CHECK_DVPP_MPI(hi_mpi_vpc_convert_color(self->channel_id_color_, &self->input_pic_, &self->output_pic_, &task_id, -1));
                CHECK_DVPP_MPI(hi_mpi_vpc_get_process_result(self->channel_id_color_, task_id, -1));
                int size = self->width_ * self->height_ * 3;
                if(!self->image_ptr_){
                    self->image_ptr_ = (unsigned char *)malloc(size);
                }
                CHECK_ACL(aclrtMemcpy(self->image_ptr_ , size, self->output_pic_.picture_address, size, ACL_MEMCPY_DEVICE_TO_HOST));
                cv::Mat frame_mat(self->height_, self->width_, CV_8UC3, self->image_ptr_);
                cv::Mat frame_ret = frame_mat.clone();
                if (self->callback_ != NULL) {
                    self->now_frames_++;
                    if (!self->time_inited_) {
                        self->time_inited_ = 1;
                        self->time_now_ = std::chrono::steady_clock::now();
                        self->time_pre_ = self->time_now_;
                    } else {
                        self->time_now_ = std::chrono::steady_clock::now();
                        long tmp_time = std::chrono::duration_cast<std::chrono::milliseconds>(self->time_now_ - self->time_pre_).count();
                        if (tmp_time > 1000) { // 1s
                            int tmp_frame_rate = (self->now_frames_ - self->pre_frames_ + 1) * 1000 / tmp_time;
                            log_debug("input frame rate {} ", tmp_frame_rate);
                            self->time_pre_ = self->time_now_;
                            self->pre_frames_ = self->now_frames_;
                        }
                    }
                    self->callback_->OnRGBData(frame_ret);
                }
            }
            if(output_buffer != NULL){
                self->PutOutAddr(output_buffer);
            }
            CHECK_DVPP_MPI(hi_mpi_vdec_release_frame(self->channel_id_, &frame));
        }
    }
    log_info("GetPic Finished");
    return NULL;
}
#endif
