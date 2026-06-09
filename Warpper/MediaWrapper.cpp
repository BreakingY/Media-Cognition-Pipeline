#include "MediaWrapper.h"
MediaWrapper::MediaWrapper(const char *input, const char *output, const char *eng_path, int device_id)
{
    device_id_ = device_id;
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
    eng_path_ = eng_path;
    DetectModelInit(eng_path_, device_id_);
#endif
    size_t len = strlen(input);
    if( memcmp("rtsp://", input, strlen("rtsp://")) == 0 ){
        rtsp_client_proxy_ = new RtspClientProxy(input);
        rtsp_client_proxy_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else if( memcmp("rtmp://", input, strlen("rtmp://")) == 0 ){
        rtmp_pull_client_ = new RtmpPullClient(input, FLV_RTMP);
        rtmp_pull_client_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else if( len >= 4 && memcmp(input + len - 4, ".flv", 4) == 0 ){
        rtmp_pull_client_ = new RtmpPullClient(input, FLV_FILE);
        rtmp_pull_client_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else if( memcmp("srt://", input, strlen("srt://")) == 0 ){
        ts_demuxer_client_ = new TsDemuxerClient(input, TS_SRT);
        ts_demuxer_client_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else if( len >= 3 && memcmp(input + len - 3, ".ts", 3) == 0 ){
        ts_demuxer_client_ = new TsDemuxerClient(input, TS_FILE);
        ts_demuxer_client_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }
    else if( len >= 4 && memcmp(input + len - 4, ".mp4", 4) == 0 ){
        reader_ = new MediaReader(input);
        reader_->SetDataListner(static_cast<MediaDataListner *>(this), [this]() {
            return this->MediaOverhandle();
        });
    }

    len = strlen(output);
    if( memcmp("rtmp://", output, strlen("rtmp://")) == 0 ){
        rtmp_push_client_ = new RtmpPushClient(output, FLV_RTMP);
    }
    else if( len >= 4 && memcmp(output + len - 4, ".flv", 4) == 0 ){
        rtmp_push_client_ = new RtmpPushClient(output, FLV_FILE);
    }
    else if( len >= 4 && memcmp(output + len - 4, ".mp4", 4) == 0 ){
        mp4_muxer_ = new Muxer(output);
    }
    else if( len >= 3 && memcmp(output + len - 3, ".ts", 3) == 0 ){
        ts_muxer_client_ = new TsMuxerClient(output, TS_FILE);
    }
    else if( memcmp("srt://", output, strlen("srt://")) == 0 ){
        ts_muxer_client_ = new TsMuxerClient(output, TS_SRT);
    }
}
void MediaWrapper::MediaOverhandle()
{
    over_flag_ = true;
    return;
}
/**
 * 音视频解封装、解码
 */
// with startcode
void MediaWrapper::OnVideoData(VideoData data)
{
    if(rtsp_client_proxy_){ // rtsp
        video_type_ = rtsp_client_proxy_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
        rtsp_client_proxy_->GetVideoCon(width_, height_, fps_);
    }
    else if(rtmp_pull_client_){ // rtmp flv
        video_type_ = rtmp_pull_client_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
        rtmp_pull_client_->GetVideoCon(width_, height_, fps_);
    }
    else if(ts_demuxer_client_){ // srt ts
        video_type_ = ts_demuxer_client_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
        ts_demuxer_client_->GetVideoCon(width_, height_, fps_);
    }
    else if(reader_){ // mp4 
        video_type_ = reader_->GetVideoType();
        if (video_type_ == VIDEO_NONE) {
            log_error("only support H264/H265");
            exit(1);
        }
        reader_->GetVideoCon(width_, height_, fps_);
    }
    if (!hard_decoder_) {
        log_debug("video_type:{} width:{} height:{} fps_:{}", video_type_ == VIDEO_H264 ? "VIDEO_H264" : "VIDEO_H265", width_, height_, fps_);
        CODEC_TYPE type = CODEC_NONE;
        if(video_type_ == VIDEO_H264){
            type = CODEC_H264;
        }
        else if(video_type_ == VIDEO_H265){
            type = CODEC_H265;
        }
        hard_decoder_ = new HardVideoDecoder(type);
        hard_decoder_->SetFrameFetchCallback(static_cast<DecDataCallListner *>(this));
#if defined(USE_DVPP_MPI) || defined(USE_NVIDIA_X86) || defined(USE_NVIDIA_ARM)
        hard_decoder_->Init(device_id_, width_, height_); // dvpp nvidia
#endif
    }
    // int type;
    // if(video_type_ == VIDEO_H264){
    //     type = data.data[4] & 0x1f;
    // }
    // else{
    //     type = (data.data[4] >> 1) & 0x3f;
    // }
    hard_decoder_->InputVideoData(data.data, data.data_len, 0, 0); // 实时解码，不需要传递pts
    return;
}
// width adts
void MediaWrapper::OnAudioData(AudioData data)
{
    if(rtsp_client_proxy_){ // rtsp
        audio_type_ = rtsp_client_proxy_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    else if(rtmp_pull_client_){ // rtmp flv
        audio_type_ = rtmp_pull_client_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    else if(ts_demuxer_client_){ // srt ts
        audio_type_ = ts_demuxer_client_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    else if(reader_){ // mp4
        audio_type_ = reader_->GetAudioType();
        if (audio_type_ != AUDIO_AAC) {
            log_error("only support AAC");
            exit(1);
        }
    }
    if (aac_decoder_ == nullptr) {
        log_debug("audio_type:AAC profile:{} samplerate:{} channels:{}", data.profile, data.samplerate, data.channels);
        aac_decoder_ = new AACDecoder();
        aac_decoder_->SetResampleArg(AV_SAMPLE_FMT_S16, 2, 44100); // 重采样输出格式，解码器会把解码后的PCM数据重采样成设定的格式
        aac_decoder_->SetCallback(static_cast<DecDataCallListner *>(this));
    }
    aac_decoder_->InputAACData(data.data, data.data_len); // 实时解码，不需要传递pts
    return;
}

/**
 * 解码后音视频数据
 */
void MediaWrapper::OnRGBData(cv::Mat frame)
{
    // 拿到解码后的图像就可以根据自己的业务需求进行处理，例如：AI识别、opencv检测、图像渲染等。
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
    if(context_ == nullptr){
        context_ = AddStream(static_cast<InferDataListner *>(this), width_, height_, fps_);
    }
    StreamPushData(frame, context_);
    return;
#endif
    // 之后再把处理后的图像进行编码
    if (!hard_encoder_) {
#if defined(USE_NVIDIA_X86)
        if(use_nv_enc_flag_){
            hard_encoder_ = new NVHardVideoEncoder();
        }
        else{
            hard_encoder_ =  new NVSoftVideoEncoder();
        }
        hard_encoder_->SetDevice(device_id_);
#elif defined(USE_NVIDIA_ARM)
        hard_encoder_ = new HardVideoEncoder();
        hard_encoder_->SetDevice(device_id_);
#elif defined(USE_DVPP_MPI)
        hard_encoder_ = new HardVideoEncoder();
        hard_encoder_->SetDevice(device_id_);
#else
        hard_encoder_ = new HardVideoEncoder();
#endif
        hard_encoder_->Init(frame, fps_);
        hard_encoder_->SetDataCallback(static_cast<EncDataCallListner *>(this));
    }
    hard_encoder_->AddVideoFrame(frame);
    return;
}
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
static void DetectDraw(cv::Mat& img, DetectionInfo& info) {
    for (const auto& det : info.dets) {
        std::string label = "id:" + std::to_string(det.track_id);
        if (det.class_id >= 0 && det.class_id < info.class_names.size()) {
            label = info.class_names[det.class_id] + " " + label;
        }
        cv::Scalar color(0, 255, 0);
        if (det.class_id >= 0) {
            int c = det.class_id * 50;
            color = cv::Scalar(c % 255, (c * 2) % 255, (c * 3) % 255);
        }
        cv::rectangle(img, det.box, color, 2);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point text(det.box.x, std::max(0.0f, det.box.y - 5));

        cv::putText(img, label, text, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}
void MediaWrapper::OnInferData(cv::Mat& img, DetectionInfo& info){
    DetectDraw(img, info);
    if (!hard_encoder_) {
#if defined(USE_NVIDIA_X86)
        if(use_nv_enc_flag_){
            hard_encoder_ = new NVHardVideoEncoder();
        }
        else{
            hard_encoder_ =  new NVSoftVideoEncoder();
        }
        hard_encoder_->SetDevice(device_id_);
#elif defined(USE_NVIDIA_ARM)
        hard_encoder_ = new HardVideoEncoder();
        hard_encoder_->SetDevice(device_id_);
#elif defined(USE_DVPP_MPI)
        hard_encoder_ = new HardVideoEncoder();
        hard_encoder_->SetDevice(device_id_);
#else
        hard_encoder_ = new HardVideoEncoder();
#endif
        hard_encoder_->Init(img, fps_);
        hard_encoder_->SetDataCallback(static_cast<EncDataCallListner *>(this));
    }
    hard_encoder_->AddVideoFrame(img);
    return;
}
#endif
// FILE *fp_file = nullptr;
// data_len是单通道当本个数 LC-AAC:1024 HE-AAC:2048
void MediaWrapper::OnPCMData(unsigned char **data, int data_len)
{
    // 拿到解码后的PCM音频根据自己的业务需求进行处理，例如语音识别、语音合成等。
    // 之后再把处理后的音频进行编码
    if (aac_encoder_ == nullptr) {
        aac_encoder_ = new AACEncoder();
        // aac编码模块只接受packed模式的pcm数据
        // 和 aac_decoder_->SetResampleArg(AV_SAMPLE_FMT_S16,2,44100)保持一致即可，但如果aac_decoder_->SetResampleArg中指定了AV_SAMPLE_FMT_S16P,这里使用AV_SAMPLE_FMT_S16，数据就要转换成packed模型在送入队列
        aac_encoder_->Init(AV_SAMPLE_FMT_S16, 2 , 44100, data_len); // 输入格式，编码器会把PCM数据重采样成AAC编码器需要的格式然后进行编码
        aac_encoder_->SetCallback(static_cast<EncDataCallListner *>(this));
    }
    
    // 转换成packed在传送给aac编码模块
    enum AVSampleFormat dst_sample_fmt = AV_SAMPLE_FMT_S16;
    int dst_nb_channels = 2;
    int out_spb = av_get_bytes_per_sample(dst_sample_fmt);
    int buf_len = data_len * out_spb * dst_nb_channels;
    if (buffer_pcm_ == nullptr || (buffer_pcm_len_ < buf_len)) {
        buffer_pcm_ = (unsigned char *)realloc(buffer_pcm_, buf_len);
        buffer_pcm_len_ = buf_len;
    }
    int pos = 0;
    if (av_sample_fmt_is_planar(dst_sample_fmt)) { // plannar,dst_linesize=data_len*out_spb
        for (int i = 0; i < data_len; i++) {
            for (int c = 0; c < dst_nb_channels; c++)
                memcpy(buffer_pcm_ + pos, data[c] + i * out_spb, out_spb);
            pos += out_spb;
        }
    } else { // packed,dst_linesize=data_len*out_spb*out_channels
        memcpy(buffer_pcm_, data[0], data_len * out_spb * dst_nb_channels);
    }
    aac_encoder_->AddPCMFrame(buffer_pcm_, buf_len);
    // if (fp_file == nullptr) {
    //     fp_file = fopen("test.pcm", "wb+");
    // }
    // fwrite(buffer_pcm_, 1, buf_len, fp_file); // ffplay -ar 44100 -ac 2 -f s16le -i test.pcm
    return;
}
// static const char *enc_h264_filename = "out.h264";
// static FILE *enc_h264_fd = nullptr;
bool MediaWrapper::SetMediaInfo(){
    std::unique_lock<std::mutex> unique(media_info_mtx_);
    if((video_type_ == VIDEO_H264 || video_type_ == VIDEO_H265) && audio_type_ == AUDIO_AAC){
        if(mp4_muxer_){
            mp4_muxer_->SetMediaInfo(VideoType::VIDEO_H264, AudioType::AUDIO_AAC);
        }
        else if(ts_muxer_client_){
            ts_muxer_client_->SetVideoInfo(VideoType::VIDEO_H264);
            ts_muxer_client_->SetAudioInfo(AudioType::AUDIO_AAC);

        }
        return true;
    }
    bool have_video = false;
    bool have_audio = false;
    bool set_flag = false;
    if(reader_){
        // for file
        if(reader_->GetVideoType() == VIDEO_H264 || reader_->GetVideoType() == VIDEO_H265){
            have_video = true;
        }
        if(reader_->GetAudioType() == AUDIO_AAC){
            have_audio = true;
        }
        set_flag = true;
    }
    else if(rtsp_client_proxy_){
        // for sdp
        if(rtsp_client_proxy_->GetVideoType() == VIDEO_H264 || rtsp_client_proxy_->GetVideoType() == VIDEO_H265){
            have_video = true;
        }
        if(rtsp_client_proxy_->GetAudioType() == AUDIO_AAC){
            have_audio = true;
        }
        set_flag = true;
    }
    else if(rtmp_pull_client_){
        // need to probe
        time_now_ = std::chrono::steady_clock::now();
        nframe_counter_++;
        if (nframe_counter_ == 1) {
            time_pre_ = time_now_;
        }
        int64_t duration_t = std::chrono::duration_cast<std::chrono::seconds>(time_now_ - time_pre_).count();
        if(duration_t > 2){
            if(rtmp_pull_client_->GetVideoType() == VIDEO_H264 || rtmp_pull_client_->GetVideoType() == VIDEO_H265){
                have_video = true;
            }
            if(rtmp_pull_client_->GetAudioType() == AUDIO_AAC){
                have_audio = true;
            }
            set_flag = true;
        }
    }
    else if(ts_demuxer_client_){
        // need to probe
        time_now_ = std::chrono::steady_clock::now();
        nframe_counter_++;
        if (nframe_counter_ == 1) {
            time_pre_ = time_now_;
        }
        int64_t duration_t = std::chrono::duration_cast<std::chrono::seconds>(time_now_ - time_pre_).count();
        if(duration_t > 2){
            if(ts_demuxer_client_->GetVideoType() == VIDEO_H264 || ts_demuxer_client_->GetVideoType() == VIDEO_H265){
                have_video = true;
            }
            if(ts_demuxer_client_->GetAudioType() == AUDIO_AAC){
                have_audio = true;
            }
            set_flag = true;
        }
    }
    if(have_video && have_audio){
        if(mp4_muxer_){
            mp4_muxer_->SetMediaInfo(VideoType::VIDEO_H264, AudioType::AUDIO_AAC);
        }
        else if(ts_muxer_client_){
            ts_muxer_client_->SetVideoInfo(VideoType::VIDEO_H264);
            ts_muxer_client_->SetAudioInfo(AudioType::AUDIO_AAC);

        }
    }
    else if(have_video && !have_audio){
        if(mp4_muxer_){
            mp4_muxer_->SetMediaInfo(VideoType::VIDEO_H264, AudioType::AUDIO_NONE);
        }
        else if(ts_muxer_client_){
            ts_muxer_client_->SetVideoInfo(VideoType::VIDEO_H264);

        }
    }
    else if(!have_video && have_audio){
        if(mp4_muxer_){
            mp4_muxer_->SetMediaInfo(VideoType::VIDEO_NONE, AudioType::AUDIO_AAC);
        }
        else if(ts_muxer_client_){
            ts_muxer_client_->SetAudioInfo(AudioType::AUDIO_AAC);

        }
    }
    return set_flag;
}
void MediaWrapper::OnVideoEncData(unsigned char *data, int data_len, int64_t pts)
{
    // if (enc_h264_fd == nullptr) {
    //     enc_h264_fd = fopen(enc_h264_filename, "wb");
    // }
    // fwrite(data, 1, data_len, enc_h264_fd);
    if(rtmp_push_client_){
        rtmp_push_client_->SetVideoInfo(VideoType::VIDEO_H264);
        rtmp_push_client_->InputVideoData(data, data_len, pts);
        return;
    }

    if(!set_media_info_over_){
        if(!SetMediaInfo()){
            return;
        }
        set_media_info_over_ = true;
    }
    if(mp4_muxer_){
        mp4_muxer_->WriteVideo2File(data, data_len);
    }
    if(ts_muxer_client_){
        ts_muxer_client_->InputVideoData(data, data_len, pts);
    }
    return;
}
// static const char *enc_aac_filename = "out.aac";
// static FILE *enc_aac_fd = nullptr;
void MediaWrapper::OnAudioEncData(unsigned char *data, int data_len, int64_t pts)
{
    // if (enc_aac_fd == nullptr) {
    //     enc_aac_fd = fopen(enc_aac_filename, "wb");
    // }
    // fwrite(data, 1, data_len, enc_aac_fd);
    if(rtmp_push_client_){
        rtmp_push_client_->SetAudioInfo(AudioType::AUDIO_AAC);
        rtmp_push_client_->InputAudioData(data, data_len, pts);
        return;
    }

    if(!set_media_info_over_){
        if(!SetMediaInfo()){
            return;
        }
        set_media_info_over_ = true;
    }
    if(mp4_muxer_){
        mp4_muxer_->WriteAudio2File(data, data_len);
    }
    
    if(ts_muxer_client_){
        ts_muxer_client_->InputAudioData(data, data_len, pts);
    }
    return;
}
MediaWrapper::~MediaWrapper()
{

    if (reader_) {
        delete reader_;
        reader_ = nullptr;
    }
    if(rtsp_client_proxy_){
        delete rtsp_client_proxy_;
        rtsp_client_proxy_ = nullptr;
    }
    if(rtmp_pull_client_){
        delete rtmp_pull_client_;
        rtmp_pull_client_ = nullptr;
    }
    if(ts_demuxer_client_){
        delete ts_demuxer_client_;
        ts_demuxer_client_ = nullptr;
    }
    if (hard_decoder_) {
        delete hard_decoder_;
        hard_decoder_ = nullptr;
    }
    if (hard_encoder_) {
        delete hard_encoder_;
        hard_encoder_ = nullptr;
    }
    if (aac_decoder_) {
        delete aac_decoder_;
        aac_decoder_ = nullptr;
    }
    if (aac_encoder_) {
        delete aac_encoder_;
        aac_encoder_ = nullptr;
    }
    if (mp4_muxer_) {
        delete mp4_muxer_;
        mp4_muxer_ = nullptr;
    }
    if(rtmp_push_client_){
        delete rtmp_push_client_;
        rtmp_push_client_ = nullptr;
    }
    if(ts_muxer_client_){
        delete ts_muxer_client_;
        ts_muxer_client_ = nullptr;
    }
    if (buffer_pcm_) {
        free(buffer_pcm_);
        buffer_pcm_ = nullptr;
    }
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
    EndStream(context_);
#endif
    log_debug("~MediaWrapper");
}