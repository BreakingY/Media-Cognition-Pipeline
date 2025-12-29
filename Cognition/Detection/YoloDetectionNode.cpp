#define DETECTION_NVIDIA
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#include "YoloDetectionNode.h"
class Logger: public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if(severity <= Severity::kWARNING)
            log_error("{}", msg);
    }
}logger;
#define MY_ASSERT(x)                     \
    do {                                \
        if (!(x)) {                     \
            std::abort();               \
        }                               \
    } while (0)
static char* ReadFromPath(std::string eng_path,int &model_size){
    std::ifstream file(eng_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << eng_path << " error!" << std::endl;
        return nullptr;
    }
    char *trt_model_stream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if(!trt_model_stream){
        return nullptr;
    }
    file.read(trt_model_stream, size);
    file.close();
    model_size = size;
    return trt_model_stream;
}
static size_t CountElement(const nvinfer1::Dims &dims, int batch_zise)
{
    int64_t total = batch_zise;
    for (int32_t i = 1; i < dims.nbDims; ++i){
        total *= dims.d[i];
    }
    return static_cast<size_t>(total);
}
// 返回值: tuple<cv::Mat, float, float, float> -> (resized_img, scale_ratio, dw, dh)
static std::tuple<float, float, float> Letterbox_resize_GPU(int orig_h, int orig_w, void *img_buffer, void *out_buffer,int new_h, int new_w, const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);
    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));
    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    Npp8u *pu8_src = static_cast<Npp8u*>(img_buffer);
    Npp8u *pu8_dst = static_cast<Npp8u*>(out_buffer);

    Npp8u color_array[3] = {(Npp8u)color[0], (Npp8u)color[1], (Npp8u)color[2]};
    NppiSize dst_size{new_w, new_h};
    NppStatus ret = nppiSet_8u_C3R(color_array, pu8_dst, new_w * 3, dst_size);
    if(ret != 0){
        log_error("nppiSet_8u_C3R error: {}", ret);
        return std::make_tuple(r, dw, dh);
    }
    Npp8u *pu8_resized = nullptr;
    CHECK_CUDA(cudaMalloc(&pu8_resized, new_unpad_h * new_unpad_w * 3));

    NppiSize src_size{orig_w, orig_h};
    NppiRect src_roi{0,0,orig_w,orig_h};
    NppiSize resize_size{new_unpad_w, new_unpad_h};
    NppiRect dst_roi{0,0,new_unpad_w,new_unpad_h};

    ret = nppiResize_8u_C3R(pu8_src, orig_w * 3, src_size, src_roi, pu8_resized, new_unpad_w * 3, resize_size, dst_roi, NPPI_INTER_LINEAR);
    if(ret != 0){
        log_error("nppiResize_8u_C3R error: {}", ret);
        CHECK_CUDA(cudaFree(pu8_resized));
        return std::make_tuple(r, dw, dh);
    }
    NppiSize copy_size{new_unpad_w, new_unpad_h};
    ret = nppiCopy_8u_C3R(pu8_resized, new_unpad_w * 3, pu8_dst + top * new_w * 3 + left * 3, new_w * 3, copy_size);
    if(ret != 0){
        log_error("nppiCopy_8u_C3R error: {}", ret);
        CHECK_CUDA(cudaFree(pu8_resized));
        return std::make_tuple(r, dw, dh);
    }

    CHECK_CUDA(cudaFree(pu8_resized));
#if 0
    cv::Mat img_cpu(new_h, new_w, CV_8UC3);
    
    size_t bytes = new_w * new_h * 3 * sizeof(Npp8u);
    CHECK_CUDA(cudaMemcpy(img_cpu.data, out_buffer, bytes, cudaMemcpyDeviceToHost));
    if(!cv::imwrite("output.jpg", img_cpu)){
        log_error("Failed to save image");
    } 
#endif
    return std::make_tuple(r, dw, dh);
}
static std::tuple<float, float, float>  PreprocessImage_GPU(cv::Mat &img, void *buffer, int channel, int input_h, int input_w, cudaStream_t stream){
    void *img_buffer = nullptr;
    int orig_h = img.rows;
    int orig_w = img.cols;
    CHECK_CUDA(cudaMalloc(&img_buffer, orig_h * orig_w * 3));
    void *img_ptr = img.data;
    CHECK_CUDA(cudaMemcpyAsync(img_buffer, img_ptr, orig_h * orig_w * 3, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::tuple<float, float, float> res = Letterbox_resize_GPU(orig_h, orig_w, img_buffer, buffer, input_h, input_w);

    float &r = std::get<0>(res);
    float &dw = std::get<1>(res);
    float &dh = std::get<2>(res);
    
    Npp8u *pu8_rgb = nullptr;
    CHECK_CUDA(cudaMalloc(&pu8_rgb, input_h * input_w * 3));
    // BGR-->RGB
    int aOrder[3] = {2, 1, 0};
    NppiSize size = {input_w, input_h};
    NppStatus ret = nppiSwapChannels_8u_C3R((Npp8u*)buffer, input_w * 3, pu8_rgb, input_w * 3, size, aOrder);
    if(ret != 0){
        log_error("nppiSwapChannels_8u_C3R error: {}", ret);
    }

    // 转 float 并归一化
    NppiSize fsize = {input_w, input_h};
    ret = nppiConvert_8u32f_C3R(pu8_rgb, input_w * 3, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        log_error("nppiConvert_8u32f_C3R error: {}", ret);
    }
    Npp32f aConstants[3] = {1.f / 255.f, 1.f / 255.f,1.f / 255.f};
    ret = nppiMulC_32f_C3IR(aConstants, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        log_error("nppiMulC_32f_C3IR error: {}", ret);
    }

    // HWC TO CHW
    NppiSize chw_size = {input_w, input_h};
    float* buffer_chw = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer_chw, input_h * input_w * 3 * sizeof(float)));
    Npp32f* dst_planes[3];
    dst_planes[0] = (Npp32f*)buffer_chw;                           // R
    dst_planes[1] = (Npp32f*)buffer_chw + input_h * input_w;       // G
    dst_planes[2] = (Npp32f*)buffer_chw + input_h * input_w * 2;   // B
    ret = nppiCopy_32f_C3P3R((Npp32f*)buffer, input_w * 3 * sizeof(float), dst_planes, input_w * sizeof(float), chw_size);
    if (ret != 0) {
        std::cerr << "nppiCopy_32f_C3P3R error: " << ret << std::endl;
    }
    CHECK_CUDA(cudaMemcpy(buffer, buffer_chw, input_h * input_w * 3 * sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(buffer_chw));
    CHECK_CUDA(cudaFree(img_buffer));
    CHECK_CUDA(cudaFree(pu8_rgb));
    return std::move(std::make_tuple(r, dw, dh));
}
static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

static std::vector<int> NMS(const std::vector<Detection>& dets, float iou_thres) {
    std::vector<int> order(dets.size());
    for (size_t idx = 0; idx < order.size(); ++idx) {
        order[idx] = static_cast<int>(idx);
    }
    std::sort(order.begin(), order.end(), [&](int i, int j){
        return dets[i].score > dets[j].score;
    });

    std::vector<int> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t _i = 0; _i < order.size(); ++_i) {
        int i = order[_i];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < order.size(); ++_j) {
            int j = order[_j];
            if (removed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue;
            if (IoU(dets[i].box, dets[j].box) > iou_thres) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}
// 解析形状: [output_pred, anchors], 每一列表示一个目标的所有信息 4 + len(class_names)
static std::vector<Detection> PostprocessDetections(
    const float* feat,              // 指向单张图的输出首地址
    int output_pred,                // 4 + num_classes
    int anchors,                    // 锚点总数
    float r, float dw, float dh,    // 反 letterbox 参数
    int orig_w, int orig_h,         // 原图大小
    float conf_thres = 0.5f,
    float iou_thres  = 0.5f)
{
    int num_classes = (int)(sizeof(class_names)/sizeof(class_names[0]));
    std::vector<Detection> dets;
    dets.reserve(512);

    // feat 的内存布局：维度 [output_pred, anchors]
    // 访问方式：feat[i * anchors + j]  (i: 0..output_pred-1, j: 0..anchors-1)
    const float* cx_ptr = feat + 0 * anchors;
    const float* cy_ptr = feat + 1 * anchors;
    const float* w_ptr  = feat + 2 * anchors;
    const float* h_ptr  = feat + 3 * anchors;
    const float* cls_ptr= feat + 4 * anchors;  // 后面紧跟 num_classes * anchors

    for (int j = 0; j < anchors; ++j) {
        // 取类别最大值与 id
        int best_c = -1;
        float best_s = -1.f;
        for (int c = 0; c < num_classes; ++c) {
            float s = cls_ptr[c * anchors + j];
            if (s > best_s) { best_s = s; best_c = c; }
        }
        if (best_s < conf_thres) continue;

        float cx = cx_ptr[j];
        float cy = cy_ptr[j];
        float w  = w_ptr[j];
        float h  = h_ptr[j];

        float x = (cx - w * 0.5f - dw) / r;
        float y = (cy - h * 0.5f - dh) / r;
        float ww = w / r;
        float hh = h / r;

        x  = std::max(0.f, std::min(x,  (float)orig_w  - 1.f));
        y  = std::max(0.f, std::min(y,  (float)orig_h - 1.f));
        ww = std::max(0.f, std::min(ww, (float)orig_w  - x));
        hh = std::max(0.f, std::min(hh, (float)orig_h - y));

        if (ww <= 0.f || hh <= 0.f) continue;

        Detection d;
        d.box = cv::Rect2f(x, y, ww, hh);
        d.score = best_s;
        d.class_id = best_c;
        d.track_id = -1;
        dets.push_back(d);
    }

    // NMS
    std::vector<int> keep = NMS(dets, iou_thres);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) out.push_back(dets[idx]);
    return out;
}
YoloDetectionNode::YoloDetectionNode(std::string eng_path, int device_id){
    eng_path_ = eng_path;
    device_id_ = device_id;
    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUDA(cudaStreamCreate(&stream_));
    runtime_ = nvinfer1::createInferRuntime(logger);
    MY_ASSERT(runtime != nullptr);
    int model_size = 0;
    char *trt_model_stream = ReadFromPath(eng_path, model_size);
    MY_ASSERT(trt_model_stream != nullptr);
    engine_ = runtime_->deserializeCudaEngine(trt_model_stream, model_size);
	MY_ASSERT(engine_ != nullptr);
    context_ = engine->createExecutionContext();
	MY_ASSERT(context != nullptr);
    delete []trt_model_stream;

    int num_bindings = engine->getNbIOTensors();
    log_debug("input/output : {}", num_bindings);
    for (int i = 0; i < num_bindings; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_tensor_info_.push_back({i, std::string(tensor_name)});
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_tensor_info_.push_back({i, std::string(tensor_name)});
    }
    for(int idx = 0; idx < in_tensor_info_.size(); idx++){
        nvinfer1::Dims in_dims = context->getTensorShape(in_tensor_info_[idx].second.c_str());
        log_debug("input: {}", in_tensor_info_[idx].second);
        for(int i = 0; i < in_dims.nbDims; i++){
            log_debug("dims [{}] : {}", i, in_dims.d[i]);
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(in_tensor_info_[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            log_debug("  int32");
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            log_debug("  int64");
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            log_debug("  float");
        }
    }
    for(int idx = 0; idx < out_tensor_info_.size(); idx++){
        nvinfer1::Dims out_dims=context->getTensorShape(out_tensor_info_[idx].second.c_str());
        log_debug("output: {}", out_tensor_info_[idx].second);
        for(int i = 0; i < out_dims.nbDims; i++){
            log_debug("dims [{}] : {}", i, out_dims.d[i]);
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(out_tensor_info_[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            log_debug("  int32");
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            log_debug("  int64");
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            log_debug("  float");
        }
    }
    MY_ASSERT(in_tensor_info_.size() == 1);
    MY_ASSERT(out_tensor_info_.size() == 1);
    nvinfer1::Dims in_dims = context->getTensorShape(in_tensor_info[0].second.c_str());
    nvinfer1::Dims out_dims = context->getTensorShape(out_tensor_info[0].second.c_str());
    size_t max_in_size_byte = CountElement(in_dims, batch_size) * sizeof(float); // batch_size * input_h * input_w * 3 * sizeof(float)
    size_t max_out_size_byte = CountElement(out_dims, batch_size) * sizeof(float); // batch_size * output_pred_ * anchors_ * sizeof(float)

    // in_dims.d[0] dynamic batch_size == -1 
    int channel = in_dims.d[1];
    MY_ASSERT(channel == 3);
    input_h_ = in_dims.d[2];
	input_w_ = in_dims.d[3];
    log_debug("batch_size: {} channel:{} input_h:{} input_w:{}", batch_size_, channel, input_h_, input_w_);
    // out_dims.d[0] dynamic batch_size == -1 
    output_pred_ = out_dims.d[1];
	anchors_ = out_dims.d[2];
    log_debug("output_pred: {} anchors: {}", output_pred, anchors);

    CHECK_CUDA(cudaMalloc(&buffers_[in_tensor_info_[0].first], max_in_size_byte));
    int one_output_len = output_pred_ * anchors_;
	CHECK_CUDA(cudaMalloc(&buffers_[out_tensor_info_[0].first], max_out_size_byte));
    output_ = new float[max_out_size_byte];
    context_->setInputTensorAddress(in_tensor_info_[0].second.c_str(), buffers_[in_tensor_info_[0].first]);
    context_->setOutputTensorAddress(out_tensor_info_[0].second.c_str(), buffers_[out_tensor_info_[0].first]);
    worker_ = std::thread(&YoloDetectionNode::DetectThreadLoop, this);
}
YoloDetectionNode::~YoloDetectionNode(){
    abort_ = true;
    worker_.join();
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
    delete []output_;
    CHECK_CUDA(cudaFree(buffers_[0]));
    CHECK_CUDA(cudaFree(buffers_[1]));
    CHECK_CUDA(cudaStreamDestroy(stream_));
    if(collector_){
        delete collector_;
        collector_ = nullptr;
    }
    if(relayer_){
        delete relayer_;
        relayer_ = nullptr;
    }
    if(distributor_){
        delete distributor_;
        distributor_ = nullptr;
    }

}
int YoloDetectionNode::Inference(const int batch_size){
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = batch_size;
    trt_in_dims.d[1] = 3;
    trt_in_dims.d[2] = input_h_;
    trt_in_dims.d[3] = input_w_;
    context_->setInputShape(in_tensor_info_[0].second.c_str(), trt_in_dims);
    if(!context->enqueueV3(stream_)) {
        slog_error("enqueueV3 failed!");
        return -1;
    }
    int one_output_len = output_pred_ * anchors_;
    CHECK_CUDA(cudaMemcpyAsync(output_, buffers[out_tensor_info_[0].first], batch_size * one_output_len * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    return 0;
}
void YoloDetectionNode::DetectThreadLoop(){
    while (!abort_) {
        if(!collector_){
            log_error("No data source available");
            return;
        }
        std::vector<ImgPacket*> packets= collector_->GetBatch(batch_size_);
        if(packets.empty()){
            continue;
        }
        std::vector<std::tuple<float, float, float>> res_pre;
        int buffer_idx = 0;
        char* input_ptr = static_cast<char*>(buffers[in_tensor_info_[0].first]);
        for(int i = 0; i < packets.size(); i++){  
            ImgPacket* packet = packets[i];
            std::tuple<float, float, float> res = PreprocessImage_GPU(packet->GetImg(), input_ptr + buffer_idx, channel, input_h_, input_w_, stream);
            buffer_idx += input_h_ * input_w_ * 3 * sizeof(float);
            res_pre.push_back(res);
        }
        MY_ASSERT(packets.size() == res_pre.size());
        if(Inference(packets.size()) < 0){
            log_error("Inference error");
        }
        int one_output_len = output_pred_ * anchors_;
        for (int b = 0; b < res_pre.size(); ++b) {
            ImgPacket* packet = packets[i];
            auto [r, dw, dh] = res_pre[b];
            float* feat_b = output_ + b * one_output_len;
            int orig_h = packet->GetImg().rows;
            int orig_w = packet->GetImg().cols;
            DetectionInfo info;
            info.dets = PostprocessDetections(feat_b, output_pred, anchors, r, dw, dh, orig_w, orig_h, /*conf*/0.5f, /*iou*/0.5f);
            info.class_names = std::vector<std::string>(std::begin(class_names),std::end(class_names));
            packet->SetDetectionInfo(info);
            if(relayer_){
                relayer_->Push(packet);
            }
            if(distributor_){
                distributor_->Push(packet);
            }
        }
    }
    
}
#endif // DETECTION_NVIDIA DETECTION_ASCEND