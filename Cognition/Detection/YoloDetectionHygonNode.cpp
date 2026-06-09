#if defined(DETECTION_HYGON)
#include "YoloDetectionNode.h"
#include "dataProcess.h"
#define MY_ASSERT(x)                     \
    do {                                \
        if (!(x)) {                     \
            std::abort();               \
        }                               \
    } while (0)

static float GetUnpadSize(int new_w, int new_h, int orig_w, int orig_h, int &new_unpad_w, int &new_unpad_h){
    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);
    new_unpad_w = static_cast<int>(std::round(orig_w * r));
    new_unpad_h = static_cast<int>(std::round(orig_h * r));
    return r;
}
static std::tuple<float, float, float>  PreprocessImage_GPU(cv::Mat &img, void *img_buffer, void *buffer, int input_h, int input_w, cudaStream_t stream)
{
    TimeMetrics t;
    t.startTimer();
    int orig_h = img.rows;
    int orig_w = img.cols;
    void *img_ptr = img.data;
    CHECK_CUDA(cudaMemcpyAsync(img_buffer, img_ptr, orig_h * orig_w * 3, cudaMemcpyHostToDevice, stream));

    preprocess_letter_bbox_resize((unsigned char *)img_buffer, nullptr, (float *)buffer, orig_w, orig_h, input_w, input_h, stream);
    // test
#if 0
    void *addr;
    CHECK_CUDA(cudaMalloc(&addr, input_w * input_h * 3));
    preprocess_letter_bbox_resize((unsigned char *)img_buffer, (unsigned char *)addr, nullptr, orig_w, orig_h, input_w, input_h);
    cv::Mat img_cpu(input_h, input_w, CV_8UC3);
    size_t bytes = input_w * input_h * 3;
    CHECK_CUDA(cudaMemcpy(img_cpu.data, addr, bytes, cudaMemcpyDeviceToHost));
    if(!cv::imwrite("output.jpg", img_cpu)){
        log_error("Failed to save image");
    } 
    CHECK_CUDA(cudaFree(addr));
    exit(0);
#endif

    int new_unpad_w, new_unpad_h;
    float r = GetUnpadSize(input_w, input_h, orig_w, orig_h, new_unpad_w, new_unpad_h);
    float dw = input_w - new_unpad_w;
    float dh = input_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;
    return std::make_tuple(r, dw, dh);
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
    int num_classes,
    float conf_thres = 0.5f,
    float iou_thres  = 0.5f)
{
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
}
YoloDetectionNode::~YoloDetectionNode(){
    abort_ = true;
    // if(thread_run_flag_)
    //     worker_.join();
    if(worker_.joinable()) {
        worker_.join();
    }
    CHECK_CUDA(cudaStreamDestroy(stream_));
    log_debug("~YoloDetectionNode");
}
// 不同线程不能共用同一个program对象
void YoloDetectionNode::Init(){
#if 0
    // onnx
    migraphx::onnx_options onnx_options;
    onnx_options.map_input_dims["images"] = {(long unsigned int)batch_size_, 3, 640, 640};
    net_ = migraphx::parse_onnx(eng_path_, onnx_options);
#endif
    // mxr
    migraphx::file_options options;
    // options.device_id = device_id_;
    net_ = migraphx::load(eng_path_, options);

    log_debug("inputs:");
    std::unordered_map<std::string, migraphx::shape> inputs = net_.get_inputs();
    for (auto i : inputs){
        log_debug("  {}", i.first);
    }
    log_debug("outputs:");
    std::unordered_map<std::string, migraphx::shape> outputs = net_.get_outputs();
    for (auto i : outputs){
        log_debug("  {}", i.first);
    }
    input_name_ = inputs.begin()->first;
    input_shape_ = inputs.begin()->second;
    int N = input_shape_.lens()[0];
    int C = input_shape_.lens()[1];
    int H = input_shape_.lens()[2];
    int W = input_shape_.lens()[3];
    input_h_ = H;
    input_w_ = W;
    log_debug("batch size:{} channels:{} input_h:{} input_w:{}", N, C, H, W);
    output_name_ = outputs.begin()->first;
    output_shape_ = outputs.begin()->second;
    // int N = output_shape_.lens()[0];
    output_pred_ = output_shape_.lens()[1];
	anchors_ = output_shape_.lens()[2];
    log_debug("output_pred_:{} anchors_:{}", output_pred_, anchors_);
#if 0
    // onnx
    migraphx::target gpu_target = migraphx::gpu::target{};
    migraphx::quantize_fp16(net_);
    migraphx::compile_options options;
    options.device_id = device_id_;
    options.offload_copy = false;
    net_.compile(gpu_target, options);
    log_debug("{} compile success", eng_path_);
#endif

    for (auto x : net_.get_parameter_shapes()){
        parameter_map_[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
    } 

    // warm up
    net_.eval(parameter_map_);

    size_t input_len = batch_size_ * input_h_ * input_w_ * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&buffers_[0], input_len));
    size_t output_len = batch_size_ * output_pred_ * anchors_ * sizeof(float);
	CHECK_CUDA(cudaMalloc(&buffers_[1], output_len));
    output_ = new float[output_len];
}
void YoloDetectionNode::UnInit(){
    CHECK_CUDA(cudaFree(buffers_[0]));
    CHECK_CUDA(cudaFree(buffers_[1]));
    delete []output_;
}
int YoloDetectionNode::Inference(const int batch_size){
    // migraphx::shape infer_shape = migraphx::shape(input_shape_.type(), {(long unsigned int)batch_size, 3, (long unsigned int)input_h_, (long unsigned int)input_w_});
    migraphx::shape infer_shape = migraphx::shape(input_shape_.type(), {4, 3, (long unsigned int)input_h_, (long unsigned int)input_w_});
    parameter_map_[input_name_] = migraphx::argument{infer_shape, (float*)buffers_[0]};
    std::vector<migraphx::argument> results = net_.eval(parameter_map_);

    migraphx::argument result  = migraphx::gpu::from_gpu(results[0]);
    memcpy(output_, result.data(), batch_size * output_pred_ * anchors_ * sizeof(float));

    return 0;
}
void YoloDetectionNode::SetDataNode(std::shared_ptr<CollectorNode> collector, std::shared_ptr<RelayNode> relayer,std::shared_ptr<DistributorNode> distributor){
    collector_ = collector; 
    relayer_ = relayer; 
    distributor_ = distributor;
    worker_ = std::thread(&YoloDetectionNode::DetectThreadLoop, this);
    std::unique_lock<std::mutex> guard(mutex_init_);
    cond_init_.wait(guard);
}
void YoloDetectionNode::DetectThreadLoop(){
    thread_run_flag_ = true;
    CHECK_CUDA(cudaSetDevice(device_id_));
    Init();
    cond_init_.notify_one();
    TimeMetrics time_for_log;
    time_for_log.startTimer();
    while (!abort_) {
        if(!collector_){
            log_error("No data source available");
            return;
        }
        TimeMetrics t_detect;
        std::vector<ImgPacket*> packets= collector_->GetBatch(batch_size_);
        if(packets.empty()){
            continue;
        }
        std::vector<std::tuple<float, float, float>> res_pre;
        int buffer_idx = 0;
        char* input_ptr = static_cast<char*>(buffers_[0]);
        TimeMetrics t;
        t.startTimer();
        for(int i = 0; i < packets.size(); i++){  
            ImgPacket* packet = packets[i];
            int new_unpad_w, new_unpad_h;
            GetUnpadSize(input_w_, input_h_, packet->context->width, packet->context->height, new_unpad_w, new_unpad_h);
            MemAllocate(packet->context, new_unpad_w, new_unpad_h, 3);
            void *img_buffer = packet->context->img_buffer;
            std::tuple<float, float, float> res = PreprocessImage_GPU(packet->img, img_buffer, input_ptr + buffer_idx, input_h_, input_w_, stream_);
            buffer_idx += input_h_ * input_w_ * 3 * sizeof(float);
            res_pre.push_back(res);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        int pre_time = t.stopTimer();

        MY_ASSERT(packets.size() == res_pre.size());

        t.startTimer();
        if(Inference(packets.size()) < 0){
            log_error("Inference error");
        }
        int infer_time = t.stopTimer();

        t.startTimer();
        int one_output_len = output_pred_ * anchors_;
        for (int b = 0; b < res_pre.size(); ++b) {
            ImgPacket* packet = packets[b];
            auto [r, dw, dh] = res_pre[b];
            float* feat_b = output_ + b * one_output_len;
            int orig_h = packet->img.rows;
            int orig_w = packet->img.cols;
            DetectionInfo info;
            int num_classes = (int)(sizeof(class_names_)/sizeof(class_names_[0]));
            info.dets = PostprocessDetections(feat_b, output_pred_, anchors_, r, dw, dh, orig_w, orig_h, num_classes, /*conf*/0.5f, /*iou*/0.5f);
            info.class_names = std::vector<std::string>(std::begin(class_names_),std::end(class_names_));
            packet->info = info;
            if(relayer_){
                relayer_->Push(packet);
            }
            if(distributor_){
                distributor_->Push(packet);
            }
        }
        int after_time = t.stopTimer();
        int detect_all_time = t_detect.stopTimer();
        if(time_for_log.stopTimer() >= 1000) {
            time_for_log.startTimer();
            log_debug("detect_all_time:{} pre_time:{} infer_time:{} after_time:{}", detect_all_time, pre_time, infer_time, after_time);
        }
    }
    UnInit();
    log_debug("DetectThreadLoop finished");
}
#endif // DETECTION_HYGON