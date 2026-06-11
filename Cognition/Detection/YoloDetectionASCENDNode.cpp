#if defined(DETECTION_ASCEND)
#include "YoloDetectionNode.h"
#define MY_ASSERT(x)                     \
    do {                                \
        if (!(x)) {                     \
            std::abort();               \
        }                               \
    } while (0)
#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)
std::string GetType(aclDataType type){
    std::string type_str;
    switch (type)
    {
    case ACL_DT_UNDEFINED:
        type_str = std::string("ACL_DT_UNDEFINED");
        break;
    case ACL_FLOAT:
        type_str = std::string("ACL_FLOAT");
        break;
    case ACL_FLOAT16:
        type_str = std::string("ACL_FLOAT16");
        break;
    case ACL_INT8:
        type_str = std::string("ACL_INT8");
        break;
    case ACL_INT32:
        type_str = std::string("ACL_INT32");
        break;
    case ACL_UINT8:
        type_str = std::string("ACL_UINT8");
        break;
    case ACL_INT16:
        type_str = std::string("ACL_INT16");
        break;
    case ACL_UINT16:
        type_str = std::string("ACL_UINT16");
        break;
    case ACL_UINT32:
        type_str = std::string("ACL_UINT32");
        break;
    case ACL_INT64:
        type_str = std::string("ACL_INT64");
        break;
    case ACL_UINT64:
        type_str = std::string("ACL_UINT64");
        break;
    case ACL_DOUBLE:
        type_str = std::string("ACL_DOUBLE");
        break;
    case ACL_BOOL:
        type_str = std::string("ACL_BOOL");
        break;
    case ACL_STRING:
        type_str = std::string("ACL_STRING");
        break;
    case ACL_COMPLEX64 :
        type_str = std::string("ACL_COMPLEX64");
        break;
    case ACL_COMPLEX128:
        type_str = std::string("ACL_COMPLEX128");
        break;
    case ACL_BF16:
        type_str = std::string("ACL_BF16");
        break;
    case ACL_INT4:
        type_str = std::string("ACL_INT4");
        break;
    case ACL_UINT1:
        type_str = std::string("ACL_UINT1");
        break;
    case ACL_COMPLEX32:
        type_str = std::string("ACL_COMPLEX32");
        break;
    case ACL_HIFLOAT8:
        type_str = std::string("ACL_HIFLOAT8");
        break;
    case ACL_FLOAT8_E5M2:
        type_str = std::string("ACL_FLOAT8_E5M2");
        break;
    case ACL_FLOAT8_E4M3FN:
        type_str = std::string("ACL_FLOAT8_E4M3FN");
        break;
    case ACL_FLOAT8_E8M0:
        type_str = std::string("ACL_FLOAT8_E8M0");
        break;
    case ACL_FLOAT6_E3M2:
        type_str = std::string("ACL_FLOAT6_E3M2");
        break;
    case ACL_FLOAT6_E2M3:
        type_str = std::string("ACL_FLOAT6_E2M3");
        break;
    case ACL_FLOAT4_E2M1:
        type_str = std::string("ACL_FLOAT4_E2M1");
        break;
    case ACL_FLOAT4_E1M2:
        type_str = std::string("ACL_FLOAT4_E1M2");
        break;
    default:
        break;
    }
    return type_str;
}
static float GetUnpadSize(int new_w, int new_h, int orig_w, int orig_h, int &new_unpad_w, int &new_unpad_h){
    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);
    new_unpad_w = static_cast<int>(std::round(orig_w * r));
    new_unpad_h = static_cast<int>(std::round(orig_h * r));
    return r;
}
// src_addr & dst_addr:YUV420SP_U8
void LetterBox(hi_vpc_chn channel_id_letterbox, void *src_addr, uint64_t src_size, int src_w, int src_h, void *dst_addr, uint64_t dst_size, int dst_w, int dst_h){
    hi_vpc_pic_info input_pic;
    input_pic.picture_width = src_w;
    input_pic.picture_height = src_h;
    input_pic.picture_format = HI_PIXEL_FORMAT_BGR_888;
    input_pic.picture_width_stride = src_w * 3;
    input_pic.picture_height_stride = src_h;
    input_pic.picture_buffer_size = src_size;
    input_pic.picture_address = src_addr;

    hi_vpc_pic_info output_pic;
    output_pic.picture_width = dst_w;
    output_pic.picture_height = dst_h;
    output_pic.picture_format = HI_PIXEL_FORMAT_RGB_888;
    output_pic.picture_width_stride = dst_w * 3;
    output_pic.picture_height_stride = dst_h;
    output_pic.picture_buffer_size = dst_size;
    output_pic.picture_address = dst_addr;
    
    // 裁剪区域
    uint32_t cropLeftOffset = 0; // 裁剪的左上角x
    // must even
    uint32_t cropTopOffset = 0; // 裁剪的左上角y
    // must odd
    uint32_t cropRightOffset = (((cropLeftOffset + src_w) >> 1) << 1);
    // must odd
    uint32_t cropBottomOffset = (((cropTopOffset + src_h) >> 1) << 1);

    //paste area
    float rx = (float)src_w / (float)dst_w;
    float ry = (float)src_h / (float)dst_h;
    int dx = 0;//左右填充大小
    int dy = 0;//上下填充大小
    float r = 0.0f;
    if (rx > ry) { //宽的比例比较大，则按照宽进行裁剪，高适应，就是高上下会有填充
        dx = 0;
        r = rx;
        dy = (dst_h - src_h / r) / 2;
    } else { // 高的比例比较大，按照高进行裁剪，宽适应，就是左右会有填充
        dy = 0;
        r = ry;
        dx = (dst_w - src_w / r) / 2;
    }
    // must even
    uint32_t pasteLeftOffset = ALIGN_UP16(dx); // 填充宽必须是16的倍数，粘贴偏移
    // must even
    uint32_t pasteTopOffset = ALIGN_UP2(dy); // 填充高必须是2的倍数，粘贴偏移
    // must odd
    uint32_t pasteRightOffset = (((dst_w + pasteLeftOffset - 2 * dx) >> 1) << 1);
    // must odd
    uint32_t pasteBottomOffset = (((dst_h - dy) >> 1) << 1);
    hi_vpc_crop_resize_paste_region crop_resize_paste_infos;
    crop_resize_paste_infos.dest_pic_info = output_pic;
    crop_resize_paste_infos.crop_region.left_offset = cropLeftOffset;
    crop_resize_paste_infos.crop_region.top_offset = cropTopOffset;
    crop_resize_paste_infos.crop_region.crop_width = cropRightOffset - cropLeftOffset;
    crop_resize_paste_infos.crop_region.crop_height = cropBottomOffset - cropTopOffset;
    crop_resize_paste_infos.resize_info.resize_width = pasteRightOffset - pasteLeftOffset;
    crop_resize_paste_infos.resize_info.resize_height = pasteBottomOffset - pasteTopOffset;
    crop_resize_paste_infos.resize_info.interpolation = 0;
    crop_resize_paste_infos.dest_left_offset = pasteLeftOffset;
    crop_resize_paste_infos.dest_top_offset = pasteTopOffset;
    uint32_t task_id = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_crop_resize_paste(channel_id_letterbox, &input_pic, &crop_resize_paste_infos, 1, &task_id, -1));
    CHECK_DVPP_MPI(hi_mpi_vpc_get_process_result(channel_id_letterbox, task_id, -1));
    return;
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
    CHECK_ACL(aclrtSetDevice(device_id_));
    CHECK_DVPP_MPI(hi_mpi_sys_init()); // hi_mpi_sys_init 必须在aclrtSetDevice之后
    CHECK_ACL(aclmdlLoadFromFile(eng_path_.c_str(), &model_id_));
    model_desc_ = aclmdlCreateDesc();
    if(model_desc_ == nullptr){
        log_error("YoloDetectionNode aclmdlCreateDesc error");
        exit(0);
    }
    CHECK_ACL(aclmdlGetDesc(model_desc_, model_id_));
    if(model_desc_ == nullptr){
        log_error("YoloDetectionNode aclmdlGetDesc error");
        exit(0);
    }
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        log_error("YoloDetectionNode aclmdlCreateDataset ouput error");
        exit(0);
    }
    input_num_ = aclmdlGetNumInputs(model_desc_);
    for (size_t i = 0; i < input_num_; ++i) { // 3个输入，原始图像、AIPP、动态batch
        log_debug("intput name: {} type: {}", aclmdlGetInputNameByIndex(model_desc_, i), GetType(aclmdlGetInputDataType(model_desc_, i)));
    }
    outputs_num_ = aclmdlGetNumOutputs(model_desc_); // 1个输出
    for (size_t i = 0; i < outputs_num_; ++i) {
        log_debug("output name: {} type: {}", aclmdlGetOutputNameByIndex(model_desc_, i), GetType(aclmdlGetOutputDataType(model_desc_, i)));
        size_t buf_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        void *output_buffer = nullptr;
        CHECK_ACL(aclrtMalloc(&output_buffer, buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
        aclDataBuffer* data_buf = aclCreateDataBuffer(output_buffer, buf_size);
        if (data_buf == nullptr) {
            log_error("Yolov5Model aclCreateDataBuffer error");
            exit(0);
        }
        CHECK_ACL(aclmdlAddDatasetBuffer(output_, data_buf));
        output_buf_.push_back(output_buffer); // 第i个输出的显存地址
        output_size_.push_back(buf_size); // 第i个输出的显存大小
        output_data_buf_.push_back(data_buf); // 包装了第i个输出显存地址的输出描述
    }


    // AIPP
    CHECK_ACL(aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_AIPP_NAME, &aipp_index_));
    size_t data_len_aipp = aclmdlGetInputSizeByIndex(model_desc_, aipp_index_);
    
    CHECK_ACL(aclrtMalloc(&input_AIPP_, data_len_aipp, ACL_MEM_MALLOC_NORMAL_ONLY));

    // 动态batch
    CHECK_ACL(aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &dynamic_batch_idx));
    size_t data_len = aclmdlGetInputSizeByIndex(model_desc_, dynamic_batch_idx);
    CHECK_ACL(aclrtMalloc(&input_batch_, data_len, ACL_MEM_MALLOC_NORMAL_ONLY));
    hi_vpc_chn_attr st_chn_attr {};
    st_chn_attr.attr = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_sys_create_chn(&channel_id_letterbox_, &st_chn_attr));

    /*
    intput name: images type: ACL_UINT8
    intput name: ascend_mbatch_shape_data type: ACL_INT32
    intput name: ascend_dynamic_aipp_data type: ACL_UINT8
    output name: /model.23/Concat_25:0:output0 type: ACL_FLOAT 
    */
    //    typedef struct aclmdlIODims {
    //     char name[ACL_MAX_TENSOR_NAME_LEN]; /**< tensor name */
    //     size_t dimCount; /**< dim array count */
    //     int64_t dims[ACL_MAX_DIM_CNT]; /**< dim data array */
    //     } aclmdlIODims;
    aclmdlIODims input_dims;
    CHECK_ACL(aclmdlGetInputDims(model_desc_, 0, &input_dims));
    // int batch_size = input_dims.dims[0] // 动态batch 
    int channel = 3;
    if(input_dims.dims[1] > 0){
        channel = input_dims.dims[1];
    }
    MY_ASSERT(channel == 3);
    input_dims.dims[2] > 0 ? input_h_ = input_dims.dims[2]: input_h_ = 640;
    input_dims.dims[3] > 0 ? input_w_ = input_dims.dims[3]: input_w_ = 640;
    log_debug("batch_size: {} channel:{} input_h:{} input_w:{}", batch_size_, channel, input_h_, input_w_);
    aclmdlIODims output_dims;
    CHECK_ACL(aclmdlGetOutputDims(model_desc_, 0, &output_dims));
    output_pred_ = output_dims.dims[1];
    anchors_ = output_dims.dims[2];
    log_debug("output_pred: {} anchors: {}", output_pred_, anchors_);

    output_prob_len_ =  anchors_ * output_pred_;
    output_prob_ = new float[output_prob_len_ * batch_size_];
    setInput();
    // input
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_img_, batch_size_ * input_h_ * input_w_ * 3));
    CHECK_ACL(aclrtCreateStream(&stream_));
}
void YoloDetectionNode::setInput(){
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        log_error("YoloDetectionNode aclmdlCreateDataset input error");
        exit(0);
    }
    for (size_t i = 0; i < input_num_; ++i) { // 3个输入，原始图像、AIPP、动态batch
        size_t buf_size = aclmdlGetInputSizeByIndex(model_desc_, i);
        aclDataBuffer* data_buf;
        if(i == 0){ // image
            // printf("image:%d\n", i);
            data_buf = aclCreateDataBuffer(nullptr,0);
            if (data_buf == nullptr) {
                log_error("YoloDetectionNode aclCreateDataBuffer input error");
                exit(0);
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input_, data_buf));
            data_buf_input_0_ = data_buf;
        }
        else if(i == aipp_index_){ // AIPP
            // printf("aipp_index_:%d\n", i);
            data_buf = aclCreateDataBuffer(input_AIPP_, buf_size);
            if (data_buf == nullptr) {
                log_error("YoloDetectionNode aclCreateDataBuffer input error");
                exit(0);
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input_, data_buf));
            aclmdlAIPP* aipp_param_tensor = aclmdlCreateAIPP(batch_size_);
            if(aipp_param_tensor == nullptr){
                log_error("YoloDetectionNode aclmdlCreateAIPP input error");
                exit(0);
            }
            CHECK_ACL(aclmdlSetAIPPInputFormat(aipp_param_tensor, ACL_RGB888_U8)); // 设置输入格式
            CHECK_ACL(aclmdlSetAIPPSrcImageSize(aipp_param_tensor, input_w_, input_h_)); // 输入图像尺寸
#if 0
            CHECK_ACL(aclmdlSetAIPPInputFormat(aipp_param_tensor, ACL_YUV420SP_U8)); // 设置输入格式
            /*
            设置CSC色域转换相关的参数，若色域转换开关关闭，则调用该接口设置以下参数无效。
            YUV转BGR：
            | B |   | cscMatrixR0C0 cscMatrixR0C1 cscMatrixR0C2 | | Y - cscInputBiasR0 |
            | G | = | cscMatrixR1C0 cscMatrixR1C1 cscMatrixR1C2 | | U - cscInputBiasR1 | >> 8
            | R |   | cscMatrixR2C0 cscMatrixR2C1 cscMatrixR2C2 | | V - cscInputBiasR2 |
            BGR转YUV：
            | Y |   | cscMatrixR0C0 cscMatrixR0C1 cscMatrixR0C2 | | B |        | cscOutputBiasR0 |
            | U | = | cscMatrixR1C0 cscMatrixR1C1 cscMatrixR1C2 | | G | >> 8 + | cscOutputBiasR1 |
            | V |   | cscMatrixR2C0 cscMatrixR2C1 cscMatrixR2C2 | | R |        | cscOutputBiasR2 |
            */
            CHECK_ACL(aclmdlSetAIPPCscParams(aipp_param_tensor, 1, 298, 0, 409, 298, -100, -208, 298, 516,  0, 0, 0, 0, 16, 128, 128));

            CHECK_ACL(aclmdlSetAIPPRbuvSwapSwitch(aipp_param_tensor, 0)); // 通道（R/B 、U/V）交换开关 0-不交换 1-交换
            CHECK_ACL(aclmdlSetAIPPAxSwapSwitch(aipp_param_tensor, 0)); // 控制 RGBA 到 ARGB 或 YUVA 到 AYUV 的交换开关。0-不交换。
#endif
            /*
            设置 AIPP（图像预处理引擎）中的缩放参数 
            int8_t scfSwitch: 缩放开关。非零值-启用缩放；0-禁用缩放。
            int32_t scfInputSizeW: 输入图像的宽度，用于缩放计算。
            int32_t scfInputSizeH: 输入图像的高度，用于缩放计算。
            int32_t scfOutputSizeW: 输出图像的目标宽度。
            int32_t scfOutputSizeH: 输出图像的目标高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPScfParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的裁剪参数
            int8_t cropSwitch: 裁剪开关。非零值-启用裁剪 0-禁用裁剪。
            int32_t cropStartPosW: 裁剪区域的起始水平位置（X 坐标）。
            int32_t cropStartPosH: 裁剪区域的起始垂直位置（Y 坐标）。
            int32_t cropSizeW: 裁剪区域的宽度。
            int32_t cropSizeH: 裁剪区域的高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPCropParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的填充参数
            int8_t paddingSwitch: 填充开关。非零值-启用填充；0-禁用填充。
            int32_t paddingSizeTop: 顶部填充的大小。
            int32_t paddingSizeBottom: 底部填充的大小。
            int32_t paddingSizeLeft: 左侧填充的大小。
            int32_t paddingSizeRight: 右侧填充的大小。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPPaddingParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            图像预处理归一化操作过程如下：
            pixel_out_chx(i)=[pixel_in_chx(i)-mean_chn_i-min_chn_i]*var_reci_chn_i
            */
            for(int idx = 0; idx < batch_size_; idx++){
                float dtcPixelMeanChni0 = 0.0 * 255.0;
                float dtcPixelMeanChni1 = 0.0 * 255.0;
                float dtcPixelMeanChni2 = 0.0 * 255.0;
                CHECK_ACL(aclmdlSetAIPPDtcPixelMean(aipp_param_tensor, dtcPixelMeanChni0, dtcPixelMeanChni1, dtcPixelMeanChni2, 0, idx));
            }
            for(int idx = 0; idx < batch_size_; idx++){
                CHECK_ACL(aclmdlSetAIPPDtcPixelMin(aipp_param_tensor, 0.0, 0.0, 0.0, 0.0, idx));
            }
            for(int idx = 0; idx < batch_size_; idx++){
                float dtcPixelVarReciChn0 = 1.0 / 255.0;
                float dtcPixelVarReciChn1 = 1.0 / 255.0;
                float dtcPixelVarReciChn2 = 1.0 / 255.0;
                CHECK_ACL(aclmdlSetAIPPPixelVarReci(aipp_param_tensor, dtcPixelVarReciChn0, dtcPixelVarReciChn1, dtcPixelVarReciChn2, 1.0, idx));
            }
            CHECK_ACL(aclmdlSetInputAIPP(model_id_, input_, aipp_index_, aipp_param_tensor));
            CHECK_ACL(aclmdlDestroyAIPP(aipp_param_tensor));
        }
        else if(i == dynamic_batch_idx){ // 动态batch
            // printf("dynamic_batch_idx:%d\n", i);
            data_buf = aclCreateDataBuffer(input_batch_, buf_size);
            if (data_buf == nullptr) {
                log_error("YoloDetectionNode aclCreateDataBuffer input error");
                exit(0);
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input_, data_buf));
        }
        else{
            log_error("input error");
        }
    }
}
YoloDetectionNode::~YoloDetectionNode(){
    abort_ = true;
    // if(thread_run_flag_)
    //     worker_.join();
    if(worker_.joinable()) {
        worker_.join();
    }
    CHECK_ACL(aclmdlUnload(model_id_));
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(data_buffer);
        CHECK_ACL(aclrtFree(data));
        CHECK_ACL(aclDestroyDataBuffer(data_buffer));
        data_buffer = nullptr;
    }
    CHECK_ACL(aclmdlDestroyDataset(output_));
    CHECK_DVPP_MPI(hi_mpi_dvpp_free(input_addr_img_));
    
    CHECK_ACL(aclrtFree(input_AIPP_));
    CHECK_ACL(aclrtFree(input_batch_));
    CHECK_ACL(aclmdlDestroyDesc(model_desc_));
    CHECK_DVPP_MPI(hi_mpi_vpc_destroy_chn(channel_id_letterbox_));
    if(output_prob_){
        delete []output_prob_;
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(input_, i);
        CHECK_ACL(aclDestroyDataBuffer(data_buffer));
        data_buffer = nullptr;
    }
    CHECK_ACL(aclmdlDestroyDataset(input_));
    input_ = nullptr;
    CHECK_ACL(aclrtDestroyStream(stream_));
    log_debug("~YoloDetectionNode");
}
int YoloDetectionNode::Inference(const int batch_size){
    CHECK_ACL(aclUpdateDataBuffer(data_buf_input_0_, input_addr_img_, batch_size * input_w_ * input_h_ * 3));
    CHECK_ACL(aclmdlSetDynamicBatchSize(model_id_, input_, dynamic_batch_idx, batch_size));
    CHECK_ACL(aclmdlExecuteAsync(model_id_, input_, output_, stream_));
    /*
    intput name: images type: ACL_UINT8
    intput name: ascend_mbatch_shape_data type: ACL_INT32
    intput name: ascend_dynamic_aipp_data type: ACL_UINT8
    output name: /model.23/Concat_25:0:output0 type: ACL_FLOAT 
    */
    for (uint32_t i = 0; i < outputs_num_; i++) {
        switch (i)
        {
        case 0: // output
            memset(output_prob_, 0, batch_size_ *  output_prob_len_ * sizeof(float));
            CHECK_ACL(aclrtMemcpyAsync(output_prob_ , batch_size_ *  output_prob_len_ * sizeof(float), output_buf_[i], batch_size * output_prob_len_ * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST, stream_));
            break;
        default:
            break;
        }
        
    }
    CHECK_ACL(aclrtSynchronizeStream(stream_));
    return 0;
}
void YoloDetectionNode::SetDataNode(std::shared_ptr<CollectorNode> collector, std::shared_ptr<RelayNode> relayer,std::shared_ptr<DistributorNode> distributor){
    collector_ = collector; 
    relayer_ = relayer; 
    distributor_ = distributor;
    worker_ = std::thread(&YoloDetectionNode::DetectThreadLoop, this);
}
void YoloDetectionNode::DetectThreadLoop(){
    thread_run_flag_ = true;
    CHECK_ACL(aclrtSetDevice(device_id_));
    TimeMetrics time_for_log;
    time_for_log.startTimer();
    while (!abort_) {
        if(!collector_){
            log_error("No data source available");
            return;
        }
        TimeMetrics t_detect;
        int list_size;
        std::vector<ImgPacket*> packets= collector_->GetBatch(batch_size_, list_size);
        if(packets.empty()){
            continue;
        }
        int model_img_size = input_h_ * input_w_ * 3;
        int buffer_idx = 0;
        std::vector<std::tuple<float, float, float>> res_pre;
        TimeMetrics t;
        t.startTimer();
        for(int i = 0; i < packets.size(); i++){
            ImgPacket *packet = packets[i];
            MemAllocate(packet->context);
            int img_size = packet->context->width * packet->context->height * 3;
            CHECK_ACL(aclrtMemcpyAsync(packet->context->img_buffer, img_size, packet->img.data, img_size, ACL_MEMCPY_HOST_TO_DEVICE, stream_));
            CHECK_ACL(aclrtSynchronizeStream(stream_));
            LetterBox(channel_id_letterbox_, packet->context->img_buffer, img_size, packet->context->width, packet->context->height, input_addr_img_ + buffer_idx, model_img_size, input_w_, input_h_);
            buffer_idx += input_h_ * input_w_ * 3;

            int new_unpad_w, new_unpad_h;
            float r = GetUnpadSize(input_w_, input_h_, packet->context->width, packet->context->height, new_unpad_w, new_unpad_h);
            float dw = input_w_ - new_unpad_w;
            float dh = input_h_ - new_unpad_h;
            dw /= 2.0f;
            dh /= 2.0f;
            res_pre.push_back(std::make_tuple(r, dw, dh));
        }
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
            float* feat_b = output_prob_ + b * one_output_len;
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
            log_debug("detect_all_time:{} pre_time:{} infer_time:{} after_time:{} list_size:{}", detect_all_time, pre_time, infer_time, after_time, list_size);
        }
    }
    log_debug("DetectThreadLoop finished");
}
#endif // DETECTION_ASCEND