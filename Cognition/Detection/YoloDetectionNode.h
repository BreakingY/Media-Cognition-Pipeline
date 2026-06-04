#ifndef YOLO_DETECTION_NODE_H
#define YOLO_DETECTION_NODE_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>
#include <cmath>
#include <thread>
#include <memory>
#include <opencv2/opencv.hpp>
#include "DetectionInfo.h"
#include "log_helpers.h"
#include "DataChannel.h"
#include "TimeMetrics.h"
#if defined(DETECTION_NVIDIA)
#include <npp.h>
#include <NvInfer.h>
// version TensorRT-10.4.0.26
// ultralytics/ultralytics
class YoloDetectionNode{
public:
    YoloDetectionNode(std::string eng_path, int device_id);
    ~YoloDetectionNode();
    int Inference(const int batch_size);
    void SetDataNode(std::shared_ptr<CollectorNode> collector = nullptr, std::shared_ptr<RelayNode> relayer = nullptr, std::shared_ptr<DistributorNode> distributor = nullptr);
    void DetectThreadLoop();
private:
    std::string eng_path_;
    int device_id_;
    cudaStream_t stream_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<std::pair<int, std::string>> in_tensor_info_;
	std::vector<std::pair<int, std::string>> out_tensor_info_;
    int batch_size_ = 4;
    int input_h_;
    int input_w_;
    int output_pred_; // 4(c_x, c_y, w, h) + len(class_names)
	int anchors_; // The total number of anchors after the fusion of three feature maps
    const char *class_names_[2] = {"dog", "person"};
    void* buffers_[2] = {nullptr, nullptr};
    float* output_ = nullptr;

    std::thread worker_;
    bool abort_ = false;
    bool thread_run_flag_ = false;
    std::shared_ptr<CollectorNode> collector_;
    std::shared_ptr<RelayNode> relayer_;
    std::shared_ptr<DistributorNode> distributor_;

    Npp8u *pu8_rgb_ = nullptr;
    float* buffer_chw_ = nullptr;
}; 
#endif // DETECTION_NVIDIA
#if defined(DETECTION_ASCEND)
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#include <ops/acl_dvpp.h>
// version CANN7.0.0/8.2.RC1
// ultralytics/ultralytics
class YoloDetectionNode{
public:
    YoloDetectionNode(std::string eng_path, int device_id);
    ~YoloDetectionNode();
    int Inference(const int batch_size);
    void SetDataNode(std::shared_ptr<CollectorNode> collector = nullptr, std::shared_ptr<RelayNode> relayer = nullptr, std::shared_ptr<DistributorNode> distributor = nullptr);
    void DetectThreadLoop();
    void setInput();
private:
    std::string eng_path_;
    int device_id_;
    uint32_t model_id_;
    aclmdlDesc *model_desc_ = nullptr;
    aclmdlDataset *output_ = nullptr;
    size_t outputs_num_;
    std::vector<void*> output_buf_;
    std::vector<size_t> output_size_;
    std::vector<aclDataBuffer*> output_data_buf_;
    aclmdlDataset *input_ = nullptr;
    aclDataBuffer* data_buf_input_0_;
    uint32_t input_num_;
    void *input_addr_img_ = nullptr;
    

    size_t aipp_index_;
    void *input_AIPP_ = nullptr;

    size_t dynamic_batch_idx;
    void *input_batch_ = nullptr;
    hi_vpc_chn channel_id_letterbox_;

    aclrtStream stream_;
    
    int batch_size_ = 4;
    int input_h_;
    int input_w_;
    int output_pred_; // 4(c_x, c_y, w, h) + len(class_names)
	int anchors_; // The total number of anchors after the fusion of three feature maps
    const char *class_names_[2] = {"dog", "person"};
    float* output_prob_ = nullptr;
    int output_prob_len_;

    std::thread worker_;
    bool abort_ = false;
    bool thread_run_flag_ = false;
    std::shared_ptr<CollectorNode> collector_;
    std::shared_ptr<RelayNode> relayer_;
    std::shared_ptr<DistributorNode> distributor_;
}; 
#endif // DETECTION_ASCEND
#endif // YOLO_DETECTION_NODE_H
