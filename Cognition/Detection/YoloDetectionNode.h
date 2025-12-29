#if defined(DETECTION_NVIDIA)
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
#include <npp.h>
#include <NvInfer.h>
#include "DetectionInfo.h"
#include "log_helpers.h"
#include "DataChannel.h"
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
}; 

#endif // YOLO_DETECTION_NODE_H
#endif // DETECTION_NVIDIA