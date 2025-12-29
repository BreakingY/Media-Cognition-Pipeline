#define DETECTION_NVIDIA
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
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
    void setDataNode(CollectorNode *collector = nullptr, 
                    RelayNode *relayer = nullptr, 
                    DistributorNode *distributor = nullptr){collector_ = collector; relayer_ = relayer; distributor_ = distributor;}
    void DetectThreadLoop();
private:
    std::string eng_path_;
    int device_id_;
    cudaStream_t stream_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<std::pair<int, std::string>> in_tensor_info_;
	std::vector<std::pair<int, std::string>> out_tensor_info_;
    int batch_size_ = 4;
    int input_h_;
    int input_w_;
    int output_pred; // 4(c_x, c_y, w, h) + len(class_names)
	int anchors; // The total number of anchors after the fusion of three feature maps
    const char *class_names[2] = {"dog", "person"};
    void* buffers_[2] = {nullptr, nullptr};
    float* output_ = nullptr;

    std::thread worker_;
    bool abort_ = false;
    CollectorNode *collector_ = nullptr;
    RelayNode *relayer_ = nullptr;
    DistributorNode *distributor_ = nullptr;
}; 

#endif // YOLO_DETECTION_NODE_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND