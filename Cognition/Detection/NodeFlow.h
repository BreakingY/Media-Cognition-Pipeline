#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#ifndef NODE_FLOW_H
#define NODE_FLOW_H
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "DetectionInfo.h"
#include "log_helpers.h"
#include "DataChannel.h"
#include "YoloDetectionNode.h"
#include "TrackerNode.h"

void DetectModelInit(std::string eng_path, int device_id);
void* AddStream(InferDataListner* listener, int width, int height, int fps = 30);
void EndStream(void* context);
void StreamPushData(cv::Mat &img, void* context);
#endif // NODE_FLOW_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND