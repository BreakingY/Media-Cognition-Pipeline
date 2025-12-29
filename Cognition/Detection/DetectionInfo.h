#define DETECTION_NVIDIA
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#ifndef DETECTION_INFO_H
#define DETECTION_INFO_H
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <stdio.h>
#include <stdilb.h>
#include <stdint.h>
#include <string.h>
typedef struct DetectionSt {
    cv::Rect2f box;   // l_x, l_y, w, h
    float score;
    int class_id;
    int track_id;
}Detection;
typedef struct DetectionInfoSt {
    std::vector<Detection> dets;
    std::vector<std::string> class_names;
}DetectionInfo;
#endif // DETECTION_INFO_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND