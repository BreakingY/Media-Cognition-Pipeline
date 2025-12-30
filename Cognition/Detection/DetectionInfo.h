#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#ifndef DETECTION_INFO_H
#define DETECTION_INFO_H
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#if defined(DETECTION_NVIDIA)
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
#elif defined(DETECTION_ASCEND)
#ifndef CHECK_ACL
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif
#ifndef CHECK_DVPP_MPI
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif
#else
// It's impossible for this to happen
#endif
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