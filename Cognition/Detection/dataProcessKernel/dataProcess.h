#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H
#include <stdint.h>
#include <cuda_runtime.h>
#include "DetectionInfo.h"
void preprocess_letter_bbox_resize(unsigned char *src, unsigned char *dst_hwc, float *dst_chw, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream = 0);
#endif // DATA_PROCESS_H
#endif // DETECTION_NVIDIA DETECTION_HYGON