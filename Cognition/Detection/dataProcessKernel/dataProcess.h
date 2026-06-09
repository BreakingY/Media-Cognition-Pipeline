#if defined(DETECTION_HYGON)
#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H
#include <stdint.h>
#include <cuda_runtime.h>
#include "DetectionInfo.h"

void hwc_To_chw_normalize_float_rgb(void *pu8_rgb, void *buffer,int input_w, int input_h, cudaStream_t stream = 0);
#endif // DATA_PROCESS_H
#endif // DETECTION_HYGON