#if defined(DETECTION_HYGON)
#include "dataProcess.h"
__global__ void hwc_To_chw_normalize_float_rgb_kernel(const uint8_t* src, float* dst, int width, int height){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_num = width * height;
    if (idx >= pixel_num)
        return;

    int src_idx = idx * 3;

    constexpr float scale = 1.0f / 255.0f;

    dst[idx]                 = src[src_idx + 0] * scale; // R
    dst[idx + pixel_num]     = src[src_idx + 1] * scale; // G
    dst[idx + pixel_num * 2] = src[src_idx + 2] * scale; // B
}
void hwc_To_chw_normalize_float_rgb(void *pu8_rgb, void *buffer,int input_w, int input_h, cudaStream_t stream){
    size_t  pixel_num = input_w * input_h;
    int threads = 256;
    int blocks = (pixel_num + threads - 1) / threads;
    hwc_To_chw_normalize_float_rgb_kernel<<<blocks, threads, 0, stream>>>((const uint8_t*)pu8_rgb, (float*)buffer, input_w, input_h);
    CHECK_CUDA(cudaGetLastError());
    return;
}
#endif // DETECTION_HYGON