#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
#include "dataProcess.h"
#define threadNum 256
#define WARP_SIZE 64
#define elemsPerThread 1
/**
 * letter box resize
 * 原始图像(BGR)：originalImage、w、h
 * 原图resize后的实际尺寸：w2、h2
 * 模型输入尺寸：w3、h3
 * 填充信息：x_b( (w3 - w2)/2 )、y_b( (h3 - h2)/2 )
 * letter box目的地址(CHW)：channelImage(float类型)
 * letter box目的地地址(HWC、 BGR for opencv)：resizedImage(unsigned char类型)
 * note: 如果把x_b、y_b都设置为0，并且w3、h3与w2、h2相等，就是普通的resize
 */
__global__ void letter_box_kernel(unsigned char *originalImage, unsigned char *resizedImage, float *channelImage, int w, int h, int w2, int h2, int w3, int h3, int x_b, int y_b)
{
    int32_t tile;
    const float x_ratio = ((float)(w)) / w2;
    const float y_ratio = ((float)(h)) / h2;
    unsigned int threadId = blockIdx.x * threadNum * elemsPerThread + threadIdx.x * elemsPerThread;
    unsigned int shift = 0;
    unsigned int outer_idx = 0;
    while ((threadId < w2 * h2 && shift < elemsPerThread)) {
        const int32_t i = threadId / w2;
        const int32_t j = threadId - (i * w2);

        outer_idx = (i + y_b) * w3 + j + x_b;
        const int32_t x = (int)(x_ratio * j);
        const int32_t y = (int)(y_ratio * i);
        const float x_diff = (x_ratio * j) - x;
        const float y_diff = (y_ratio * i) - y;
        const int32_t index = (y * w + x);
        const unsigned char a_b = originalImage[index * 3];
        const unsigned char a_g = originalImage[index * 3 + 1];
        const unsigned char a_r = originalImage[index * 3 + 2];
        const int32_t a = 0xff000000 | ((((int32_t)a_r) << 16) & 0xff0000) | ((((int32_t)a_g) << 8) & 0xff00) | ((int32_t)a_b);

        unsigned char b_b = 0;
        unsigned char b_g = 0;
        unsigned char b_r = 0;

        if (x + 1 >= w) {
            b_b = a_b;
            b_g = a_g;
            b_r = a_r;
        } else {
            b_b = originalImage[(index + 1) * 3];
            b_g = originalImage[(index + 1) * 3 + 1];
            b_r = originalImage[(index * 1) * 3 + 2];
        }
        const int32_t b = 0xff000000 | ((((int32_t)b_r) << 16) & 0xff0000) | ((((int32_t)b_g) << 8) & 0xff00) | ((int32_t)b_b);

        unsigned char c_b = 0;
        unsigned char c_g = 0;
        unsigned char c_r = 0;

        if (y + 1 >= h) {
            c_b = a_b;
            c_g = a_g;
            c_r = a_r;
        } else {
            c_b = originalImage[(index + w) * 3];
            c_g = originalImage[(index + w) * 3 + 1];
            c_r = originalImage[(index + w) * 3 + 2];
        }

        const int32_t c = 0xff000000 | ((((int32_t)c_r) << 16) & 0xff0000) | ((((int32_t)c_g) << 8) & 0xff00) | ((int32_t)c_b);

        unsigned char d_b = 0;
        unsigned char d_g = 0;
        unsigned char d_r = 0;

        if (x + 1 >= w || y + 1 >= h) {
            d_b = a_b;
            d_g = a_g;
            d_r = a_r;
        } else {
            d_b = originalImage[(index + w + 1) * 3];
            d_g = originalImage[(index + w + 1) * 3 + 1];
            d_r = originalImage[(index + w + 1) * 3 + 2];
        }

        const int32_t d = 0xff000000 | ((((int32_t)d_r) << 16) & 0xff0000) | ((((int32_t)d_g) << 8) & 0xff00) | ((int32_t)d_b);

        const float blue = (a & 0xff) * (1 - x_diff) * (1 - y_diff) + (b & 0xff) * (x_diff) * (1 - y_diff) + (c & 0xff) * (y_diff) * (1 - x_diff) + (d & 0xff) * (x_diff * y_diff);

        const float green = ((a >> 8) & 0xff) * (1 - x_diff) * (1 - y_diff) + ((b >> 8) & 0xff) * (x_diff) * (1 - y_diff) + ((c >> 8) & 0xff) * (y_diff) * (1 - x_diff) + ((d >> 8) & 0xff) * (x_diff * y_diff);

        const float red = ((a >> 16) & 0xff) * (1 - x_diff) * (1 - y_diff) + ((b >> 16) & 0xff) * (x_diff) * (1 - y_diff) + ((c >> 16) & 0xff) * (y_diff) * (1 - x_diff) + ((d >> 16) & 0xff) * (x_diff * y_diff);

        tile = 0xff000000 | ((((int32_t)red) << 16) & 0xff0000) | ((((int32_t)green) << 8) & 0xff00) | ((int32_t)blue);

        threadId++;
        shift++;
    }

    int32_t c_blue = tile & 0xff;
    int32_t c_green = ((tile >> 8) & 0xff);
    int32_t c_red = ((tile >> 16) & 0xff);

    if (resizedImage) {
        resizedImage[outer_idx * 3] = c_blue;
        resizedImage[outer_idx * 3 + 1] = c_green;
        resizedImage[outer_idx * 3 + 2] = c_red;
    }

    if (channelImage) {
        channelImage[outer_idx] = (float)c_red / 255.0;
        channelImage[outer_idx + w3 * h3] = (float)c_green / 255.0;
        channelImage[outer_idx + 2 * w3 * h3] = (float)c_blue / 255.0;
    }
    return;
}
void preprocess_letter_bbox_resize(unsigned char *src, unsigned char *dst_hwc, float *dst_chw, int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream){
    int w, h, x, y;
    float r_w = dst_w / (src_w * 1.0);
    float r_h = dst_h / (src_h * 1.0);
    if (r_h > r_w) {
        w = dst_w;
        h = r_w * src_h;
        x = 0;
        y = (dst_h - h) / 2;
    } else {
        w = r_h * src_w;
        h = dst_h;
        x = (dst_w - w) / 2;
        y = 0;
    }
    dim3 threads = dim3(threadNum, 1, 1);
    dim3 blocks = dim3(w * h / threadNum * elemsPerThread, 1, 1);
    letter_box_kernel<<<blocks, threads, 0, stream>>>(src, dst_hwc, dst_chw, src_w, src_h, w, h, dst_w, dst_h, x, y);
    CHECK_CUDA(cudaGetLastError());
    return;
}
#endif // DETECTION_NVIDIA DETECTION_HYGON