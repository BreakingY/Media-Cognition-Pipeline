#ifndef __DVPP_COMM_H__
#define __DVPP_COMM_H__
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)
#define SAMPLE_PRT(fmt...)    \
    do { \
        printf("[%s]-%d: ", __FUNCTION__, __LINE__); \
        printf(fmt); \
    } while (0)
uint32_t configure_stride_and_buffer_size(hi_vpc_pic_info& pic, uint32_t widthAlign = 16,
    uint32_t heightAlign = 2, bool widthStride32Align = true);

int32_t prepare_input_data_from_host(hi_vpc_pic_info& inputPic, const char *srcAddr_host, int src_width_stride, int src_height_stride);
int32_t prepare_input_data_from_device(hi_vpc_pic_info& inputPic, const char *srcAddr_device, int src_width_stride, int src_height_stride);
int32_t handle_output_data_from_device_to_host(const char *dstAddr_host, int dst_width_stride, int dst_height_stride, hi_vpc_pic_info& inputPic);
int32_t handle_output_data_from_device_to_device(const char *dstAddr_device, int dst_width_stride, int dst_height_stride, hi_vpc_pic_info& inputPic);

#endif