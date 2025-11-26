#include "dvpp_common.h"

uint32_t configure_stride_and_buffer_size(hi_vpc_pic_info& pic, uint32_t widthAlign, uint32_t heightAlign,
    bool widthStride32Align)
{
    if ((widthAlign == 0) || (widthAlign > 128) || ((widthAlign & (widthAlign - 1)) != 0)) {
        return 0;
    }
    if ((heightAlign == 0) || (heightAlign > 128) || ((heightAlign & (heightAlign - 1)) != 0)) {
        return 0;
    }

    uint32_t width = pic.picture_width;
    uint32_t height = pic.picture_height;
    uint32_t format = pic.picture_format;
    uint32_t dstBufferSize = 0;
    uint32_t minWidthAlignNum = widthStride32Align ? 32 : 1;

    switch (format) {
        case HI_PIXEL_FORMAT_YUV_400:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign);
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);
            if (pic.picture_width_stride < minWidthAlignNum) pic.picture_width_stride = minWidthAlignNum;

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride;
            dstBufferSize = width * height;
            break;

        case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign);
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);
            if (pic.picture_width_stride < minWidthAlignNum) pic.picture_width_stride = minWidthAlignNum;

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride * 3 / 2;
            dstBufferSize = width * height * 3 / 2;
            break;

        case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_440:
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_440:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign);
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);
            if (pic.picture_width_stride < minWidthAlignNum) pic.picture_width_stride = minWidthAlignNum;

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride * 2;
            dstBufferSize = width * height * 2;
            break;

        case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422:
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign);
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);
            if (pic.picture_width_stride < minWidthAlignNum) pic.picture_width_stride = minWidthAlignNum;

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride * 2;
            dstBufferSize = width * height * 2;
            break;

        case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444:
        case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign);
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);
            if (pic.picture_width_stride < minWidthAlignNum) pic.picture_width_stride = minWidthAlignNum;

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride * 3;
            dstBufferSize = width * height * 3;
            break;

        case HI_PIXEL_FORMAT_YUYV_PACKED_422:
        case HI_PIXEL_FORMAT_UYVY_PACKED_422:
        case HI_PIXEL_FORMAT_YVYU_PACKED_422:
        case HI_PIXEL_FORMAT_VYUY_PACKED_422:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign) * 2;
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride;
            dstBufferSize = width * height * 2;
            break;

        case HI_PIXEL_FORMAT_YUV_PACKED_444:
        case HI_PIXEL_FORMAT_RGB_888:
        case HI_PIXEL_FORMAT_BGR_888:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign) * 3;
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride;
            dstBufferSize = width * height * 3;
            break;

        case HI_PIXEL_FORMAT_ARGB_8888:
        case HI_PIXEL_FORMAT_ABGR_8888:
        case HI_PIXEL_FORMAT_RGBA_8888:
        case HI_PIXEL_FORMAT_BGRA_8888:
        case HI_PIXEL_FORMAT_FLOAT32:
            pic.picture_width_stride = ALIGN_UP(width, widthAlign) * 4;
            pic.picture_height_stride = ALIGN_UP(height, heightAlign);

            pic.picture_buffer_size = pic.picture_width_stride * pic.picture_height_stride;
            dstBufferSize = width * height * 4;
            break;

        default:
            pic.picture_buffer_size = 0;
            dstBufferSize = 0;
            break;
    }

    return dstBufferSize;
}

static int32_t prepare_input_data(hi_vpc_pic_info& pic,
                                  const char *srcAddr,
                                  int src_width_stride,
                                  int src_height_stride,
                                  aclrtMemcpyKind kind)
{
    if (!srcAddr || !pic.picture_address) return -1;

    const uint32_t width   = pic.picture_width;
    const uint32_t height  = pic.picture_height;
    const uint32_t wStride = pic.picture_width_stride;

    uint8_t* dst_base       = reinterpret_cast<uint8_t*>(pic.picture_address);
    const uint8_t* src_base = reinterpret_cast<const uint8_t*>(srcAddr);

    auto copy_plane = [&](uint8_t* dst, const uint8_t* src,
                          uint32_t realWidth, uint32_t realHeight,
                          uint32_t dstStride, uint32_t srcStride)
    {
        if (dstStride == srcStride) {
            CHECK_ACL(aclrtMemcpy(dst, dstStride * realHeight, src, srcStride * realHeight, kind));
        } else {
            for (uint32_t h = 0; h < realHeight; ++h) {
                CHECK_ACL(aclrtMemcpy(
                    dst + h * dstStride,
                    realWidth,
                    src + h * srcStride,
                    realWidth,
                    kind));
            }
        }
    };

    switch (pic.picture_format)
    {
    case HI_PIXEL_FORMAT_YUV_400:
        copy_plane(dst_base, src_base, width, height, wStride, src_width_stride);
        break;

    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
    {
        uint32_t uvHeight = height / 2;
        uint32_t uvOffset = wStride * height;

        copy_plane(dst_base, src_base, width, height, wStride, src_width_stride);

        uint8_t* dst_uv = dst_base + uvOffset;
        const uint8_t* src_uv = src_base + src_width_stride * height;
        copy_plane(dst_uv, src_uv, width, uvHeight, wStride, src_width_stride);
        break;
    }

    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422:
    {
        uint32_t uvOffset = wStride * height;

        copy_plane(dst_base, src_base, width, height, wStride, src_width_stride);

        uint8_t* dst_uv = dst_base + uvOffset;
        const uint8_t* src_uv = src_base + src_width_stride * height;
        copy_plane(dst_uv, src_uv, width, height, wStride, src_width_stride);
        break;
    }

    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444:
    {
        uint32_t uvOffset = wStride * height;

        copy_plane(dst_base, src_base, width, height, wStride, src_width_stride);

        uint8_t* dst_uv = dst_base + uvOffset;
        const uint8_t* src_uv = src_base + src_width_stride * height;
        copy_plane(dst_uv, src_uv, width, height, wStride, src_width_stride);
        break;
    }

    case HI_PIXEL_FORMAT_YUYV_PACKED_422:
    case HI_PIXEL_FORMAT_UYVY_PACKED_422:
    case HI_PIXEL_FORMAT_YVYU_PACKED_422:
    case HI_PIXEL_FORMAT_VYUY_PACKED_422:
    {
        uint32_t bytesPerPixel = 2;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * wStride,
                width * bytesPerPixel,
                src_base + h * src_width_stride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    case HI_PIXEL_FORMAT_RGB_888:
    case HI_PIXEL_FORMAT_BGR_888:
    case HI_PIXEL_FORMAT_YUV_PACKED_444:
    {
        uint32_t bytesPerPixel = 3;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * wStride,
                width * bytesPerPixel,
                src_base + h * src_width_stride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    case HI_PIXEL_FORMAT_ARGB_8888:
    case HI_PIXEL_FORMAT_ABGR_8888:
    case HI_PIXEL_FORMAT_RGBA_8888:
    case HI_PIXEL_FORMAT_BGRA_8888:
    case HI_PIXEL_FORMAT_FLOAT32:
    {
        uint32_t bytesPerPixel = 4;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * wStride,
                width * bytesPerPixel,
                src_base + h * src_width_stride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    default:
        return -1;
    }

    return 0;
}

int32_t prepare_input_data_from_host(hi_vpc_pic_info& inputPic,
                                     const char *srcAddr_host,
                                     int src_width_stride,
                                     int src_height_stride)
{
    return prepare_input_data(inputPic, srcAddr_host, src_width_stride, src_height_stride, ACL_MEMCPY_HOST_TO_DEVICE);
}

int32_t prepare_input_data_from_device(hi_vpc_pic_info& inputPic,
                                       const char *srcAddr_device,
                                       int src_width_stride,
                                       int src_height_stride)
{
    return prepare_input_data(inputPic, srcAddr_device, src_width_stride, src_height_stride, ACL_MEMCPY_DEVICE_TO_DEVICE);
}
static int32_t handle_output_data_common(const char *dstAddr,
                                         int dst_width_stride,
                                         int dst_height_stride,
                                         hi_vpc_pic_info& inputPic,
                                         aclrtMemcpyKind kind)
{
    if (!dstAddr || !inputPic.picture_address) {
        SAMPLE_PRT("dstAddr or inputPic.picture_address is nullptr!\n");
        return -1;
    }

    const uint32_t width   = inputPic.picture_width;
    const uint32_t height  = inputPic.picture_height;
    const uint32_t wStride = inputPic.picture_width_stride;  // device stride

    const uint8_t* src_base = reinterpret_cast<const uint8_t*>(inputPic.picture_address);
    uint8_t* dst_base       = reinterpret_cast<uint8_t*>(const_cast<char*>(dstAddr));

    auto copy_plane = [&](uint8_t* dst, const uint8_t* src,
                          uint32_t realWidth, uint32_t realHeight,
                          uint32_t srcStride, uint32_t dstStride)
    {
        if (srcStride == dstStride) {
            CHECK_ACL(aclrtMemcpy(dst,
                                  srcStride * realHeight,
                                  src,
                                  srcStride * realHeight,
                                  kind));
        } else {
            for (uint32_t h = 0; h < realHeight; ++h) {
                CHECK_ACL(aclrtMemcpy(
                    dst + h * dstStride,
                    realWidth,
                    src + h * srcStride,
                    realWidth,
                    kind));
            }
        }
    };

    switch (inputPic.picture_format)
    {
    // --------------------- Y-only ---------------------
    case HI_PIXEL_FORMAT_YUV_400:
    {
        copy_plane(dst_base, src_base,
                   width, height,
                   wStride, dst_width_stride);
        break;
    }

    // --------------------- Semi-planar 420 ---------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
    {
        // Y plane
        copy_plane(dst_base, src_base,
                   width, height,
                   wStride, dst_width_stride);

        // UV plane
        uint32_t uvOffset = wStride * height;
        uint32_t uvHeight = height / 2;

        uint8_t* dst_uv = dst_base + dst_width_stride * dst_height_stride; // 按 dstStride 累加
        const uint8_t* src_uv = src_base + uvOffset;

        copy_plane(dst_uv, src_uv,
                   width, uvHeight,
                   wStride, dst_width_stride);
        break;
    }

    // --------------------- Semi-planar 422 ---------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422:
    {
        copy_plane(dst_base, src_base,
                   width, height,
                   wStride, dst_width_stride);

        uint32_t uvOffset = wStride * height;

        uint8_t* dst_uv = dst_base + dst_width_stride * dst_height_stride;
        const uint8_t* src_uv = src_base + uvOffset;

        copy_plane(dst_uv, src_uv,
                   width, height,
                   wStride, dst_width_stride);
        break;
    }

    // --------------------- Semi-planar 444 ---------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444:
    {
        copy_plane(dst_base, src_base,
                   width, height,
                   wStride, dst_width_stride);

        uint32_t uvOffset = wStride * height;

        uint8_t* dst_uv = dst_base + dst_width_stride * dst_height_stride;
        const uint8_t* src_uv = src_base + uvOffset;

        copy_plane(dst_uv, src_uv,
                   width, height,
                   wStride, dst_width_stride);
        break;
    }

    // --------------------- Packed 422 ---------------------
    case HI_PIXEL_FORMAT_YUYV_PACKED_422:
    case HI_PIXEL_FORMAT_UYVY_PACKED_422:
    case HI_PIXEL_FORMAT_YVYU_PACKED_422:
    case HI_PIXEL_FORMAT_VYUY_PACKED_422:
    {
        uint32_t bytesPerPixel = 2;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * dst_width_stride,
                width * bytesPerPixel,
                src_base + h * wStride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    // --------------------- Packed RGB/BGR 888 / YUV 444 ---------------------
    case HI_PIXEL_FORMAT_RGB_888:
    case HI_PIXEL_FORMAT_BGR_888:
    case HI_PIXEL_FORMAT_YUV_PACKED_444:
    {
        uint32_t bytesPerPixel = 3;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * dst_width_stride,
                width * bytesPerPixel,
                src_base + h * wStride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    // --------------------- ARGB / ABGR / RGBA / BGRA / FLOAT32 ---------------------
    case HI_PIXEL_FORMAT_ARGB_8888:
    case HI_PIXEL_FORMAT_ABGR_8888:
    case HI_PIXEL_FORMAT_RGBA_8888:
    case HI_PIXEL_FORMAT_BGRA_8888:
    case HI_PIXEL_FORMAT_FLOAT32:
    {
        uint32_t bytesPerPixel = 4;
        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dst_base + h * dst_width_stride,
                width * bytesPerPixel,
                src_base + h * wStride,
                width * bytesPerPixel,
                kind));
        }
        break;
    }

    default:
        SAMPLE_PRT("Unsupported format = %u\n", inputPic.picture_format);
        return -1;
    }

    return 0;
}

int32_t handle_output_data_from_device_to_host(const char *dstAddr_host,
                                               int dst_width_stride,
                                               int dst_height_stride,
                                               hi_vpc_pic_info& inputPic)
{
    return handle_output_data_common(dstAddr_host,
                                     dst_width_stride,
                                     dst_height_stride,
                                     inputPic,
                                     ACL_MEMCPY_DEVICE_TO_HOST);
}

int32_t handle_output_data_from_device_to_device(const char *dstAddr_device,
                                                 int dst_width_stride,
                                                 int dst_height_stride,
                                                 hi_vpc_pic_info& inputPic)
{
    return handle_output_data_common(dstAddr_device,
                                     dst_width_stride,
                                     dst_height_stride,
                                     inputPic,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
}


