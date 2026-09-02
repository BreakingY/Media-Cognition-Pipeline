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

/*
 * 通用的带 stride 图片数据拷贝函数
 *
 * 功能：
 *   在外部 buffer 和 hi_vpc_pic_info::picture_address 之间进行数据拷贝。
 *
 * 参数：
 *   pic             : DVPP 图片信息，包含图片格式、实际宽高以及 device stride
 *   srcAddr         : 源地址
 *   srcWidthStride  : 源 buffer 的 width stride
 *   srcHeightStride : 源 buffer 的 height stride
 *   dstAddr         : 目标地址
 *   dstWidthStride  : 目标 buffer 的 width stride
 *   dstHeightStride : 目标 buffer 的 height stride
 *   kind            : ACL memcpy 类型
 *
 * 注意：
 *   这里的 src/dst 是真正的数据源和目标，而不是固定代表 host/device。
 */
static int32_t copy_picture_data(hi_vpc_pic_info& pic,
                                 const uint8_t* srcAddr,
                                 uint32_t srcWidthStride,
                                 uint32_t srcHeightStride,
                                 uint8_t* dstAddr,
                                 uint32_t dstWidthStride,
                                 uint32_t dstHeightStride,
                                 aclrtMemcpyKind kind)
{
    if (!srcAddr || !dstAddr) {
        SAMPLE_PRT("srcAddr or dstAddr is nullptr!\n");
        return -1;
    }

    const uint32_t width  = pic.picture_width;
    const uint32_t height = pic.picture_height;
    
    /*
     * pic.picture_width_stride 是 DVPP/device buffer 的 stride。
     *
     * 这里虽然通用函数同时接收 src/dst stride，
     * 但调用者可以直接把实际 stride 传进来。
     */
    auto copy_plane =
        [&](uint8_t* dst,
            const uint8_t* src,
            uint32_t realWidth,
            uint32_t realHeight,
            uint32_t srcStride,
            uint32_t dstStride) -> int32_t
    {
        if (srcStride == dstStride) {
            CHECK_ACL(aclrtMemcpy(
                dst,
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

        return 0;
    };

    switch (pic.picture_format)
    {
    // -------------------------------------------------------------------------
    // YUV 400
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_YUV_400:
    {
        copy_plane(
            dstAddr,
            srcAddr,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        break;
    }

    // -------------------------------------------------------------------------
    // YUV 420 Semi-Planar
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_420:
    {
        const uint32_t uvHeight = height / 2;

        // Y plane
        copy_plane(
            dstAddr,
            srcAddr,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        // UV plane
        //
        // 源 UV 起始地址：
        //   srcWidthStride * srcHeightStride
        //
        // 目标 UV 起始地址：
        //   dstWidthStride * dstHeightStride
        const uint8_t* srcUV =
            srcAddr + srcWidthStride * srcHeightStride;

        uint8_t* dstUV =
            dstAddr + dstWidthStride * dstHeightStride;

        copy_plane(
            dstUV,
            srcUV,
            width,
            uvHeight,
            srcWidthStride,
            dstWidthStride);

        break;
    }

    // -------------------------------------------------------------------------
    // YUV 422 Semi-Planar
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_422:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_422:
    {
        // Y plane
        copy_plane(
            dstAddr,
            srcAddr,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        // UV plane
        const uint8_t* srcUV =
            srcAddr + srcWidthStride * srcHeightStride;

        uint8_t* dstUV =
            dstAddr + dstWidthStride * dstHeightStride;

        copy_plane(
            dstUV,
            srcUV,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        break;
    }

    // -------------------------------------------------------------------------
    // YUV 444 Semi-Planar
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_YUV_SEMIPLANAR_444:
    case HI_PIXEL_FORMAT_YVU_SEMIPLANAR_444:
    {
        // Y plane
        copy_plane(
            dstAddr,
            srcAddr,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        // UV plane
        const uint8_t* srcUV =
            srcAddr + srcWidthStride * srcHeightStride;

        uint8_t* dstUV =
            dstAddr + dstWidthStride * dstHeightStride;

        copy_plane(
            dstUV,
            srcUV,
            width,
            height,
            srcWidthStride,
            dstWidthStride);

        break;
    }

    // -------------------------------------------------------------------------
    // Packed 422
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_YUYV_PACKED_422:
    case HI_PIXEL_FORMAT_UYVY_PACKED_422:
    case HI_PIXEL_FORMAT_YVYU_PACKED_422:
    case HI_PIXEL_FORMAT_VYUY_PACKED_422:
    {
        const uint32_t bytesPerPixel = 2;
        const uint32_t realRowBytes = width * bytesPerPixel;

        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dstAddr + h * dstWidthStride,
                realRowBytes,
                srcAddr + h * srcWidthStride,
                realRowBytes,
                kind));
        }

        break;
    }

    // -------------------------------------------------------------------------
    // RGB/BGR 888 / YUV 444 Packed
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_RGB_888:
    case HI_PIXEL_FORMAT_BGR_888:
    case HI_PIXEL_FORMAT_YUV_PACKED_444:
    {
        const uint32_t bytesPerPixel = 3;
        const uint32_t realRowBytes = width * bytesPerPixel;

        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dstAddr + h * dstWidthStride,
                realRowBytes,
                srcAddr + h * srcWidthStride,
                realRowBytes,
                kind));
        }

        break;
    }

    // -------------------------------------------------------------------------
    // ARGB / ABGR / RGBA / BGRA / FLOAT32
    // -------------------------------------------------------------------------
    case HI_PIXEL_FORMAT_ARGB_8888:
    case HI_PIXEL_FORMAT_ABGR_8888:
    case HI_PIXEL_FORMAT_RGBA_8888:
    case HI_PIXEL_FORMAT_BGRA_8888:
    case HI_PIXEL_FORMAT_FLOAT32:
    {
        const uint32_t bytesPerPixel = 4;
        const uint32_t realRowBytes = width * bytesPerPixel;

        for (uint32_t h = 0; h < height; ++h) {
            CHECK_ACL(aclrtMemcpy(
                dstAddr + h * dstWidthStride,
                realRowBytes,
                srcAddr + h * srcWidthStride,
                realRowBytes,
                kind));
        }

        break;
    }

    default:
        SAMPLE_PRT(
            "Unsupported format = %u\n",
            pic.picture_format);
        return -1;
    }

    return 0;
}

/*
 * Host -> Device
 *
 * 外部 host buffer
 *        |
 *        | srcWidthStride
 *        v
 *   inputPic.picture_address
 *        |
 *        | inputPic.picture_width_stride
 *        v
 */
int32_t prepare_input_data_from_host(hi_vpc_pic_info& inputPic, const char *srcAddr_host, int src_width_stride, int src_height_stride)
{
    if (!inputPic.picture_address) {
        SAMPLE_PRT("inputPic.picture_address is nullptr!\n");
        return -1;
    }

    return copy_picture_data(
        inputPic,
        reinterpret_cast<const uint8_t*>(srcAddr_host),
        static_cast<uint32_t>(src_width_stride),
        static_cast<uint32_t>(src_height_stride),
        reinterpret_cast<uint8_t*>(inputPic.picture_address),
        inputPic.picture_width_stride,
        inputPic.picture_height_stride,
        ACL_MEMCPY_HOST_TO_DEVICE);
}


/*
 * Device -> Device
 *
 * 外部 device buffer
 *        |
 *        | srcWidthStride
 *        v
 *   inputPic.picture_address
 *        |
 *        | inputPic.picture_width_stride
 *        v
 */
int32_t prepare_input_data_from_device(hi_vpc_pic_info& inputPic,
                                       const char *srcAddr_device,
                                       int src_width_stride,
                                       int src_height_stride)
{
    if (!inputPic.picture_address) {
        SAMPLE_PRT("inputPic.picture_address is nullptr!\n");
        return -1;
    }

    return copy_picture_data(
        inputPic,
        reinterpret_cast<const uint8_t*>(srcAddr_device),
        static_cast<uint32_t>(src_width_stride),
        static_cast<uint32_t>(src_height_stride),
        reinterpret_cast<uint8_t*>(inputPic.picture_address),
        inputPic.picture_width_stride,
        inputPic.picture_height_stride,
        ACL_MEMCPY_DEVICE_TO_DEVICE);
}


/*
 * Device -> Host
 *
 *   inputPic.picture_address
 *        |
 *        | inputPic.picture_width_stride
 *        v
 *   外部 host buffer
 *        |
 *        | dstWidthStride
 *        v
 */
int32_t handle_output_data_from_device_to_host(
    const char *dstAddr_host,
    int dst_width_stride,
    int dst_height_stride,
    hi_vpc_pic_info& inputPic)
{
    if (!inputPic.picture_address) {
        SAMPLE_PRT("inputPic.picture_address is nullptr!\n");
        return -1;
    }

    return copy_picture_data(
        inputPic,
        reinterpret_cast<const uint8_t*>(inputPic.picture_address),
        inputPic.picture_width_stride,
        inputPic.picture_height_stride,
        reinterpret_cast<uint8_t*>(const_cast<char*>(dstAddr_host)),
        static_cast<uint32_t>(dst_width_stride),
        static_cast<uint32_t>(dst_height_stride),
        ACL_MEMCPY_DEVICE_TO_HOST);
}


/*
 * Device -> Device
 *
 *   inputPic.picture_address
 *        |
 *        | inputPic.picture_width_stride
 *        v
 *   外部 device buffer
 *        |
 *        | dstWidthStride
 *        v
 */
int32_t handle_output_data_from_device_to_device(
    const char *dstAddr_device,
    int dst_width_stride,
    int dst_height_stride,
    hi_vpc_pic_info& inputPic)
{
    if (!inputPic.picture_address) {
        SAMPLE_PRT("inputPic.picture_address is nullptr!\n");
        return -1;
    }

    return copy_picture_data(
        inputPic,
        reinterpret_cast<const uint8_t*>(inputPic.picture_address),
        inputPic.picture_width_stride,
        inputPic.picture_height_stride,
        reinterpret_cast<uint8_t*>(const_cast<char*>(dstAddr_device)),
        static_cast<uint32_t>(dst_width_stride),
        static_cast<uint32_t>(dst_height_stride),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
}