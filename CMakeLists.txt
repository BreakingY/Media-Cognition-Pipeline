cmake_minimum_required(VERSION 3.10)
project(VIDEOCODECPROJ)

set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Feature switches
# ------------------------------------------------------------------------------

# input/output
option(ENABLE_SRT       "Enable srt streaming" OFF)

# codec
option(FFMPEG_SOFT      "Enable FFmpeg soft codec" OFF)
option(FFMPEG_NVIDIA    "Enable FFmpeg nvidia codec" OFF)
option(DVPP_MPI         "Enable Ascend DVPP V2 codec" OFF)
option(NVIDIA_SDK_X86   "Enable NVIDIA Video Codec SDK codec" OFF)
option(NVIDIA_SDK_ARM   "Enable Jetson codec" OFF)

# detection
option(DETECTION_NVIDIA "Enable Nvidia detection" OFF)
option(DETECTION_ASCEND "Enable Ascend detection" OFF)
option(DETECTION_HYGON  "Enable hygon detection" OFF)

# python interface
option(MCP_PYBIND      "Enable Python binding" OFF)

# Python binding mode does not use any detection module.
if(MCP_PYBIND)
    message(STATUS "MCP_PYBIND enabled: disable all detection modules")

    set(DETECTION_NVIDIA OFF CACHE BOOL "Disable Nvidia detection" FORCE)
    set(DETECTION_ASCEND OFF CACHE BOOL "Disable Ascend detection" FORCE)
    set(DETECTION_HYGON  OFF CACHE BOOL "Disable Hygon detection" FORCE)
endif()

if(WIN32)
    # FFmpeg
    include_directories("D:/msys64/mingw64/include")
    include_directories("D:/msys64/home/sunkx1/ff-build/include")
    link_directories("D:/msys64/mingw64/lib")
    link_directories("D:/msys64/home/sunkx1/ff-build/lib")

    # OpenCV
    include_directories("D:/opencv-mingw/install/include")
    link_directories("D:/opencv-mingw/install/x64/mingw/lib")

elseif(UNIX)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib /usr/lib)

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# ==============================================================================
# Common include directories
# ==============================================================================
include_directories(
    Warpper

    #------ Log ------
    Log
    third_party/spdlog/include
    #-----------------

    #------ Hardware codec ------
    HardCodec
    HardCodec/Decoder
    HardCodec/Encoder
    #----------------------------

    #------ input/output ------
    # Media
    Media/MediaCommon
    Media/MediaMuxer
    Media/MediaReader
    Media/MediaReader/FileReader
    Media/MediaReader/RtspReader
    Media/RtmpClient
    Media/TsTransport

    # H264 / H265
    third_party/Bitstream/h264/include
    third_party/Bitstream/h265/include

    # RTSP
    third_party/simple-rtsp-client
    third_party/simple-rtsp-client/3rdparty
    third_party/simple-rtsp-client/rtp

    # RTMP / FLV
    third_party/libflv/include
    third_party/rtmpdump/librtmp

    # MPEG-TS
    third_party/libmpeg2core/include
    #--------------------------

    Time
)


# ==============================================================================
# Common source files
# ==============================================================================

# Test / wrapper
aux_source_directory(Test TEST)
aux_source_directory(Warpper WRAPPER)

# Hardware codec
aux_source_directory(HardCodec/Decoder HARDDEC)
aux_source_directory(HardCodec/Encoder HARDENC)
#------------------ input/output ------------------
# Media
aux_source_directory(Media/MediaCommon MEDIACOMMON)
aux_source_directory(Media/MediaMuxer MUXER)
aux_source_directory(Media/MediaReader/FileReader FILEREADER)
aux_source_directory(Media/MediaReader/RtspReader RTSPREADER)

# RTSP
aux_source_directory(third_party/simple-rtsp-client RTSPSRC)
aux_source_directory(third_party/simple-rtsp-client/3rdparty RTSP3Rd)
aux_source_directory(third_party/simple-rtsp-client/rtp RTP)

# Bitstream
aux_source_directory(third_party/Bitstream/h264/source BH264)
aux_source_directory(third_party/Bitstream/h265/src BH265)

# RTMP / FLV
aux_source_directory(Media/RtmpClient RTMPCLIENT)
aux_source_directory(third_party/libflv/source LIBFLV)
aux_source_directory(third_party/rtmpdump/librtmp LIBRTMP)

# MPEG-TS
aux_source_directory(Media/TsTransport TSTRANSPORT)
aux_source_directory(third_party/libmpeg2core/source MPEG2CORE)
#--------------------------------------------------


set(EXECUTABLE_OUTPUT_PATH ./)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_compile_options(-g -Wno-deprecated-declarations)
set(CMAKE_BUILD_TYPE Release)

# ==============================================================================
# Input / Output modules
# ==============================================================================
if(ENABLE_SRT)
    message(STATUS "ENABLE_SRT OK")
    add_definitions(-DENABLE_LIBSRT)
endif()

# ==============================================================================
# Codec modules
# ==============================================================================
if(FFMPEG_SOFT)
    message(STATUS "USE_FFMPEG_SOFT OK")
    add_definitions(-DUSE_FFMPEG_SOFT)
elseif(FFMPEG_NVIDIA)
    message(STATUS "USE_FFMPEG_NVIDIA OK")
    add_definitions(-DUSE_FFMPEG_NVIDIA)
elseif(DVPP_MPI)
    message(STATUS "USE_DVPP_MPI OK")
    add_definitions(-DUSE_DVPP_MPI -DHMEV_PLATFORM_SDK)

    include_directories(
        /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl
        /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp
        HardCodec/DVPP_utils/dvpp_enc
        HardCodec/DVPP_utils
    )
    link_directories(
        /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64
    )
    aux_source_directory(HardCodec/DVPP_utils/dvpp_enc DVPP_ENC)
    aux_source_directory(HardCodec/DVPP_utils DVPP_UTILS)
elseif(NVIDIA_SDK_X86)
    message(STATUS "USE_NVIDIA_X86 OK")
    add_definitions(-DUSE_NVIDIA_X86)
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES "AUTO")
    # export PATH=$PATH:/usr/local/cuda/bin
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    include_directories(/usr/local/cuda/include)
    include_directories(
        HardCodec/Nvcodec_utils/NvDec
        HardCodec/Nvcodec_utils/NvEnc
        HardCodec/Nvcodec_utils/Interface
        HardCodec/Nvcodec_utils/CudaProc
    )
    link_directories(/usr/local/cuda/lib64)
    aux_source_directory(HardCodec/Nvcodec_utils/NvDec NVDEC)
    aux_source_directory(HardCodec/Nvcodec_utils/NvEnc NVENC)
    # CUDA source files required by the NVIDIA codec path.
    set(CUDA_SOURCES HardCodec/Nvcodec_utils/CudaProc/ColorSpace.cu)

elseif(NVIDIA_SDK_ARM)
    message(STATUS "USE_NVIDIA_ARM OK")
    add_definitions(-DUSE_NVIDIA_ARM)
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES "AUTO")
    # export PATH=$PATH:/usr/local/cuda/bin
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    include_directories(/usr/local/cuda/include)
    include_directories(
        HardCodec/Jetson_utils
        HardCodec/Jetson_utils/include
        HardCodec/Jetson_utils/include/libjpeg-8b
        /usr/include/libdrm
        HardCodec/Jetson_utils/common/algorithm/cuda
    )
    link_directories(
        /usr/lib/aarch64-linux-gnu/tegra
        /lib/aarch64-linux-gnu
        /usr/local/lib
        /usr/lib
        /usr/local/cuda/lib64
    )
    aux_source_directory(HardCodec/Jetson_utils JETSON_UTILS)
    aux_source_directory(HardCodec/Jetson_utils/common/classes JETSON_UTILS_COMMON)
    aux_source_directory(HardCodec/Jetson_utils/common/algorithm/cuda JETSON_UTILS_COMMON_CUDA)
endif()

# ==============================================================================
# Cognition / Detection modules
# ==============================================================================
if(DETECTION_NVIDIA OR DETECTION_ASCEND OR DETECTION_HYGON)
    message(STATUS "Enable detection")
    include_directories(
        Cognition/Detection
        third_party/ByteTrack-cpp/include
        /usr/include/eigen3
    )
    aux_source_directory(Cognition/Detection DETECTION)
    aux_source_directory(third_party/ByteTrack-cpp/src BYTETRACK)

endif()

if(DETECTION_NVIDIA)
    enable_language(CUDA)
    add_definitions(-DDETECTION_NVIDIA)

    include_directories(/data/sunkx/TensorRT-10.4.0.26/include)
    link_directories(/data/sunkx/TensorRT-10.4.0.26/lib)

    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    add_compile_options(-Wno-deprecated-declarations)

    include_directories(Cognition/Detection/dataProcessKernel)

    set(DATA_PROCESS_CUDA_SOURCES Cognition/Detection/dataProcessKernel/dataProcess.cu)
elseif(DETECTION_ASCEND)

    add_definitions(-DDETECTION_ASCEND)
    add_definitions(-DENABLE_DVPP_INTERFACE)

    include_directories(
        /usr/local/Ascend/ascend-toolkit/latest/runtime/include
        /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl
        /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp
    )

    link_directories(
        /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64
    )
elseif(DETECTION_HYGON)

    enable_language(CUDA)

    add_definitions(-DDETECTION_HYGON)

    include_directories(Cognition/Detection/dataProcessKernel)

    set(DATA_PROCESS_CUDA_SOURCES Cognition/Detection/dataProcessKernel/dataProcess.cu)

    include_directories(
        /opt/dtk/cuda/cuda/include
        /opt/dtk/include
    )

    link_directories(
        /opt/dtk/cuda/cuda/lib64
        /opt/dtk/lib
    )

endif()


# ==============================================================================
# Python interface
# ==============================================================================
if(MCP_PYBIND)
    add_definitions(-DMCP_PYBIND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    set(PYBIND11_FINDPYTHON ON)

    include_directories(
        third_party/pybind11/include
        ${Python3_INCLUDE_DIRS}
    )
endif()


# ==============================================================================
# Main shared library
# ==============================================================================
add_library(
    mcp
    SHARED

    # Core / wrapper
    ${WRAPPER}

    #------ input/output ------
    # Media
    ${MEDIACOMMON}
    ${MUXER}
    ${FILEREADER}
    ${RTSPREADER}

    # RTSP
    ${RTSPSRC}
    ${RTSP3Rd}
    ${RTP}

    # Bitstream
    ${BH264}
    ${BH265}

    # RTMP / FLV
    ${RTMPCLIENT}
    ${LIBFLV}
    ${LIBRTMP}

    # MPEG-TS
    ${TSTRANSPORT}
    ${MPEG2CORE}
    #--------------------------

    # Hardware codec
    ${HARDDEC}
    ${HARDENC}
    ${DVPP_ENC}
    ${DVPP_UTILS}
    ${NVDEC}
    ${NVENC}
    ${CUDA_SOURCES}
    ${JETSON_UTILS}
    ${JETSON_UTILS_COMMON}
    ${JETSON_UTILS_COMMON_CUDA}

    # Detection
    ${DETECTION}
    ${BYTETRACK}
    ${DATA_PROCESS_CUDA_SOURCES}
)


# ==============================================================================
# Common libraries
# ==============================================================================
target_link_libraries(
    mcp
    avutil
    avformat
    avcodec
    swscale
    swresample
    ${OpenCV_LIBS}
    ssl
    crypto
)

# ==============================================================================
# Optional module libraries
# ==============================================================================

# SRT
if(ENABLE_SRT)
    target_link_libraries(mcp srt)
endif()

# Codec
if(FFMPEG_SOFT)
    # do nothing
elseif(FFMPEG_NVIDIA)
    # do nothing
elseif(DVPP_MPI)
    target_link_libraries(mcp ascendcl acl_dvpp_mpi)
elseif(NVIDIA_SDK_X86)
    target_link_libraries(
        mcp
        nvcuvid
        nvidia-encode
        cudart
        cuda
        nppidei
        nppicc
    )
elseif(NVIDIA_SDK_ARM)
    target_link_libraries(
        mcp
        cudart
        cuda
        nppidei
        nppicc
        v4l2
        EGL
        GLESv2
        X11
        nvbufsurface
        nvbufsurftransform
        nvjpeg
        nvosd
        drm
    )
endif()

# ==============================================================================
#  Detection libraries
# ==============================================================================
if(DETECTION_NVIDIA)
    target_link_libraries(
        mcp
        nvinfer
        cudart
        nppig
        nppidei
        nppial
    )
elseif(DETECTION_ASCEND)
    target_link_libraries(
        mcp
        ascendcl
        acl_dvpp_mpi
    )
elseif(DETECTION_HYGON)
    target_link_libraries(
        mcp
        cudart
        nppig
        nppidei
        nppial
        migraphx
        migraphx_gpu
        migraphx_onnx
    )
endif()

# ==============================================================================
# Platform-specific libraries
# ==============================================================================
if(WIN32)
    # Set the function stack size to 6MB.
    add_compile_options(-Wl,--stack,6000000)
    target_link_libraries(
        mcp
        ws2_32
        opencv_core470
        opencv_imgproc470
        opencv_highgui470
    )
elseif(UNIX)
    target_link_libraries(mcp pthread)
endif()


# ==============================================================================
# Final target
# ==============================================================================
if(MCP_PYBIND)

    # --------------------------------------------------------------------------
    # Python interface
    # --------------------------------------------------------------------------
    add_subdirectory(third_party/pybind11)

    pybind11_add_module(mcp_module python_module.cpp)
    target_link_libraries(mcp_module PRIVATE mcp)

else()

    # --------------------------------------------------------------------------
    # C++ interface
    # --------------------------------------------------------------------------
    add_executable(MediaCodec ${TEST})
    target_link_libraries(MediaCodec mcp)

endif()