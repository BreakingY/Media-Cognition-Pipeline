cmake_minimum_required(VERSION 3.10)
project (VIDEOCODECPROJ)
set(CMAKE_CXX_STANDARD 14)
option(FFMPEG_SOFT "Enable FFmpeg soft codec" OFF)
option(FFMPEG_NVIDIA "Enable FFmpeg nvidia codec" OFF)
option(DVPP_MPI "Enable Ascend DVPP V2 codec" OFF)
option(NVIDIA_SDK_X86 "Enable NVIDIA Video Codec SDK codec" OFF)
if(WIN32)
    # ffmpeg
    include_directories("D:/msys64/mingw64/include")
    include_directories("D:/msys64/home/sunkx1/ff-build/include")
    link_directories("D:/msys64/mingw64/lib")
    link_directories("D:/msys64/home/sunkx1/ff-build/lib")
    # opencv
    include_directories("D:/opencv-mingw/install/include")
    link_directories("D:/opencv-mingw/install/x64/mingw/lib")
elseif(UNIX)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib /usr/lib)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()
include_directories(Warpper Log Log/spdlog 
                    HardCodec HardCodec/Decoder HardCodec/Encoder 
                    Media/MediaCommon
                    Media/MediaMuxer
                    Media/Bitstream/h264/include Media/Bitstream/h265/include
                    Media/MediaReader Media/MediaReader/FileReader Media/MediaReader/RtspReader Media/MediaReader/RtspReader/3rdparty Media/MediaReader/RtspReader/rtp)

aux_source_directory(Test TEST)
aux_source_directory(Warpper WRAPPER)
# codec
aux_source_directory(HardCodec/Decoder HARDDEC)
aux_source_directory(HardCodec/Encoder HARDENC)
# Media
aux_source_directory(Media/MediaCommon MEDIACOMMON)
aux_source_directory(Media/MediaMuxer MUXER)
aux_source_directory(Media/MediaReader/FileReader FILEREADER)
aux_source_directory(Media/MediaReader/RtspReader RTSPREADER)
aux_source_directory(Media/MediaReader/RtspReader/3rdparty RTSP3Rd)
aux_source_directory(Media/MediaReader/RtspReader/rtp RTP)

aux_source_directory(Media/Bitstream/h264/source BH264)
aux_source_directory(Media/Bitstream/h265/source BH265)

set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g -fpermissive -std=c++14) 
set(CMAKE_BUILD_TYPE Release)

if(FFMPEG_NVIDIA)
    message(STATUS "USE_FFMPEG_NVIDIA OK")
    add_definitions(-DUSE_FFMPEG_NVIDIA)
elseif(FFMPEG_SOFT)
    message(STATUS "USE_FFMPEG_SOFT OK")
    add_definitions(-DUSE_FFMPEG_SOFT)
elseif(DVPP_MPI)
    message(STATUS "USE_DVPP_MPI OK")
    add_definitions(-DUSE_DVPP_MPI -DHMEV_PLATFORM_SDK)
    include_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp
                        HardCodec/Encoder/dvpp_enc)
    link_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub)
    aux_source_directory(HardCodec/Encoder/dvpp_enc DVPP_ENC)
elseif(NVIDIA_SDK_X86)
    message(STATUS "USE_NVIDIA_X86 OK")
    add_definitions(-DUSE_NVIDIA_X86)
    enable_language(CUDA)
    # set(CMAKE_CUDA_ARCHITECTURES "AUTO")
    # export PATH=$PATH:/usr/local/cuda/bin
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    include_directories(/usr/local/cuda/include)
    include_directories(HardCodec/Nvcodec_utils/NvDec HardCodec/Nvcodec_utils/NvEnc HardCodec/Nvcodec_utils/Interface HardCodec/Nvcodec_utils/CudaProc)
    link_directories(/usr/local/cuda/lib64)
    aux_source_directory(HardCodec/Nvcodec_utils/NvDec NVDEC)
    aux_source_directory(HardCodec/Nvcodec_utils/NvEnc NVENC)
    set(CUDA_SOURCES HardCodec/Nvcodec_utils/CudaProc/ColorSpace.cu)  # Add all your .cu files here
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(mcp SHARED ${WRAPPER} ${MEDIACOMMON} ${HARDDEC} ${HARDENC} ${MUXER} ${FILEREADER} ${RTSPREADER} ${RTSP3Rd} ${RTP} ${BH264} ${BH265} ${DVPP_ENC} ${NVDEC} ${NVENC} ${CUDA_SOURCES})
target_link_libraries(mcp avutil avformat avcodec swscale swresample ${OpenCV_LIBS})
if(DVPP_MPI)
    target_link_libraries(mcp ascendcl acl_dvpp_mpi)
endif()
if(NVIDIA_SDK_X86)
    target_link_libraries(mcp nvcuvid nvidia-encode cudart cuda nppidei)
endif()
if(WIN32)
    # Set the function stack size to 6MB
    add_compile_options(-Wl,--stack,6000000)
    target_link_libraries(mcp ws2_32 opencv_core470 opencv_imgproc470 opencv_highgui470)
elseif(UNIX)
    target_link_libraries(mcp pthread)
endif()


add_executable(MediaCodec ${TEST})
target_link_libraries(MediaCodec mcp)

