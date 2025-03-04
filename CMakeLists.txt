cmake_minimum_required(VERSION 3.0)
project (VIDEOCODECPROJ)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/local/include)
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

link_directories(/usr/local/lib /usr/lib)
set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g  -fpermissive -std=c++14) 
set(CMAKE_BUILD_TYPE Release)

######ffmpeg nvidia######
# set(FFMPEG_NVIDIA FORCE)
#########################

#######ffmpeg soft#######
set(FFMPEG_SOFT FORCE)
#########################

#####Ascend DVPP V2######
# set(DVPP_MPI FORCE)
#########################

if(DEFINED  FFMPEG_NVIDIA)
    message(STATUS "USE_FFMPEG_NVIDIA OK")
    add_definitions(-DUSE_FFMPEG_NVIDIA)
elseif(DEFINED FFMPEG_SOFT)
    message(STATUS "USE_FFMPEG_SOFT OK")
    add_definitions(-DUSE_FFMPEG_SOFT)
elseif(DEFINED DVPP_MPI)
    message(STATUS "USE_DVPP_MPI OK")
    add_definitions(-DUSE_DVPP_MPI -DHMEV_PLATFORM_SDK)
    include_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp
                        HardCodec/Encoder/dvpp_enc)
    link_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub)
    aux_source_directory(HardCodec/Encoder/dvpp_enc DVPP_ENC)
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(NX SHARED ${WRAPPER} ${MEDIACOMMON} ${HARDDEC} ${HARDENC} ${MUXER} ${FILEREADER} ${RTSPREADER} ${RTSP3Rd} ${RTP} ${BH264} ${BH265} ${DVPP_ENC})
target_link_libraries(NX pthread avutil avformat avcodec swscale swresample ${OpenCV_LIBS})
if(DEFINED  DVPP_MPI)
    target_link_libraries(NX ascendcl acl_dvpp_mpi)
endif()


add_executable(MediaCodec ${TEST})
target_link_libraries(MediaCodec NX)

