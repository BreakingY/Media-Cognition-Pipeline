cmake_minimum_required(VERSION 3.0)
project (VIDEOCODECPROJ)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/local/include)
include_directories(Warpper Log Log/spdlog 
                    FFmpegHardCodec FFmpegHardCodec/Decoder FFmpegHardCodec/Encoder 
                    Media/MediaCommon
                    Media/MediaMuxer
                    Media/Bitstream/h264/include Media/Bitstream/h265/include
                    Media/MediaReader Media/MediaReader/FileReader Media/MediaReader/RtspReader Media/MediaReader/RtspReader/3rdparty Media/MediaReader/RtspReader/rtp)

aux_source_directory(Test TEST)
aux_source_directory(Warpper WRAPPER)
# codec
aux_source_directory(FFmpegHardCodec/Decoder HARDDEC)
aux_source_directory(FFmpegHardCodec/Encoder HARDENC)
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

########################################ffmpeg nvidia###################################################
# add_definitions(-DUSE_FFMPEG_NVIDIA)
########################################################################################################

########################################ffmpeg soft#####################################################
add_definitions(-DUSE_FFMPEG_SOFT)
########################################################################################################

########################################Ascend DVPP V2##################################################
# add_definitions(-DUSE_DVPP_MPI -DHMEV_PLATFORM_SDK)
# include_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp
#                     FFmpegHardCodec/Encoder/dvpp_enc)
# link_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub)
# aux_source_directory(FFmpegHardCodec/Encoder/dvpp_enc DVPP_ENC)
########################################################################################################


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(NX SHARED ${WRAPPER} ${MEDIACOMMON} ${HARDDEC} ${HARDENC} ${MUXER} ${FILEREADER} ${RTSPREADER} ${RTSP3Rd} ${RTP} ${BH264} ${BH265} ${DVPP_ENC})
target_link_libraries(NX pthread avutil avformat avcodec swscale swresample ${OpenCV_LIBS})
########################################Ascend DVPP V2##################################################
# target_link_libraries(NX ascendcl acl_dvpp_mpi)
########################################################################################################

add_executable(MediaCodec ${TEST})
target_link_libraries(MediaCodec NX)

