cmake_minimum_required(VERSION 3.0)
project (VIDEOCODECPROJ)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/local/include)
include_directories(Warpper MediaCommon Log Log/spdlog FFmpegHardCodec FFmpegHardCodec/Decoder FFmpegHardCodec/Encoder MediaMuxer 
                    MediaReader Bitstream/h264/include Bitstream/h265/include
                    MediaReader/FileReader MediaReader/RtspReader MediaReader/RtspReader/3rdparty MediaReader/RtspReader/rtp)

aux_source_directory(Test TEST)
aux_source_directory(Warpper WRAPPER)
aux_source_directory(MediaCommon MEDIACOMMON)
aux_source_directory(FFmpegHardCodec/Decoder HARDDEC)
aux_source_directory(FFmpegHardCodec/Encoder HARDENC)
aux_source_directory(MediaMuxer MUXER)
aux_source_directory(MediaReader/FileReader FILEREADER)
aux_source_directory(MediaReader/RtspReader RTSPREADER)
aux_source_directory(MediaReader/RtspReader/3rdparty RTSP3Rd)
aux_source_directory(MediaReader/RtspReader/rtp RTP)

aux_source_directory(Bitstream/h264/source BH264)
aux_source_directory(Bitstream/h265/source BH265)

link_directories(/usr/local/lib /usr/lib)
set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g  -fpermissive -std=c++14) 
set(CMAKE_BUILD_TYPE Release)

#add_definitions(-DUSE_FFMPEG_NVIDIA) # 使用ffmpeg-nvidia代码(ffmpeg编译安装的时候增加nvidia选项)
add_definitions(-DUSE_FFMPEG_SOFT)  # 使用ffmpeg纯软件编解码代码(ffmpeg编译安装的时候不需要指定额外参数)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(NX SHARED ${WRAPPER} ${MEDIACOMMON} ${HARDDEC} ${HARDENC} ${COMMON} ${MUXER} ${FILEREADER} ${RTSPREADER} ${RTSP3Rd} ${RTP} ${BH264} ${BH265})
target_link_libraries(NX pthread avutil avformat avcodec swscale swresample ${OpenCV_LIBS})

add_executable(MediaCodec ${TEST})
target_link_libraries(MediaCodec NX)

