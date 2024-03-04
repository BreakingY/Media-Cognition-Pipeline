cmake_minimum_required(VERSION 3.0)
project (VIDEOCODECPROJ)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 17)  # 设置 C++ 标准为 C++17
include_directories(/usr/local/include)
include_directories(Warpper MediaCommon Log Log/spdlog FFmpegHardCodec FFmpegHardCodec/Decoder FFmpegHardCodec/Encoder MediaMuxer MediaReader)

aux_source_directory(Test TEST)
aux_source_directory(Warpper WRAPPER)
aux_source_directory(MediaCommon MEDIACOMMON)
aux_source_directory(FFmpegHardCodec/Decoder HARDDEC)
aux_source_directory(FFmpegHardCodec/Encoder HARDENC)
aux_source_directory(MediaMuxer MUXER)
aux_source_directory(MediaReader READER)

link_directories(/usr/local/lib /usr/lib)
set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g  -fpermissive -std=c++17) 
set(CMAKE_BUILD_TYPE Release)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
add_library(NX SHARED ${WRAPPER} ${MEDIACOMMON} ${HARDDEC} ${HARDENC} ${COMMON} ${MUXER} ${READER})
target_link_libraries(NX pthread lzma avutil avformat avcodec swscale swresample ${OpenCV_LIBS})

add_executable(MediaCodec ${TEST})
target_link_libraries(MediaCodec NX)

