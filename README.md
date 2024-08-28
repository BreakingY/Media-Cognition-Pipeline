# FFmpeg-Media-Codec-Pipeline
ffmpeg实现音视频封装、编解码pipeline

* 音视频解封装(MP4、RTSP)、重采样、编解码、封装(MP4)，采用模块化和接口化管理
* 音频编解码使用纯软方案，视频编解码有两种实现
  1. 实现了视频的硬编解码(HardDecoder.cpp、H264HardEncoder.cpp)，仅支持英伟达显卡。支持软硬编解码自动切换(优先使用硬编解码-不是所有nvidia显卡都支持编解码、不支持则自动切换到软编解码，ffmpeg需要在编译安装的时候添加Nvidia硬编解码功能),需要在CMakeLists.txt中打开add_definitions(-DUSE_FFMPEG_NVIDIA)。 博客地址：https://blog.csdn.net/weixin_43147845/article/details/136812735
  2. 使用ffmpeg纯软件编解码(SoftDecoder.cpp、H264SoftEncoder.cpp)，使用ffmpeg的nvidia硬编解码功能必须在ffmpeg编译安装的时候添加nvidia选项，如果不想使用ffmpeg-nvidia，需要在CMakeLists.txt中把add_definitions(-DUSE_FFMPEG_NVIDIA)注释掉，此时就不需要ffmpeg编译安装的时候添加nvidia选项，而是使用ffmpeg的纯软件编解码，这么做是考虑了实际环境没有显卡或者不是nvidia显卡，此时代码可以在任何Linux环境下运行。
  3. add_definitions(-DUSE_FFMPEG_NVIDIA)默认是关闭状态
* 支持格式，视频：H264/H265，音频：AAC
* 不适用jetson。
* 支持从MP4、RTSP获取音视频。MP4解封装由FFMPEG完成；RTSP客户端纯C++实现，不依赖任何库，地址：https://github.com/BreakingY/simple-rtsp-client
* 代码包含四个模块，如下图所示：

  ![未命名绘图](https://github.com/BreakingY/FFmpeg-Media-Codec-Pipeline/assets/99859929/fbde5819-4527-4eec-8b7b-508264efc995)
* Warpper实现了对四个模块的组合，如下图所示：
  ![媒体流程](https://github.com/BreakingY/FFmpeg-Media-Codec-Pipeline/assets/99859929/f7fb8e07-ab2a-49c5-88e1-49301b6431bd)
* 采用模块化和接口化的管理方式，可自行组装扩展形成业务pipeline，比如添加视频处理模块、音频处理模块，对解码后的音视频进行处理，例如，AI检测、语音识别等。
* 日志使用的spdlog，地址：https://github.com/gabime/spdlog
* Bitstream：https://github.com/ireader/avcodec

# 准备
* 安装ffmpeg到/usr/local下，版本>=4.x 如果没有安装在/usr/local下面请修改CMakeLists.txt，把头文件和库路径添加进去
* 安装opencv，需要cmake能找到，如果不能请修改CMakeLists.txt，手动指定opencv头文件、库路径及链接的库
* 测试版本 ffmpeg4.0.5、opencv4.5.1

# Linux编译
* mkdir build
* cmake ..
* make -j
* 测试：
  1. 文件测试：./MediaCodec ../Test/test1.mp4 out.mp4 && ./MediaCodec ../Test/test2.mp4 out.mp4
  2. rtsp测试：./MediaCodec your_rtsp_url out.mp4


# 技术交流
* kxsun617@163.com


