# FFmpeg-Media-Codec-Pipeline
ffmpeg实现音视频封装、编解码pipeline

* 用ffmpeg实现对音视频解封装、重采样、编解码、封装(MP4)，并采用模块化和接口化管理
* 实现了视频的硬编解码，仅支持英伟达显卡。支持软硬编解码自动切换(优先使用硬编解码、不支持则自动切换到软编解码，ffmpeg需要在编译的时候添加Nvidia硬编解码功能)
* 支持格式，视频：H264/H265，音频：AAC
* 不适用jetson。
* 代码包含四个模块，如下图所示：

  ![未命名绘图](https://github.com/BreakingY/FFmpeg-Media-Codec-Pipeline/assets/99859929/fbde5819-4527-4eec-8b7b-508264efc995)
* Warpper实现了对四个模块的组合，如下图所示：
  ![媒体流程](https://github.com/BreakingY/FFmpeg-Media-Codec-Pipeline/assets/99859929/f7fb8e07-ab2a-49c5-88e1-49301b6431bd)
* 采用模块化和接口化的管理方式，可自行组装扩展形成业务pipeline，比如把解封装模块换成RTSP客户端模块，就可以实现从rtsp拉取实时音视频流；或者添加视频处理模块、音频处理模块，对解码后的音视频进行处理，例如，AI检测、语音识别等。
* 日志使用的spdlog，地址：https://github.com/gabime/spdlog

# Linux编译
* ffmpeg版本：>=4.x 如果ffmpeg没有安装在/usr/local下面请修改CMakeLists.txt，把头文件和库路径添加进去
* mkdir build
* cmake ..
* make -j
* 测试：./MediaCodec ../Test/test1.mp4 out.mp4 && ./MediaCodec ../Test/test2.mp4 out.mp4

# TODO
* 同步优化

# 技术交流
* kxsun617@163.com


