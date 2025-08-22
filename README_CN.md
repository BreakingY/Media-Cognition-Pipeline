# Media-Codec-Pipeline
音视频封装、解封装、编解码pipeline

* 音视频解封装(MP4、RTSP)、重采样、编解码、封装(MP4)，采用模块化和接口化管理。
* 音频编解码使用纯软方案。
* 视频编解码有三种实现：
  1. FFmpeg硬编解码(HardDecoder.cpp、H264HardEncoder.cpp)，仅支持英伟达显卡，cmake -DFFMPEG_NVIDIA=ON .. 支持软硬编解码自动切换(优先使用硬编解码-不是所有nvidia显卡都支持编解码、不支持则自动切换到软编解码，ffmpeg需要在编译安装的时候添加Nvidia硬编解码功能)。 博客地址：https://blog.csdn.net/weixin_43147845/article/details/136812735
  2. FFmpeg纯软编解码(SoftDecoder.cpp、H264SoftEncoder.cpp)，cmake -DFFMPEG_SOFT=ON .. 此时代码可以在任何Linux/Windows环境下运行,只需要安装ffmpeg即可
  3. 昇腾显卡DVPP V2版本编解码(DVPPDecoder.cpp、H264DVPPEncoder.cpp、dvpp_enc)，默认使用第0号NPU(MiedaWrapper.h), cmake -DDVPP_MPI=ON ..
  4. Video_Codec_SDK，使用NVIDIA x86原生SDK(https://developer.nvidia.com/video_codec_sdk/downloads/v11), 项目使用Video_Codec_SDK_11.0.10版本，测试驱动版本为550.163.01, Nvcodec_utils目录里的文件都是从Video_Codec_SDK_11.0.10中提取的，因为Video_Codec_SDK_11.0.10中文件很多，实际使用过程中并不是所有的都需要，Nvcodec_utils里面只提取出来本项目使用的文件，并进行分类。使用前需要设置编码方式(不是所有的显卡都支持硬编码,默认使用软编码)，默认使用第0号GPU(MiedaWrapper.h), 需要安装cuda(版本无要求), cmake -DNVIDIA_SDK_X86=ON ..(先导入环境变量export PATH=$PATH:/usr/local/cuda/bin和export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64)
* 通过设置宏的方式，使用者可以添加适配任意显卡的代码，只要保证类名和被调用的类方法一致即可，平台扩展性好。
* 支持格式，视频：H264/H265，音频：AAC。
* ffmpeg-nvidia不适用jetson，jetson的编解码库和x86不一样。jetson编解码参考：https://github.com/BreakingY/jetpack-dec-enc
* 昇腾的DVPP有两个版本:V1和V2 ,V1和V2适用不同的平台，请到官网自行查阅，不过昇腾后续的显卡应该都支持V2版本。DVPP视频输入的宽必须是16的整数倍，高必须是2的整数倍，并且DVPP不是所有格式的视频都支持。
* 支持从MP4、RTSP获取音视频。MP4解封装由FFMPEG完成；RTSP客户端纯C++实现，不依赖任何库，地址：https://github.com/BreakingY/simple-rtsp-client
* 代码包含四个模块，如下图所示：

![1](https://github.com/user-attachments/assets/2dee0b7c-46c1-4161-9de9-3b0c7f270fc7)
* Warpper实现了对四个模块的组合，如下图所示：
![2](https://github.com/user-attachments/assets/39082b4c-cba7-421d-b47e-e319c0d6a10b)
* 采用模块化和接口化的管理方式，可自行组装扩展形成业务pipeline，比如添加视频处理模块、音频处理模块，对解码后的音视频进行处理，例如，AI检测、语音识别等。
* 日志，地址：https://github.com/gabime/spdlog
* Bitstream：https://github.com/ireader/avcodec

# 准备
* ffmpeg版本==4.x 请根据安装位置修改CMakeLists.txt，把头文件和库路径添加进去。
* 音频使用fdk-aac编码，确保安装的ffmpeg包含fdk-aac。
* 测试版本 ffmpeg4.0.5、opencv4.5.1、CANN7.0.0(昇腾SDK)、NVIDIA:cuda12.4; 驱动550.163.01; Video_Codec_SDK11.0.10。
* Windows 软件安装参考：
  * https://sunkx.blog.csdn.net/article/details/146064215

# 编译
* git clone --recursive https://github.com/BreakingY/Media-Codec-Pipeline.git
1. Linux
   * mkdir build
   * cd build
   * cmake -DFFMPEG_SOFT=ON ..
   * make -j
2. Windows(MinGW + cmake)
   * mkdir build
   * cd build
   * cmake -G "MinGW Makefiles" -DFFMPEG_SOFT=ON ..
   * mingw32-make -j
# 测试：
1. 文件测试：./MediaCodec ../Test/test1.mp4 out.mp4 && ./MediaCodec ../Test/test2.mp4 out.mp4
2. rtsp测试：./MediaCodec your_rtsp_url out.mp4
3. 昇腾测试：./MediaCodec ../Test/dvpp_venc.mp4 out.mp4

# TODO
* 解除DVPP视频宽高的限制
* NVIDIA使用cuda流优化

# 技术交流
* kxsun617@163.com


