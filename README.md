[English](README_EN.md) | [中文](README.md)
# Media-Cognition-Pipeline
实时流媒体及深度学习推理加速通用处理框架，支持H264、H265、AAC、MP4、FLV、TS、RTSP、RTMP、SRT、YOLO。
* 音视频解封装(MP4、FLV、TS、RTSP、RTMP、SRT(TODO))、重采样、编解码(H264、H265、AAC；NVIDIA、晟腾)、封装(MP4、FLV、TS、RTMP、SRT)，视觉感知(YOLO目标检测 + ByteTrack多目标跟踪；NVIDIA、晟腾、海光)pipeline, 采用模块化、节点化和接口化管理。

# 解封装
* mp4
  * Media/FileReader
  * 使用ffmpeg实现
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)
* ts/srt
  * Media/TsTransport
  * libmpeg2 (https://github.com/BreakingY/libmpeg2core)
  * TS：默认支持；SRT：需要添加cmake选项，`-DENABLE_SRT=ON`，并且需要安装libsrt。
  * libsrt安装
    * git clone https://github.com/Haivision/srt.git
    * cd srt && ./configure
    * make && make install
* rtsp
  * Media/RtspReader
  * simple-rtsp-client (https://github.com/BreakingY/simple-rtsp-client)

# 编解码(只能开启其中的一种)
* 音频编解码使用纯软方案。
* 视频编解码有以下实现：
  * FFmpeg硬编解码(FFHardDecoder.cpp、H264FFHardEncoder.cpp)
    * `cmake -DFFMPEG_NVIDIA=ON ..`
    * 仅支持英伟达显卡，支持软硬编解码自动切换(优先使用硬编解码-不是所有nvidia显卡都支持编解码、不支持则自动切换到软编解码，ffmpeg需要在编译安装的时候添加Nvidia硬编解码功能)。 博客地址：https://blog.csdn.net/weixin_43147845/article/details/136812735
  * FFmpeg纯软编解码(FFSoftDecoder.cpp、H264FFSoftEncoder.cpp)
    * `cmake -DFFMPEG_SOFT=ON ..`
    * 此时代码可以在任何Linux/Windows环境下运行,只需要安装ffmpeg即可
  * 昇腾DVPP V2版本编解码(DVPPDecoder.cpp、H264DVPPEncoder.cpp、DVPP_utils)
    * `cmake -DDVPP_MPI=ON ..`(先执行`source /usr/local/Ascend/ascend-toolkit/set_env.sh`)
    * 默认使用第0号NPU(MiedaWrapper.h-->device_id_)
    * 考虑是实时性，不支持解码B帧，如果要支持B帧，请修改`DVPPDecoder.cpp-->HardVideoDecoder::Init`，把`chn_attr_.video_attr.ref_frame_num`增大即可
  * NVIDIA x86编解码(NVIDIADecoder.cpp、H264NVIDIAEncoder.cpp、Nvcodec_utils)
    * `cmake -DNVIDIA_SDK_X86=ON ..`(先导入环境变量`export PATH=$PATH:/usr/local/cuda/bin`和`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * 使用NVIDIA x86原生SDK(https://developer.nvidia.com/video_codec_sdk/downloads/v11), 项目使用Video_Codec_SDK_11.0.10版本，测试驱动版本为550.163.01, Nvcodec_utils目录里的文件都是从Video_Codec_SDK_11.0.10中提取的，因为Video_Codec_SDK_11.0.10中文件很多，实际使用过程中并不是所有的都需要，Nvcodec_utils里面只提取出来本项目使用的文件，并进行分类。使用前需要设置编码方式(不是所有的显卡都支持硬编码,默认使用软编码, MiedaWrapper.h-->use_nv_enc_flag_)，默认使用第0号GPU(MiedaWrapper.h-->device_id_), 需要安装cuda(版本无要求)
  * NVIDIA arm(Jetson)编解码(JetsonDecoder.cpp、H264JetsonEncoder.cpp、Jetson_utils)
    * include和common(来自/usr/src/jetson_multimedia_api/)两个文件夹在windows上打开有问题，所以以压缩包的形式传上来的，需要在linux下解压。
    * `cd HardCodec/Jetson_utils`
    * `tar -zxvf include.tar.gz`  
    * `tar -zxvf common.tar.gz`
    * `cmake -DNVIDIA_SDK_ARM=ON ..`(先导入环境变量`export PATH=$PATH:/usr/local/cuda/bin`和`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * Jetpack版本： 5.0.2，Jetpack 5.x编解码是通用的，但是5.0.2编译出来的库不能直接在其他5.x版本上使用，把代码放到目标机器上重新编译即可(不需要替换jetson_multimedia_api头文件)
    * 实现参考：jetson_multimedia_api/samples/02_video_dec_cuda、jetson_multimedia_api/samples/01_video_encode
    * 考虑到Jetson一般作为边缘设备，为了降低时延，Jetson默认不支持解码B帧，解码B帧会出现画面倒退的情况，如果要支持B帧请修改`Jetson_utils-->JetsonDec.cpp`，注释掉`JetsonDec::decode_pro`中的`ret = ctx.dec->disableDPB();`

# 封装
* mp4
  * Media/MediaMuxer
  * 使用ffmpeg实现
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)
* ts/srt
  * Media/TsTransport
  * libmpeg2 (https://github.com/BreakingY/libmpeg2core)
  * TS：默认支持；SRT：需要添加cmake选项，`-DENABLE_SRT=ON`
  * 需要安装libsrt(https://github.com/Haivision/srt)
    * git clone https://github.com/Haivision/srt.git
    * cd srt && ./configure
    * make && make install

# 视觉感知(YOLO + ByteTrack)
* NVIDIA TensorRT
  * `-DDETECTION_NVIDIA=ON`
  * TensorRT-10.4.0.26
  * `trtexec --onnx=yolo11s_best.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=yolo11s_best.engine --fp16`
* Ascend CANN
  * `-DDETECTION_ASCEND=ON`
  * CANN7.0.0/8.2.RC1
  * `atc --model=yolo11s_best.onnx --framework=5 --input_shape=images:-1,3,640,640 --dynamic_batch_size="1,2,3,4" --insert_op_conf=insert_op.cfg --output=yolo11s_best --soc_version=Ascend310P3  --precision_mode_v2=mixed_float16`
* Hygon MIGraphX
  * `-DDETECTION_HYGON=ON`
  * `source /etc/profile.d/dtk.sh`
  * `source /etc/profile.d/migraphx.sh`
  * `source /opt/dtk/cuda/env.sh`
  * dtk26.04/migraphx5.2.0
  * 动态batch(migraphx暂不支持yolo11)
    * `migraphx-driver compile --onnx yolo11s_best.onnx --dyn-input-dim @images "[{min:1,max:8,optimals:[1,4,8]},3,640,640]" --binary -o yolo11s_best.mxr --fp16 --gpu`
  * 静态batch
    * `migraphx-driver compile --onnx yolo11s_best.onnx --input-dim @images 4 3 640 640 --binary -o yolo11s_best.mxr --fp16 --gpu`
* yolo11s_best.onnx 包含两个类别：{"dog", "person"}
* 模型训练：https://github.com/BreakingY/yolo-onnx-tensorrt

# 框架搭建
* Warpper
* 基于解封装、编解码、封装、视觉感知模块搭建通用媒体处理及感知框架。
* 因为mp4需要在程序结束的时候写入尾数据，对于rtsp/rtmp实时流不适用，实时流写文件推荐使用flv和ts。

# 说明
* 支持格式，视频：H264/H265，音频：AAC。
* 视觉感知：YOLO11
* 测试版本：
  * ffmpeg 4.0.5(ffmpeg==4.x，音频使用fdk-aac编码，确保安装的ffmpeg包含fdk-aac)
  * opencv 4.5.1
  * Ascend: CANN 7.0.0/8.2.RC1
  * NVIDIA: 
    * TensorRT: 10.4.0.26
    * X86: cuda 12.4; 驱动 550.163.01; Video_Codec_SDK 11.0.10
    * ARM: Jetson 5.0.2
  * Hygon DCU: DTK 26.04; migraphx 5.2.0
* ByteTrack依赖: `apt install libeigen3-dev`
* Jetson依赖：v4l2
* librtmp依赖：openssl
* Windows 软件安装参考：
  * https://sunkx.blog.csdn.net/article/details/146064215
* 代码模块划分如下图：
![MCP](https://github.com/user-attachments/assets/6a9cbe6a-8a61-47de-bdac-81dc5865e182)

# 感谢以下作者(.gitmodules子模块)
* spdlog: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec
* ByteTrack: https://github.com/Vertical-Beach/ByteTrack-cpp
* librtmp: https://git.ffmpeg.org/rtmpdump
* pybind11: https://github.com/pybind/pybind11 v2.12.0
* libflv: https://github.com/BreakingY/libflv
* simple-rtsp-client: https://github.com/BreakingY/simple-rtsp-client
* libmpeg2core: https://github.com/BreakingY/libmpeg2core

# 编译
* `git clone --recursive https://github.com/BreakingY/Media-Cognition-Pipeline.git`
1. Linux
   * `mkdir build`
   * `cd build`
   * `cmake -DFFMPEG_SOFT=ON ..`
   * `make -j`
2. Windows(MinGW + cmake)
   * `mkdir build`
   * `cd build`
   * `cmake -G "MinGW Makefiles" -DFFMPEG_SOFT=ON ..`
   * `mingw32-make -j`
3. 视觉感知
   * NVIDIA: `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_NVIDIA=ON ..`
   * ASCEND: `cmake -D<FFMPEG_SOFT/DVPP_MPI>=ON -DDETECTION_ASCEND=ON ..`
   * Hygon : `cmake -DFFMPEG_SOFT=ON -DDETECTION_HYGON=ON ..`
# 测试：
1. pipeline测试：
   `./MediaCodec <mp4(../Test/test*.mp4)>/<flv(../Media/RtmpClient/libflv/test/test_1280x720_h264_aac.flv)/ts(../Media/TsTransport/libmpeg2core/media/h264_aac.ts)>/<srt url>/<rtsp url>/<rtmp url> <mp4>/<flv>/<ts>/<srt url>/<rtmp url>`
2. AI推理：
   `./MediaCodec ../Test/Cognition.mp4 <mp4>/<flv>/<ts>/<srt url>/<rtmp url>`

   https://github.com/user-attachments/assets/59b77cfd-b6a7-4fcd-b6ab-acec722d74e2

# python接口
* 实时流媒体加速处理库(输入：文件/RTSP/RTMP/SRT;输出：文件/RTMP/SRT)
* 流媒体处理库：为Python提供高性能的视频处理功能，包括音视频采集、编解码加速、帧数据交互以及实时流媒体处理。
* 典型应用场景：C++负责视频采集(文件，实时流)、编解码加速、输出(文件，实时流)；Python负责目标检测、人脸识别、行为分析等模型算法。
* `-DMCP_PYBIND=ON`此时仅作为媒体库供python使用，无法开启`DETECTION_NVIDIA/DETECTION_ASCEND/DETECTION_HYGON`
* 如果要指定特定python版本，请参考下面命令,例如编译python3.12和python3.7
  * `-DPYTHON_LIBRARY=/usr/local/lib/libpython3.12.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.12 -DPYTHON_EXECUTABLE=/usr/local/bin/python3.12`
  * `-DPYTHON_LIBRARY=/usr/local/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.7m -DPYTHON_EXECUTABLE=/usr/local/bin/python3.7`
* 编译：
  1. 源码编译python`./configure --enable-shared`必须添加--enable-shared参数，否则无法编译
  2. cd build
  3. `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/NVIDIA_SDK_X86/NVIDIA_SDK_ARM/DVPP_MPI>=ON <-DENABLE_SRT=ON> -DMCP_PYBIND=ON ..`
  4. make -j
  5. python demo.py

# 技术交流
* kxsun617@163.com
