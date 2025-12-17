[中文](./README_CN.md)
# Media-Codec-Pipeline
Audio and video packaging, depackaging, and codec pipeline

* Audio and video depackaging (MP4, RTSP), resampling, encoding/decoding, packaging (MP4), using modular and interface-based management.
* Audio encoding and decoding uses a pure software solution.
* Video encoding and decoding have three implementations:
  * FFmpeg hardware codec (FFHardDecoder.cpp, H264FFHardEncoder.cpp)
    * `cmake -DFFMPEG_NVIDIA=ON ..`
    * Only supports NVIDIA GPUs, support automatic switching between software and hardware codecs (prefer hardware codec; not all NVIDIA GPUs support encoding/decoding, if not supported, automatically switch to software codec. FFmpeg must be compiled and installed with NVIDIA hardware codec support). Blog: https://blog.csdn.net/weixin_43147845/article/details/136812735
  * FFmpeg pure software codec (FFSoftDecoder.cpp, H264FFSoftEncoder.cpp)
    * `cmake -DFFMPEG_SOFT=ON ..`
    * This code can run on any Linux/Windows environment, only requires FFmpeg installation.
  * Ascend DVPP V2 codec (DVPPDecoder.cpp, H264DVPPEncoder.cpp, DVPP_utils)
    * `cmake -DDVPP_MPI=ON ..`
    * default uses NPU 0 (MiedaWrapper.h-->device_id_) 
  * NVIDIA x86 codec(NVIDIADecoder.cpp, H264NVIDIAEncoder, Nvcodec_utils)
    * `cmake -DNVIDIA_SDK_X86=ON ..` (first import environment variables `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * Using NVIDIA x86 native SDK (https://developer.nvidia.com/video_codec_sdk/downloads/v11). The project uses Video_Codec_SDK_11.0.10, tested with driver version 550.163.01. Files in Nvcodec_utils are extracted from Video_Codec_SDK_11.0.10; not all files are needed, only those used by this project are categorized. Before use, set the encoding method (not all GPUs support hardware encoding, default uses software encoding, MiedaWrapper.h-->use_nv_enc_flag_), default uses GPU 0 (MiedaWrapper.h-->device_id_), requires CUDA installation (version not limited)
* Users can add support for any GPU by defining macros, as long as class names and methods are consistent, offering good platform scalability.
* Supported formats: Video: H264/H265, Audio: AAC.
* Not suitable for Jetson; Jetson's codec library differs from x86. Jetson encoding/decoding reference: https://github.com/BreakingY/jetpack-dec-enc
* Ascend DVPP has two versions: V1 and V2. V1 and V2 are for different platforms; please check the official site. Future Ascend GPUs should all support V2. 
* Supports fetching audio and video from MP4 and RTSP. MP4 depackaging is done by FFmpeg; RTSP client is implemented in pure C++, https://github.com/BreakingY/simple-rtsp-client
* The code includes four modules, as shown below:

![1](https://github.com/user-attachments/assets/2dee0b7c-46c1-4161-9de9-3b0c7f270fc7)
* Wrapper combines the four modules, as shown below:
![2](https://github.com/user-attachments/assets/39082b4c-cba7-421d-b47e-e319c0d6a10b)
* Uses modular and interface-based management, allowing users to assemble and extend pipelines, e.g., adding video/audio processing modules to handle decoded media, such as AI detection, speech recognition, etc.
* Log: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec

# Preparation
* FFmpeg version == 4.x, please modify CMakeLists.txt according to the installation path, adding headers and library paths.
* Audio uses fdk-aac encoding, ensure FFmpeg includes fdk-aac.
* Test versions: FFmpeg 4.0.5, OpenCV 4.5.1, CANN 7.0.0/8.2.RC1 (Ascend SDK), NVIDIA: CUDA 12.4, Driver 550.163.01, Video_Codec_SDK 11.0.10.
* Windows software installation reference:
  * https://sunkx.blog.csdn.net/article/details/146064215

# Build
* git clone --recursive https://github.com/BreakingY/Media-Codec-Pipeline.git
1. Linux
   * mkdir build
   * cd build
   * cmake -DFFMPEG_SOFT=ON ..
   * make -j
2. Windows (MinGW + cmake)
   * mkdir build
   * cd build
   * cmake -G "MinGW Makefiles" -DFFMPEG_SOFT=ON ..
   * mingw32-make -j

# Test:
1. File test: `./MediaCodec ../Test/test1.mp4 out.mp4 && ./MediaCodec ../Test/test2.mp4 out.mp4`
2. RTSP test: `./MediaCodec your_rtsp_url out.mp4`
3. Ascend test: `./MediaCodec ../Test/dvpp_venc.mp4 out.mp4`


# Technical Contact
* kxsun617@163.com
