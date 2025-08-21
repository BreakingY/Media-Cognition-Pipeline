[中文](./README_CN.md)
# Media-Codec-Pipeline
Audio and video packaging, depackaging, and codec pipeline

* Audio and video depackaging (MP4, RTSP), resampling, encoding/decoding, packaging (MP4), using modular and interface-based management.
* Audio encoding and decoding uses a pure software solution.
* Video encoding and decoding have three implementations:
  1. FFmpeg hardware codec (HardDecoder.cpp, H264HardEncoder.cpp), only supports NVIDIA GPUs. Use `cmake -DFFMPEG_NVIDIA=ON ..` to support automatic switching between software and hardware codecs (prefer hardware codec; not all NVIDIA GPUs support encoding/decoding, if not supported, automatically switch to software codec. FFmpeg must be compiled and installed with NVIDIA hardware codec support). Blog: https://blog.csdn.net/weixin_43147845/article/details/136812735
  2. FFmpeg pure software codec (SoftDecoder.cpp, H264SoftEncoder.cpp), `cmake -DFFMPEG_SOFT=ON ..`. This code can run on any Linux/Windows environment, only requires FFmpeg installation.
  3. Ascend GPU DVPP V2 codec (DVPPDecoder.cpp, H264DVPPEncoder.cpp, dvpp_enc), default uses NPU 0 (MiedaWrapper.h), `cmake -DDVPP_MPI=ON ..`.
  4. Video_Codec_SDK, using NVIDIA x86 native SDK (https://developer.nvidia.com/video_codec_sdk/downloads/v11). The project uses Video_Codec_SDK_11.0.10, tested with driver version 550.163.01. Files in Nvcodec_utils are extracted from Video_Codec_SDK_11.0.10; not all files are needed, only those used by this project are categorized. Before use, set the encoding method (not all GPUs support hardware encoding, default uses software encoding), default uses GPU 0 (MiedaWrapper.h), requires CUDA installation (version not limited), `cmake -DNVIDIA_SDK_X86=ON ..` (first import environment variables `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`).
* Users can add support for any GPU by defining macros, as long as class names and methods are consistent, offering good platform scalability.
* Supported formats: Video: H264/H265, Audio: AAC.
* ffmpeg-nvidia is not suitable for Jetson; Jetson's codec library differs from x86. Jetson encoding/decoding reference: https://github.com/BreakingY/jetpack-dec-enc
* Ascend DVPP has two versions: V1 and V2. V1 and V2 are for different platforms; please check the official site. Future Ascend GPUs should all support V2. DVPP video input width must be a multiple of 16, height a multiple of 2, and not all video formats are supported.
* Supports fetching audio and video from MP4 and RTSP. MP4 depackaging is done by FFmpeg; RTSP client is implemented in pure C++ without any dependencies, https://github.com/BreakingY/simple-rtsp-client
* The code includes four modules, as shown below:

![1_half](https://github.com/user-attachments/assets/278d1b1b-cfae-406c-9b87-3e2b318e83c1)
* Wrapper combines the four modules, as shown below:
![2_half](https://github.com/user-attachments/assets/b0cf8428-0f8d-4904-8faa-c3980a7da2ce)
* Uses modular and interface-based management, allowing users to assemble and extend pipelines, e.g., adding video/audio processing modules to handle decoded media, such as AI detection, speech recognition, etc.
* Logging uses spdlog: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec

# Preparation
* FFmpeg version == 4.x, please modify CMakeLists.txt according to the installation path, adding headers and library paths.
* Audio uses fdk-aac encoding, ensure FFmpeg includes fdk-aac.
* Test versions: FFmpeg 4.0.5, OpenCV 4.5.1, CANN 7.0.0 (Ascend SDK), NVIDIA: CUDA 12.4, Driver 550.163.01, Video_Codec_SDK 11.0.10.
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

# TODO
* Remove DVPP video width/height limitations
* Optimize NVIDIA using CUDA streams

# Technical Contact
* kxsun617@163.com
