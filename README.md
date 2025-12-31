[English](README.md) | [中文](README_CN.md)
# Media-Cognition-Pipeline
Audio/video packaging, unpackaging, encoding/decoding, visual perception (YOLO object detection + ByteTrack multi-object tracking) pipeline

* Audio/video unpackaging (MP4, RTSP), resampling, encoding/decoding, packaging (MP4), visual perception, managed using modular, node-based, and interface-based design.
* Audio codec uses a pure software solution.
* Video codec has the following implementations:
  * FFmpeg hardware encoder/decoder (FFHardDecoder.cpp, H264FFHardEncoder.cpp)
    * `cmake -DFFMPEG_NVIDIA=ON ..`
    * Only supports NVIDIA GPU, supports automatic switching between software and hardware encoding/decoding (prioritizes hardware decoding – not all NVIDIA GPUs support hardware codec; if not supported, automatically switches to software codec. FFmpeg must be compiled/installed with NVIDIA hardware codec support). Blog: https://blog.csdn.net/weixin_43147845/article/details/136812735
  * FFmpeg pure software codec (FFSoftDecoder.cpp, H264FFSoftEncoder.cpp)
    * `cmake -DFFMPEG_SOFT=ON ..`
    * Can run in any Linux/Windows environment, only requires FFmpeg installation.
  * Ascend DVPP V2 codec (DVPPDecoder.cpp, H264DVPPEncoder.cpp, DVPP_utils)
    * `cmake -DDVPP_MPI=ON ..` (first run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`)
    * Uses NPU #0 by default (MiedaWrapper.h --> device_id_)
  * NVIDIA x86 codec (NVIDIADecoder.cpp, H264NVIDIAEncoder.cpp, Nvcodec_utils)
    * `cmake -DNVIDIA_SDK_X86=ON ..` (set environment variables first: `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * Uses NVIDIA x86 native SDK (https://developer.nvidia.com/video_codec_sdk/downloads/v11), project uses Video_Codec_SDK_11.0.10, tested driver version 550.163.01. Files in Nvcodec_utils are extracted from Video_Codec_SDK_11.0.10, categorized, and only the files used in this project are included. You need to set the encoding mode (not all GPUs support hardware encoding; defaults to software encoding, MiedaWrapper.h --> use_nv_enc_flag_), default GPU #0 (MiedaWrapper.h --> device_id_). CUDA installation required (version not limited).
  * NVIDIA arm(Jetson) codec (JetsonDecoder.cpp, H264JetsonEncoder.cpp, Jetson_utils)
    * There is a problem opening the include and common(from /usr/src/jetson_multimedia_api/) folders on Windows, so they were uploaded as compressed files and need to be decompressed on Linux.
    * `cd HardCodec/Jetson_utils`
    * `tar -zxvf include.tar.gz`  
    * `tar -zxvf common.tar.gz`
    * `cmake -DNVIDIA_SDK_ARM=ON ..` (set environment variables first: `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * Jetpack version: 5.0.2, Jetpack 5. x codec is universal, but the libraries compiled from 5.0.2 cannot be directly used on other 5. x versions. Simply recompile the code on the target machine (without replacing the jetson_maultimedia.api header file)
    * Implement reference:jetson_multimedia_api/samples/02_video_dec_cuda、jetson_multimedia_api/samples/01_video_encode
    * Considering that Jetson is generally used as an edge device, in order to reduce latency, Jetson disables B-frame decoding by default. Decoding B-frames may cause frame reordering, which can result in frames being displayed out of order (frame rollback). If B-frame support is required, please modify `Jetson_utils --> JetsonDec.cpp` and comment out `ret = ctx.dec->disableDPB();` in `JetsonDec::decode_pro`.
    * Test/test2.mp4 does not contain B-frames. Test/test1.mp4 and Test/Cognition.mp4 contain B-frames (if you want to test visual perception on Jetson, please modify the code to support B-frames first, and then use the Cognition.mp4 for testing).
* Visual perception (YOLO + ByteTrack):
  * NVIDIA TensorRT
    * `-DDETECTION_NVIDIA=ON`
    * TensorRT-10.4.0.26
  * Ascend CANN
    * TODO
  * model training: https://github.com/BreakingY/yolo-onnx-tensorrt
  * The models provided in the project are converted on NVIDIA 4090 and Atlas 300V Pro(TODO).
* Supported formats: video: H264/H265, audio: AAC.
* Visual perception: YOLO11
* Ascend DVPP has two versions: V1 and V2. They support different platforms, please check the official website. Most future Ascend GPUs should support V2.
* Supports audio/video from MP4 and RTSP. MP4 unpackaging is done by FFmpeg; RTSP client is implemented in pure C++: https://github.com/BreakingY/simple-rtsp-client
* The code module division is shown in the following figure:
![MCP](https://github.com/user-attachments/assets/a33735b7-22f8-42e0-9c84-83e68d854c58)


* Managed using modular, node-based, and interface-based design, can be assembled and extended to form business pipelines.
* Logging: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec
* ByteTrack: https://github.com/Vertical-Beach/ByteTrack-cpp

# Requirements
* ffmpeg version == 4.x
* Audio uses fdk-aac codec, ensure ffmpeg includes fdk-aac.
* Tested versions: ffmpeg 4.0.5, opencv 4.5.1, CANN 7.0.0/8.2.RC1 (Ascend SDK), NVIDIA: cuda12.4; driver 550.163.01; Video_Codec_SDK 11.0.10; Jetson5.0.2; TensorRT-10.4.0.26.
* `apt install libeigen3-dev` for ByteTrack
* v4l2 for Jetson
* Windows installation guide:
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
3. Visual perception
   * NVIDIA: cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/DVPP_MPI/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_NVIDIA=ON ..
   * ASCEND: cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/DVPP_MPI/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_ASCEND=ON ..

# Test
1. File test:   `./MediaCodec ../Test/test1.mp4 out.mp4` `./MediaCodec ../Test/test2.mp4 out.mp4` `./MediaCodec ../Test/Cognition.mp4 out.mp4`
2. RTSP test:   `./MediaCodec <url> out.mp4`
3. AI inference:`./MediaCodec ../Test/Cognition.mp4 out.mp4`
4. Jetson:      `./MediaCodec ../Test/test2.mp4 out.mp4`

# Contact
* kxsun617@163.com
