[English](README.md) | [中文](README_CN.md)
# Media-Cognition-Pipeline
A general-purpose streaming media and deep learning inference acceleration framework, supporting H264, H265, AAC, MP4, FLV, RTSP, RTMP, and YOLO.
* Audio/video demuxing (MP4, RTSP, FLV, RTMP (TODO)), resampling, encoding/decoding (H264, H265, AAC; NVIDIA, Ascend), muxing (MP4, FLV, RTMP), and visual perception (YOLO object detection + ByteTrack multi-object tracking; NVIDIA, Ascend) pipeline, managed with a modular, node-based, and interface-oriented design.

# Demuxing
* mp4
  * Media/FileReader
  * Implemented using FFmpeg
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)
* rtsp
  * Media/RtspReader
  * simple-rtsp-client (https://github.com/BreakingY/simple-rtsp-client)

# Encoding / Decoding
* Audio encoding/decoding uses a pure software solution.
* Video encoding/decoding implementations include:
  * FFmpeg hardware-accelerated encoding/decoding (FFHardDecoder.cpp, H264FFHardEncoder.cpp)
    * `cmake -DFFMPEG_NVIDIA=ON ..`
    * NVIDIA GPU only, supports automatic switching between software and hardware encoding/decoding (hardware is preferred — not all NVIDIA GPUs support hardware codecs; if unsupported, it automatically falls back to software. FFmpeg must be compiled with NVIDIA hardware codec support enabled).  
      Blog: https://blog.csdn.net/weixin_43147845/article/details/136812735
  * FFmpeg pure software encoding/decoding (FFSoftDecoder.cpp, H264FFSoftEncoder.cpp)
    * `cmake -DFFMPEG_SOFT=ON ..`
    * Can run on any Linux/Windows environment, only requires FFmpeg to be installed.
  * Ascend DVPP V2 encoding/decoding (DVPPDecoder.cpp, H264DVPPEncoder.cpp, DVPP_utils)
    * `cmake -DDVPP_MPI=ON ..` (execute `source /usr/local/Ascend/ascend-toolkit/set_env.sh` first)
    * Uses NPU device 0 by default (MiedaWrapper.h → device_id_)
    * For real-time performance, B-frame decoding is not supported by default. To enable B-frame support, modify `DVPPDecoder.cpp → HardVideoDecoder::Init` and increase `chn_attr_.video_attr.ref_frame_num`.
    * Test/test2.mp4 contains no B-frames; Test/test1.mp4 and Test/Cognition.mp4 contain B-frames (if you want to test visual perception on Ascend, first modify the code to support B-frames, then use Cognition.mp4, or use software decoding instead).
  * NVIDIA x86 encoding/decoding (NVIDIADecoder.cpp, H264NVIDIAEncoder.cpp, Nvcodec_utils)
    * `cmake -DNVIDIA_SDK_X86=ON ..` (set environment variables `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * Uses the native NVIDIA x86 SDK (https://developer.nvidia.com/video_codec_sdk/downloads/v11).  
      This project uses Video_Codec_SDK_11.0.10, tested with driver version 550.163.01.  
      The files in the Nvcodec_utils directory are extracted from Video_Codec_SDK_11.0.10. Since the SDK contains many files, only those required by this project are included and categorized.  
      Before use, you need to set the encoding mode (not all GPUs support hardware encoding; software encoding is used by default, MiedaWrapper.h → use_nv_enc_flag_).  
      GPU device 0 is used by default (MiedaWrapper.h → device_id_). CUDA must be installed (version not restricted).
  * NVIDIA ARM (Jetson) encoding/decoding (JetsonDecoder.cpp, H264JetsonEncoder.cpp, Jetson_utils)
    * The include and common directories (from /usr/src/jetson_multimedia_api/) may not open correctly on Windows, so they are uploaded as compressed archives and must be extracted on Linux.
    * `cd HardCodec/Jetson_utils`
    * `tar -zxvf include.tar.gz`  
    * `tar -zxvf common.tar.gz`
    * `cmake -DNVIDIA_SDK_ARM=ON ..` (set environment variables `export PATH=$PATH:/usr/local/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`)
    * JetPack version: 5.0.2. JetPack 5.x encoding/decoding is generally compatible, but libraries compiled on 5.0.2 cannot be directly used on other 5.x versions. Recompile the code on the target machine (no need to replace jetson_multimedia_api headers).
    * Reference implementations: jetson_multimedia_api/samples/02_video_dec_cuda, jetson_multimedia_api/samples/01_video_encode
    * Considering Jetson is usually used as an edge device and to reduce latency, B-frame decoding is disabled by default. Enabling B-frames may cause frame reordering issues. To enable B-frame decoding, modify `Jetson_utils → JetsonDec.cpp` and comment out `ret = ctx.dec->disableDPB();` in `JetsonDec::decode_pro`.
    * Test/test2.mp4 contains no B-frames; Test/test1.mp4 and Test/Cognition.mp4 contain B-frames (if you want to test visual perception on Jetson, first modify the code to support B-frames, then use Cognition.mp4, or use software decoding instead).

# Muxing
* mp4
  * Media/MediaMuxer
  * Implemented using FFmpeg
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)

# Visual Perception (YOLO + ByteTrack)
* NVIDIA TensorRT
  * `-DDETECTION_NVIDIA=ON`
  * TensorRT-10.4.0.26
  * `trtexec --onnx=yolo11s_best.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=yolo11s_best.engine --fp16`
* Ascend CANN
  * `-DDETECTION_ASCEND=ON`
  * CANN 7.0.0 / 8.2.RC1
  * `atc --model=yolo11s_best.onnx --framework=5 --input_shape=images:-1,3,640,640 --dynamic_batch_size="1,2,3,4" --insert_op_conf=insert_op.cfg --output=yolo11s_best --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16`
* yolo11s_best.onnx contains two classes: {"dog", "person"}
* Model training: https://github.com/BreakingY/yolo-onnx-tensorrt

# Framework Construction
* Wrapper
* A general media processing and perception framework built on demuxing, encoding/decoding, muxing, and visual perception modules.
* Since MP4 requires writing trailer data at program termination, it is not suitable for RTSP/RTMP real-time streams. Therefore, this part of the code is commented out. If needed, enable `#define MP4MUXER` in `MediaWrapper.cpp`. For real-time streaming, FLV is recommended for file output.

# Notes
* Supported formats: Video: H264/H265, Audio: AAC.
* Visual perception: YOLO11.
* Ascend DVPP has two versions: V1 and V2. V1 and V2 apply to different platforms. Please refer to the official documentation. Newer Ascend cards generally support V2.
* Tested versions: FFmpeg 4.0.5 (requires FFmpeg 4.x; audio uses fdk-aac encoding, ensure FFmpeg is built with fdk-aac), OpenCV 4.5.1, CANN 7.0.0 / 8.2.RC1 (Ascend SDK), NVIDIA: CUDA 12.4; NVIDIA driver 550.163.01; Video_Codec_SDK 11.0.10; Jetson 5.0.2; TensorRT 10.4.0.26.
* ByteTrack dependency: `apt install libeigen3-dev`
* Jetson dependency: v4l2
* librtmp dependency: openssl
* Windows software installation reference:
  * https://sunkx.blog.csdn.net/article/details/146064215
* Code module structure is shown below:
![MCP](https://github.com/user-attachments/assets/bdb98d02-eaa6-4ad8-b30b-e2d3da399056)

# Acknowledgements (.gitmodules submodules)
* spdlog: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec
* ByteTrack: https://github.com/Vertical-Beach/ByteTrack-cpp
* libflv: https://github.com/BreakingY/libflv
* librtmp: https://git.ffmpeg.org/rtmpdump.git

# Build
* `git clone --recursive https://github.com/BreakingY/Media-Cognition-Pipeline.git`
1. Linux
   * `mkdir build`
   * `cd build`
   * `cmake -DFFMPEG_SOFT=ON ..`
   * `make -j`
2. Windows (MinGW + CMake)
   * `mkdir build`
   * `cd build`
   * `cmake -G "MinGW Makefiles" -DFFMPEG_SOFT=ON ..`
   * `mingw32-make -j`
3. Visual Perception
   * NVIDIA: `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/DVPP_MPI/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_NVIDIA=ON ..`
   * ASCEND: `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/DVPP_MPI/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_ASCEND=ON ..`

# Testing
1. Pipeline test:  
   `./MediaCodec <mp4(../Test/test*.mp4)>/<flv(../Media/RtmpClient/libflv/test/test_1280x720_h264_aac.flv)>/<rtsp url>/<rtmp url> <mp4>/<flv>/<rtmp url>`
2. AI inference:  
   `./MediaCodec ../Test/Cognition.mp4 <mp4>/<flv>/<rtmp url>`

# Technical Contact
* kxsun617@163.com