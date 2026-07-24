# Media-Cognition-Pipeline
A general-purpose real-time streaming media and deep learning inference acceleration framework, supporting H264, H265, AAC, MP4, FLV, TS, RTSP, RTMP, SRT and YOLO.
* Audio/video demuxing (MP4, FLV, TS, RTSP, RTMP, SRT(TODO)), resampling, encoding/decoding (H264, H265, AAC; NVIDIA, Ascend, Hygon DCU), muxing (MP4, FLV, TS, RTMP, SRT), and visual perception (YOLO object detection + ByteTrack multi-object tracking; NVIDIA, Ascend) pipeline, managed with a modular, node-based, and interface-oriented design.

# Demuxing
* mp4
  * Media/FileReader
  * Implemented using FFmpeg
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)
* ts/srt
  * Media/TsTransport
  * libmpeg2 (https://github.com/BreakingY/libmpeg2core)
  * TS: supported by default; SRT: requires enabling the CMake option `-DENABLE_SRT=ON` and libsrt needs to be installed.
  * libsrt installation
    * git clone https://github.com/Haivision/srt.git
    * cd srt && ./configure
    * make && make install
* rtsp
  * Media/RtspReader
  * simple-rtsp-client (https://github.com/BreakingY/simple-rtsp-client)

# Encoding / Decoding(Only one of them can be activated)
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

# Muxing
* mp4
  * Media/MediaMuxer
  * Implemented using FFmpeg
* flv/rtmp
  * Media/RtmpClient
  * libflv (https://github.com/BreakingY/libflv) + librtmp (https://git.ffmpeg.org/rtmpdump.git)
* ts/srt
  * Media/TsTransport
  * libmpeg2 (https://github.com/BreakingY/libmpeg2core)
  * TS: supported by default; SRT: requires enabling the CMake option `-DENABLE_SRT=ON` and libsrt needs to be installed.
  * libsrt installation
    * git clone https://github.com/Haivision/srt.git
    * cd srt && ./configure
    * make && make install

# Visual Perception (YOLO + ByteTrack)
* NVIDIA TensorRT
  * `-DDETECTION_NVIDIA=ON`
  * TensorRT-10.4.0.26
  * `trtexec --onnx=yolo11s_best.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=yolo11s_best.engine --fp16`
* Ascend CANN
  * `-DDETECTION_ASCEND=ON`
  * CANN 7.0.0 / 8.2.RC1
  * `atc --model=yolo11s_best.onnx --framework=5 --input_shape=images:-1,3,640,640 --dynamic_batch_size="1,2,3,4" --insert_op_conf=insert_op.cfg --output=yolo11s_best --soc_version=Ascend310P3 --precision_mode_v2=mixed_float16`
* Hygon MIGraphX
  * `-DDETECTION_HYGON=ON`
  * `source /etc/profile.d/dtk.sh`
  * `source /etc/profile.d/migraphx.sh`
  * `source /opt/dtk/cuda/env.sh`
  * dtk26.04/migraphx5.2.0
  * dynamic batch (migraphx-driver does not support yolo11 temporarily)
    * `migraphx-driver compile --onnx yolo11s_best.onnx --dyn-input-dim @images "[{min:1,max:8,optimals:[1,4,8]},3,640,640]" --binary -o yolo11s_best.mxr --fp16 --gpu`
  * static batch
    * `migraphx-driver compile --onnx yolo11s_best.onnx --input-dim @images 4 3 640 640 --binary -o yolo11s_best.mxr --fp16 --gpu`
* yolo11s_best.onnx contains two classes: {"dog", "person"}
* Model training: https://github.com/BreakingY/yolo-onnx-tensorrt

# Framework Construction
* Wrapper
* A general media processing and perception framework built on demuxing, encoding/decoding, muxing, and visual perception modules.
* MP4 requires writing finalization metadata at the end of the file, which makes it unsuitable for RTSP/RTMP real-time streaming. FLV and TS are recommended formats for recording real-time streams.

# Notes
* Supported formats: Video: H264/H265, Audio: AAC.
* Visual perception: YOLO11.
* Tested versions:
  * FFmpeg 4.0.5 (requires FFmpeg 4.x; audio uses fdk-aac encoding, ensure FFmpeg is built with fdk-aac)
  * opencv 4.5.1
  * Ascend: CANN 7.0.0/8.2.RC1
  * NVIDIA: 
    * TensorRT: 10.4.0.26
    * X86: cuda 12.4; NVIDIA driver 550.163.01; Video_Codec_SDK 11.0.10
    * ARM: Jetson 5.0.2
  * Hygon DCU: DTK 26.04; migraphx 5.2.0
* ByteTrack dependency: `apt install libeigen3-dev`
* Jetson dependency: v4l2
* librtmp dependency: openssl
* Windows software installation reference:
  * https://sunkx.blog.csdn.net/article/details/146064215
* Code module structure is shown below:
![MCP](https://github.com/user-attachments/assets/6a9cbe6a-8a61-47de-bdac-81dc5865e182)

# Acknowledgements (.gitmodules submodules)
* spdlog: https://github.com/gabime/spdlog
* Bitstream: https://github.com/ireader/avcodec
* ByteTrack: https://github.com/Vertical-Beach/ByteTrack-cpp
* librtmp: https://git.ffmpeg.org/rtmpdump
* pybind11: https://github.com/pybind/pybind11 v2.12.0
* libflv: https://github.com/BreakingY/libflv
* simple-rtsp-client: https://github.com/BreakingY/simple-rtsp-client
* libmpeg2core: https://github.com/BreakingY/libmpeg2core

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
   * NVIDIA: `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/NVIDIA_SDK_X86/NVIDIA_SDK_ARM>=ON -DDETECTION_NVIDIA=ON ..`
   * ASCEND: `cmake -D<FFMPEG_SOFT/DVPP_MPI>=ON -DDETECTION_ASCEND=ON ..`
   * Hygon : `cmake -DFFMPEG_SOFT=ON -DDETECTION_HYGON=ON ..`

# Testing
1. Pipeline test:
   `./MediaCodec <mp4(../Test/test*.mp4)>/<flv(../Media/RtmpClient/libflv/test/test_1280x720_h264_aac.flv)/ts(../Media/TsTransport/libmpeg2core/media/h264_aac.ts)>/<srt url>/<rtsp url>/<rtmp url> <mp4>/<flv>/<ts>/<srt url>/<rtmp url>`
2. AI inference:
   `./MediaCodec ../Test/Cognition.mp4 <mp4>/<flv>/<ts>/<srt url>/<rtmp url>`

   https://github.com/user-attachments/assets/59b77cfd-b6a7-4fcd-b6ab-acec722d74e2

# Python Interface: Real-time Streaming Media Acceleration Processing Library (Input: File/RTSP/RTMP/SRT; Output: File/RTSP/RTMP/SRT)
* Streaming Media Processing Library: Provides high-performance video processing capabilities for Python, including audio/video capture, encoding/decoding acceleration, frame data interaction, and real-time streaming media processing.
* Typical Application Scenario: C++ handles video capture (file, real-time stream), encoding/decoding acceleration, and output (file, real-time stream); Python handles model algorithms such as object detection, face recognition, and behavior analysis.
* `-DMCP_PYBIND=ON`: When enabled, it serves only as a media library for Python and cannot enable `DETECTION_NVIDIA/DETECTION_ASCEND/DETECTION_HYGON`.
* To specify a particular Python version, refer to the commands below, e.g., compiling for Python 3.12 and Python 3.7:
  * `-DPYTHON_LIBRARY=/usr/local/lib/libpython3.12.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.12 -DPYTHON_EXECUTABLE=/usr/local/bin/python3.12`
  * `-DPYTHON_LIBRARY=/usr/local/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.7m -DPYTHON_EXECUTABLE=/usr/local/bin/python3.7`
* Build Steps:
  1. Compile Python from source with `./configure --enable-shared`. The `--enable-shared` flag is required; otherwise, compilation will fail.
  2. cd build
  3. `cmake -D<FFMPEG_SOFT/FFMPEG_NVIDIA/NVIDIA_SDK_X86/NVIDIA_SDK_ARM/DVPP_MPI>=ON -DMCP_PYBIND=ON ..`
  4. python demo.py

# Technical Contact
* kxsun617@163.com
