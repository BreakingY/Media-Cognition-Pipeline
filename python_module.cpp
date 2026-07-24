#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include "MediaWrapper.h"
namespace py = pybind11;
static void log_init(){
	spdlog::set_level(spdlog::level::debug);
	log_info("log init ok");
    return;
}
static void cleanup_logger(){
	return;
}
void initialization(){
    av_log_set_level(AV_LOG_FATAL);
#if defined(USE_DVPP_MPI) || defined(DETECTION_ASCEND)
    aclInit(NULL);
#endif
    log_init();
}
void cleanUp(){
#if defined(USE_DVPP_MPI) || defined(DETECTION_ASCEND)
    hi_mpi_sys_exit();
    aclFinalize();
#endif
    cleanup_logger();
}
void* createMcpObject(const char *rtsp, const char *rtmp, int device_id = 0){
    MediaWrapper *wrapper = new MediaWrapper(rtsp, rtmp, nullptr, device_id);
    return (void *)wrapper;
}

void destoryMcpObject(void* obj){
    if(obj){
       MediaWrapper *wrapper = (MediaWrapper *)obj;
       delete wrapper;
    }
    return;
}

bool overHandle(void* obj){
    if(obj){
       MediaWrapper *wrapper = (MediaWrapper *)obj;
       return wrapper->OverHandle();
    }
    return false;
}

void registerImgcallback(void* obj, FrameCallbackFuncVideo video_cb, FrameCallbackFuncAudio audio_cb) {
    if(obj){
       MediaWrapper *wrapper = (MediaWrapper *)obj;
       wrapper->InitImgPycallback(video_cb, audio_cb);
    }
}
void addVideoFrame(void* obj, py::array_t<uint8_t> arr){
    if(obj){
       MediaWrapper *wrapper = (MediaWrapper *)obj;
       py::buffer_info buf = arr.request();
       cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC(buf.shape[2]), buf.ptr);
       wrapper->PyAddVideoFrame(std::move(frame.clone()));
    }
}
/**
 * arr:packed模式pcm数据
 * data_len:单通道当本个数
 * spb:每个样本占用的字节数
 * channels:通道数
 */
void addAudioFrame(void* obj, py::array_t<uint8_t> arr, int data_len, int spb, int channels){
    if(obj){
        MediaWrapper *wrapper = (MediaWrapper *)obj;
        py::buffer_info buf = arr.request();
        uint8_t* pcm = static_cast<uint8_t*>(buf.ptr);
        wrapper->PyAddAudioFrame(pcm, data_len, spb, channels);
    }
}
PYBIND11_MODULE(mcp_module, m) {
    m.doc() = "mcp_module";
    m.def("initialization", &initialization);
    m.def("cleanUp", &cleanUp);

	m.def("createMcpObject", &createMcpObject);
    m.def("destoryMcpObject", &destoryMcpObject);
    m.def("overHandle", &overHandle);

    m.def("registerImgcallback", &registerImgcallback);
    m.def("addVideoFrame", &addVideoFrame);
    m.def("addAudioFrame", &addAudioFrame);
}

