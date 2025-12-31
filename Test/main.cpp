#include "MediaWrapper.h"
#include "log_helpers.h"
#include <iostream>
int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::debug);
    if (argc < 3) {
        log_info("only support H264/H265 AAC");
        log_info("./bin input ouput");
        return -1;
    }
    av_log_set_level(AV_LOG_FATAL);
#ifdef USE_DVPP_MPI
    aclInit(NULL);
#endif
#if defined(DETECTION_NVIDIA)
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2], "../Test/yolo11s_best_4090.engine");
#elif defined(DETECTION_ASCEND)
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2], "../Test/yolo11s_best_300VPro.om");
#else
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2]);
#endif
    while (!test->OverHandle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    delete test;
#ifdef USE_DVPP_MPI
    hi_mpi_sys_exit();
    aclFinalize();
#endif
    log_info("over");
    return 0;
}
