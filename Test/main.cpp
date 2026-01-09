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
#if defined(USE_DVPP_MPI) || defined(DETECTION_ASCEND)
    aclInit(NULL);
#endif
#if defined(DETECTION_NVIDIA)
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2], "../Test/yolo11s_best.engine");
#elif defined(DETECTION_ASCEND)
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2], "../Test/yolo11s_best.om");
#else
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2]);
#endif
    while (!test->OverHandle()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    delete test;
#if defined(USE_DVPP_MPI) || defined(DETECTION_ASCEND)
    hi_mpi_sys_exit();
    aclFinalize();
#endif
    log_info("over");
    return 0;
}
