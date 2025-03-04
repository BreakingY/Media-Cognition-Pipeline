#include "MediaWrapper.h"
#include "log_helpers.h"
#include <iostream>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
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
    hi_mpi_sys_init();
#endif
    struct rlimit core_limits;
    core_limits.rlim_cur = core_limits.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &core_limits);
    MiedaWrapper *test = new MiedaWrapper(argv[1], argv[2]);
    while (!test->OverHandle()) {
        usleep(1000 * 100);
    }
    delete test;
#ifdef USE_DVPP_MPI
    hi_mpi_sys_exit();
    aclFinalize();
#endif
    log_info("over");
    return 0;
}
