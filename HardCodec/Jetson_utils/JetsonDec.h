#ifndef JETSON_DEC_H_
#define JETSON_DEC_H_
#include "NvCudaProc.h"
#include "NvUtils.h"
#include <assert.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/prctl.h>
#include <unistd.h>

#include "nvosd.h"
#include "videodec.h"

#define TEST_ERROR(cond, str, label) \
    if (cond) {                      \
        cerr << str << endl;         \
        error = 1;                   \
        goto label;                  \
    }

#define CHUNK_SIZE 4000000
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
                                       !buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
                                        (buffer_ptr[2] == 1))

#define BORDER_WIDTH 5

#define FIRST_CLASS_CNT 1

using namespace std;

typedef struct MediaDataSt {
    unsigned char *data;
    int len;
    struct timeval time;
    MediaDataSt()
    {
        data = NULL;
        len = 0;
    };
    ~MediaDataSt()
    {
        if (data) {
            free(data);
            data = NULL;
        }
    };
} MediaData;
class JetsonDecListner
{
public:
    virtual void OnJetsonDecData(unsigned char *data, int data_len, uint64_t timestamp) = 0; // timestamp 为JetsonDec::AddEsData中传入的时间戳(视频解码前的原始时间戳)
};

class JetsonDec
{
public:
    // decoder_pixfmt:V4L2_PIX_FMT_H264 V4L2_PIX_FMT_H265
    JetsonDec(uint32_t decoder_pixfmt, int width = -1, int height = -1, unsigned char *buffer = NULL);
    ~JetsonDec();
    void SetDecCallBack(JetsonDecListner *call_func);
    void UnSetDecCallBack();
    void AddEsData(unsigned char *data, int len, uint64_t time_data); // 当前data时间戳，毫秒
    int GetQueueSize();
public:
    //jetson dec func
    static void *decode_proc(void *arg);
    static void *dec_capture_loop_fcn(void *arg);
    int read_decoder_input_nalu(NvBuffer *buffer, char *parse_buffer, streamsize parse_buffer_size, struct timeval &time);
    int dump_dmabuf_ptr(int dmabuf_fd, unsigned int plane, int *len, unsigned char *outbuffer);

private:
    JetsonDecListner *p_callback;
    bool m_abort = false;
    bool m_abort_cap = false;
    pthread_t job_tid;
    pthread_t dec_tid;
    bool proc_ready = false;
    pthread_mutex_t mutex_data;
    pthread_cond_t cond_data;
    list<MediaData *> data_list;
    context_t ctx;
    uint32_t dec_pixfmt;
    unsigned char *dec_buffer = NULL;
    bool flag_buffer = false;
};
#endif
