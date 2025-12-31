#ifndef JETSON_ENC_H_
#define JETSON_ENC_H_

#include "NvUtils.h"
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <poll.h>
#include <sstream>
#include <string.h>

#include "video_encode.h"
#include <list>

#define TEST_ERROR(cond, str, label) \
    if (cond) {                      \
        cerr << str << endl;         \
        error = 1;                   \
        goto label;                  \
    }

#define TEST_PARSE_ERROR(cond, label)                                     \
    if (cond) {                                                           \
        cerr << "Error parsing runtime parameter changes string" << endl; \
        goto label;                                                       \
    }

#define IS_DIGIT(c) (c >= '0' && c <= '9')
#define MICROSECOND_UNIT 1000000

using namespace std;

typedef struct YUVDataSt {
    unsigned char *data;
    int len;
    YUVDataSt()
    {
        data = NULL;
        len = 0;
    };
    ~YUVDataSt()
    {
        if (data) {
            free(data);
            data = NULL;
        }
    };
} YUVData;
class JetsonEncListner
{
public:
    virtual void OnJetsonEncData(unsigned char *data, int data_len) = 0;
};
class JetsonEnc
{
public:
    JetsonEnc(int width, int height, int video_fps);
    ~JetsonEnc();
    void SetDecCallBack(JetsonEncListner *call_func);
    void UnSetDecCallBack();
    void AddFrame(unsigned char *data, int len);
    int GetQueueSize();

public:
    // jetson enc func
    static void *encode_proc(void *arg);
    static bool encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer, NvBuffer *shared_buffer, void *arg);
    int encoder_proc_nonblocking(context_t_enc &ctx, bool eos);
    int read_video_frame_my(std::ifstream *stream, NvBuffer &buffer);
    int encoder_proc_blocking(context_t_enc &ctx, bool eos);

public:
    JetsonEncListner *p_callback;
    pthread_mutex_t data_mutex;
    pthread_cond_t data_cond;
    list<YUVData *> data_list;
    bool m_abort = false;
    int image_width;
    int image_height;
    int fps;
    context_t_enc ctx;
    pthread_t job_tid;
};
#endif
