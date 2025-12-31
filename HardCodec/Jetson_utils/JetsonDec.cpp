#include "JetsonDec.h"
#include <chrono>

JetsonDec::JetsonDec(uint32_t decoder_pixfmt, int width, int height, unsigned char *buffer)
{
    p_callback = NULL;
    pthread_mutex_init(&mutex_data, NULL);
    pthread_cond_init(&cond_data, NULL);
    data_list.clear();
    dec_pixfmt = decoder_pixfmt;
    m_abort = false;
    if (buffer != NULL) {
        dec_buffer = buffer;
        flag_buffer = false;
    } else if (width > 0 && height > 0) {
        int size = width * height * 3 * 2;
        dec_buffer = (unsigned char *)malloc(size);
        printf("JetsonDec w:%d h:%d\n", width, height);
        flag_buffer = true;
    }
    pthread_create(&job_tid, NULL, decode_proc, this);
    pthread_setname_np(job_tid, "OutputPlane");
}
void JetsonDec::SetDecCallBack(JetsonDecListner *call_func)
{
    p_callback = call_func;
    return;
}
void JetsonDec::UnSetDecCallBack()
{
    p_callback = NULL;
    return;
}
JetsonDec::~JetsonDec()
{
    m_abort = true;
    int ret = pthread_join(job_tid, NULL);
    if (ret != 0) {
        printf("pthread_join iob_tid error\n");
    }
    pthread_mutex_destroy(&mutex_data);
    pthread_cond_destroy(&cond_data);
    for (std::list<MediaData *>::iterator it = data_list.begin(); it != data_list.end(); ++it) {
        MediaData *node = *it;
        delete node;
    }
    data_list.clear();

    if (flag_buffer && dec_buffer != NULL) {
        free(dec_buffer);
        dec_buffer = NULL;
    }
    printf("~JetsonDec()\n");
}
static void convert_ms_to_timeval(uint64_t ms, struct timeval &tv) {
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
}
void JetsonDec::AddEsData(unsigned char *data, int len, uint64_t time_data)
{
    MediaData *pData = new MediaData();
    pData->data = (unsigned char *)malloc(len);
    memcpy(pData->data, data, len);
    pData->len = len;
    struct timeval tv;
    convert_ms_to_timeval(time_data, tv);
    pData->time = tv;

    pthread_mutex_lock(&mutex_data);
    data_list.push_back(pData);
    pthread_mutex_unlock(&mutex_data);
    pthread_cond_signal(&cond_data);
    return;
}
int JetsonDec::GetQueueSize()
{
    int size = 0;
    pthread_mutex_lock(&mutex_data);
    size = data_list.size();
    pthread_mutex_unlock(&mutex_data);
    return size;
}
static long long tm_to_ns(struct timespec tm)
{
    return tm.tv_sec * 1000000000 + tm.tv_nsec;
}

static struct timespec ns_to_tm(long long ns)
{
    struct timespec tm;
    tm.tv_sec = ns / 1000000000;
    tm.tv_nsec = ns - (tm.tv_sec * 1000000000);
    return tm;
}
int JetsonDec::read_decoder_input_nalu(NvBuffer *buffer, char *parse_buffer, streamsize parse_buffer_size, struct timeval &time)
{
    pthread_mutex_lock(&mutex_data);
    while (data_list.empty() && !m_abort) {
        struct timespec start_tm;
        struct timespec end_tm;
        int timeout_ms = 40;
        clock_gettime(CLOCK_REALTIME, &start_tm);
        end_tm = ns_to_tm(tm_to_ns(start_tm) + timeout_ms * 1000000);
        auto start = std::chrono::steady_clock::now();
        pthread_cond_timedwait(&cond_data, &mutex_data, &end_tm);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        int elapsed_ms = duration.count();

        // std::cout << "read_decoder_input_nalu wait: " << elapsed_ms << " 毫秒" << std::endl;
    }
    if (m_abort) {
        pthread_mutex_unlock(&mutex_data);
        return 0;
    }
    // printf("size:%d\n",data_list.size());
    MediaData *node = data_list.front();
    data_list.pop_front();

    pthread_mutex_unlock(&mutex_data);

    char *buffer_ptr = (char *)buffer->planes[0].data;
    memcpy(buffer_ptr, node->data, node->len);
    buffer->planes[0].bytesused = node->len;
    time = node->time;
    delete node;
    proc_ready = true;
    return 0;
}
int JetsonDec::dump_dmabuf_ptr(int dmabuf_fd, unsigned int plane, int *len, unsigned char *outbuffer)
{
    if (dmabuf_fd <= 0)
        return -1;

    int ret = -1;

    NvBufSurface *nvbuf_surf = 0;
    ret = NvBufSurfaceFromFd(dmabuf_fd, (void **)(&nvbuf_surf));
    if (ret != 0) {
        return -1;
    }
    ret = NvBufSurfaceMap(nvbuf_surf, 0, plane, NVBUF_MAP_READ_WRITE);
    if (ret < 0) {
        printf("NvBufSurfaceMap failed\n");
        return ret;
    }
    NvBufSurfaceSyncForCpu(nvbuf_surf, 0, plane);
    int pos = 0;
    for (uint i = 0; i < nvbuf_surf->surfaceList->planeParams.height[plane]; ++i) {
        memcpy(outbuffer + pos, (char *)nvbuf_surf->surfaceList->mappedAddr.addr[plane] + i * nvbuf_surf->surfaceList->planeParams.pitch[plane],
               nvbuf_surf->surfaceList->planeParams.width[plane] * nvbuf_surf->surfaceList->planeParams.bytesPerPix[plane]);
        pos += nvbuf_surf->surfaceList->planeParams.width[plane] * nvbuf_surf->surfaceList->planeParams.bytesPerPix[plane];
    }
    ret = NvBufSurfaceUnMap(nvbuf_surf, 0, plane);
    if (ret < 0) {
        printf("NvBufSurfaceUnMap failed\n");
        return ret;
    }
    *len = pos;
    return 0;
}
static void abort(context_t *ctx)
{
    printf("ctx error\n");
    exit(1);
    return;
}

/**
 * Set the default values for decoder context members.
 */
static void set_defaults(context_t *ctx)
{
    memset(ctx, 0, sizeof(context_t));
    ctx->fullscreen = false;
    ctx->window_height = 0;
    ctx->window_width = 0;
    ctx->window_x = 0;
    ctx->window_y = 0;
    ctx->out_pixfmt = 1;
    // ctx->fps = 30; // not use
    ctx->nvosd_context = NULL;
    ctx->dst_dma_fd = -1;
    return;
}

/**
 * Query and Set Capture plane.
 */
static void
query_and_set_capture(context_t *ctx)
{
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    int32_t min_dec_capture_buffers;
    int ret = 0;
    int error = 0;
    uint32_t window_width;
    uint32_t window_height;
    NvBufSurf::NvCommonAllocateParams params;

    /* Get capture plane format from the decoder.
       This may change after resolution change event.
       Refer ioctl VIDIOC_G_FMT */
    ret = dec->capture_plane.getFormat(format);
    TEST_ERROR(ret < 0,
               "Error: Could not get format from decoder capture plane", error);

    /* Get the display resolution from the decoder.
       Refer ioctl VIDIOC_G_CROP */
    ret = dec->capture_plane.getCrop(crop);
    TEST_ERROR(ret < 0,
               "Error: Could not get crop from decoder capture plane", error);

    cout << "Video Resolution: " << crop.c.width << "x" << crop.c.height
         << endl;

    ctx->dec_width = crop.c.width;
    ctx->dec_height = crop.c.height;

    if (ctx->dst_dma_fd != -1) {
        ret = NvBufSurf::NvDestroy(ctx->dst_dma_fd);
        ctx->dst_dma_fd = -1;
        TEST_ERROR(ret < 0, "Error: Error in BufferDestroy", error);
    }

    /* Create PitchLinear output buffer for transform. */
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
    params.width = crop.c.width;
    params.height = crop.c.height;
    params.layout = NVBUF_LAYOUT_PITCH;
    if (ctx->out_pixfmt == 1)
        params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    else if (ctx->out_pixfmt == 2)
        params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
    else if (ctx->out_pixfmt == 3)
        params.colorFormat = NVBUF_COLOR_FORMAT_NV16;
    else if (ctx->out_pixfmt == 4)
        params.colorFormat = NVBUF_COLOR_FORMAT_NV24;

    if (ctx->enable_osd_text)
        params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;

    params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

    ret = NvBufSurf::NvAllocate(&params, 1, &ctx->dst_dma_fd); // 分配1个内存，因为后续的使用和销毁默认都是1，如果需要创建多个buffer，则需要修改对应的释放部分，以防出现内存泄漏。
    TEST_ERROR(ret == -1, "create dmabuf failed", error);
#if 0
    if (!ctx->disable_rendering)
    {
        /* Destroy the old instance of renderer as resolution
           might have changed */
        delete ctx->renderer;

        if (ctx->fullscreen)
        {
            /* Required for fullscreen */
            window_width = window_height = 0;
        }
        else if (ctx->window_width && ctx->window_height)
        {
            /* As specified by user on commandline */
            window_width = ctx->window_width;
            window_height = ctx->window_height;
        }
        else
        {
            /* Resolution got from the decoder */
            window_width = crop.c.width;
            window_height = crop.c.height;
        }

        /* If height or width are set to zero, EglRenderer creates a fullscreen
           window for rendering */
        ctx->renderer =
            NvEglRenderer::createEglRenderer("renderer0", window_width,
                                           window_height, ctx->window_x,
                                           ctx->window_y);
        TEST_ERROR(!ctx->renderer,
                   "Error in setting up renderer. "
                   "Check if X is running or run with --disable-rendering",
                   error);

        /* Set fps for rendering */
        ctx->renderer->setFPS(ctx->fps);
    }
#endif
    /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
    dec->capture_plane.deinitPlane();

    /* Not necessary to call VIDIOC_S_FMT on decoder capture plane. But
       decoder setCapturePlaneFormat function updates the class variables */
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                     format.fmt.pix_mp.width,
                                     format.fmt.pix_mp.height);
    TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

    /* Get the min buffers which have to be requested on the capture plane */
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
    TEST_ERROR(ret < 0,
               "Error while getting value of minimum capture plane buffers",
               error);

    /* Request, Query and export (min + 5) decoder capture plane buffers.
       Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
    ret =
        dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                      min_dec_capture_buffers + 10, false,
                                      false);
    TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);

    /* For file write, first deinitialize output and capture planes
       of video converter and then use the new resolution from
       decoder resolution change event */

    /* Start streaming on decoder capture_plane */
    ret = dec->capture_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

    /* Enqueue all the empty capture plane buffers */
    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
        TEST_ERROR(ret < 0, "Error Qing buffer at output plane", error);
    }

    cout << "Query and set capture successful" << endl;
    return;

error:
    if (error) {
        abort(ctx);
        cerr << "Error in " << __func__ << endl;
    }
}
/**
 * Decoder capture thread loop function.
 */
void *JetsonDec::dec_capture_loop_fcn(void *arg)
{

    JetsonDec *self = (JetsonDec *)arg;
    unsigned char *ptr = self->dec_buffer; // 这里需要使用ptr把dec_buffer记录下来，否则后面这个地址可能会失效(地址发生变化)

    while (!self->m_abort_cap && !self->proc_ready) {
        usleep(1000);
        continue;
    }
    context_t *ctx = &self->ctx;
    NvVideoDecoder *dec = ctx->dec;
    struct v4l2_event ev;
    int ret;

    cout << "Starting decoder capture loop thread" << endl;
    prctl(PR_SET_NAME, "dec_cap", 0, 0, 0);

    /* Wait for the first Resolution change event as decoder needs
       to know the stream resolution for allocating appropriate
       buffers when calling REQBUFS */
    do {
        /* VIDIOC_DQEVENT, max_wait_ms = 1000ms */
        ret = dec->dqEvent(ev, 1000);

        if (ret < 0) {
            if (errno == EAGAIN) {
                cerr << "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE"
                     << endl;
            } else {
                cerr << "Error in dequeueing decoder event" << endl;
            }
            abort(ctx);
            break;
        }
    } while (!self->m_abort_cap && ev.type != V4L2_EVENT_RESOLUTION_CHANGE);
    printf("query_and_set_capture bre dec_buffer:%p ptr:%p\n", self->dec_buffer, ptr);
    /* Received the resolution change event, now can do query_and_set_capture */
    if (!self->m_abort_cap) {
        query_and_set_capture(ctx);
    }
    printf("query_and_set_capture after dec_buffer:%p ptr:%p\n", self->dec_buffer, ptr);
    /* Exit on error or EOS which is signalled in main() */
    while (!(self->m_abort_cap || dec->isInError())) {
        NvBuffer *dec_buffer;

        /* Check for resolution change again */
        ret = dec->dqEvent(ev, false);
        if (ret == 0) {
            switch (ev.type) {
            case V4L2_EVENT_RESOLUTION_CHANGE:
                query_and_set_capture(ctx);
                continue;
            }
        }
        /* Decoder capture loop */
        while (1) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            /* Dequeue a valid capture_plane buffer that contains YUV BL data */
            if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)) {
                if (errno == EAGAIN) {
                    printf("EAGAIN\n");
                    usleep(1000); // 1ms
                } else {
                    abort(ctx);
                    cerr << "Error while calling dequeue at capture plane" << endl;
                }
                break;
            }

            /* Clip & Stitch can be done by adjusting rectangle. */
            NvBufSurf::NvCommonTransformParams transform_params;
            transform_params.src_top = 0;
            transform_params.src_left = 0;
            transform_params.src_width = ctx->dec_width;
            transform_params.src_height = ctx->dec_height;
            transform_params.dst_top = 0;
            transform_params.dst_left = 0;
            transform_params.dst_width = ctx->dec_width;
            transform_params.dst_height = ctx->dec_height;
            transform_params.flag = NVBUFSURF_TRANSFORM_FILTER;
            transform_params.flip = NvBufSurfTransform_None;
            transform_params.filter = NvBufSurfTransformInter_Nearest;

            /* Perform Blocklinear to PitchLinear conversion. */
            ret = NvBufSurf::NvTransform(&transform_params, dec_buffer->planes[0].fd, ctx->dst_dma_fd);
            if (ret == -1) {
                cerr << "Transform failed" << endl;
                break;
            }
            /* Write raw video frame to file. */
            int len = 0, len1 = 0;
            ret = self->dump_dmabuf_ptr(ctx->dst_dma_fd, 0, &len, ptr);
            ret = self->dump_dmabuf_ptr(ctx->dst_dma_fd, 1, &len1, ptr + len);
            if (self->p_callback) {
                uint64_t v4l2_time = 1000 * v4l2_buf.timestamp.tv_sec + v4l2_buf.timestamp.tv_usec / 1000;
                self->p_callback->OnJetsonDecData((unsigned char *)ptr, len + len1, v4l2_time);
            }

            /* If not writing to file, Queue the buffer back once it has been used. */
            if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                cerr << "Error while queueing buffer at decoder capture plane"
                     << endl;
                break;
            }
        }
    }

    cout << "Exiting decoder capture loop thread" << endl;
    return NULL;
}
void *JetsonDec::decode_proc(void *arg)
{
    JetsonDec *self = (JetsonDec *)arg;
    context_t &ctx = self->ctx;
    int ret = 0;
    int error = 0;
    uint32_t i;

    /* Set default values for decoder context members */
    set_defaults(&ctx);
#if 0
    ctx.disable_rendering=true;
#endif

    ctx.decoder_pixfmt = self->dec_pixfmt;
    ctx.out_pixfmt = 1; // NV12
    ctx.input_nalu = true;
#if 0
    if (ctx.enable_osd || ctx.enable_osd_text)
        ctx.nvosd_context = nvosd_create_context();
    if (ctx.enable_osd) {
        cout << "ctx.osd_file_path:" << ctx.osd_file_path << endl;

        ctx.osd_file = new ifstream(ctx.osd_file_path);
        TEST_ERROR(!ctx.osd_file->is_open(), "Error opening osd file", cleanup);
    }
#endif

    /* Create and initialize video decoder
       more about decoder, refer to 00_video_decode sample */
    ctx.dec = NvVideoDecoder::createVideoDecoder("dec0");
    TEST_ERROR(!ctx.dec, "Could not create decoder", cleanup);

    /* Subscribe to Resolution change event */
    ret = ctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE",
               cleanup);

    /* Set the max size of the outputPlane buffers, here is
       CHUNK_SIZE, which contains the encoded data in bytes */
    ret = ctx.dec->setOutputPlaneFormat(ctx.decoder_pixfmt, CHUNK_SIZE);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    ret = ctx.dec->setFrameInputMode(0);
    TEST_ERROR(ret < 0, "Error in decoder setFrameInputMode", cleanup);

    ret = ctx.dec->disableDPB();
    TEST_ERROR(ret < 0, "Error in decoder disableDPB", cleanup);
    ret = ctx.dec->setMaxPerfMode(1);
    TEST_ERROR(ret < 0, "Error while setting decoder to max perf", cleanup);

    /* Request MMAP buffers for writing encoded video data */
    ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    TEST_ERROR(ret < 0, "Error while setting up output plane", cleanup);

    /* Start streaming on decoder output_plane */
    ret = ctx.dec->output_plane.setStreamStatus(true);
    TEST_ERROR(ret < 0, "Error in output plane stream on", cleanup);

    // self->proc_ready = true;
    pthread_create(&self->dec_tid, NULL, dec_capture_loop_fcn, self);
    pthread_setname_np(self->dec_tid, "CapturePlane");

    /* Read encoded data and enqueue all the output plane buffers.
       Exit loop in case end of file */
    i = 0;
    while (!self->m_abort && !ctx.dec->isInError() && i < ctx.dec->output_plane.getNumBuffers()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        buffer = ctx.dec->output_plane.getNthBuffer(i);

        struct timeval time_now;
        self->read_decoder_input_nalu(buffer, NULL, CHUNK_SIZE, time_now);

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        /* It is necessary to queue an empty buffer to signal EOS to sthe decoder
           i.e. set v4l2_buf.m.planes[0].bytesused = 0 and queue the buffer */
        v4l2_buf.timestamp = time_now;
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            cout << "Input file read complete" << endl;
            break;
        }
        i++;
    }

    /* Since all the output plane buffers have been queued in above loop,
       in this loop, firstly dequeue a empty buffer, then read encoded data
       into this buffer, enqueue it back for decoding at last */
    while (!self->m_abort && !ctx.dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
        if (ret < 0) {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        struct timeval time_now;
        self->read_decoder_input_nalu(buffer, NULL, CHUNK_SIZE, time_now);

        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
        v4l2_buf.timestamp = time_now;
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            cout << "Input file read complete" << endl;
            break;
        }
    }

    /* As EOS, dequeue all the output planes */
    while (ctx.dec->output_plane.getNumQueuedBuffers() > 0 && !ctx.dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if (ret < 0) {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
    }

cleanup:
    self->m_abort_cap = true;
    ret = pthread_join(self->dec_tid, NULL);
    if (ret != 0) {
        printf("pthread_join iob_tid error\n");
    }
    /* The decoder destructor does all the cleanup i.e set streamoff on output
       and capture planes, unmap buffers, tell decoder to deallocate buffer
       (reqbufs ioctl with counnt = 0), and finally call v4l2_close on the fd */
    delete ctx.dec;

    if (ctx.dst_dma_fd != -1) {
        ret = NvBufSurf::NvDestroy(ctx.dst_dma_fd);
        ctx.dst_dma_fd = -1;
        if (ret < 0) {
            cerr << "Error in BufferDestroy" << endl;
            error = 1;
        }
    }

    return NULL;
}
