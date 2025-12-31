#include "JetsonEnc.h"
#include <unistd.h>
JetsonEnc::JetsonEnc(int width, int height, int video_fps)
{
    image_width = width;
    image_height = height;
    fps = video_fps;
    m_abort = false;
    p_callback = NULL;
    pthread_mutex_init(&data_mutex, NULL);
    pthread_cond_init(&data_cond, NULL);
    data_list.clear();
    pthread_create(&job_tid, NULL, JetsonEnc::encode_proc, this);
}
void JetsonEnc::SetDecCallBack(JetsonEncListner *call_func)
{
    p_callback = call_func;
    return;
}
void JetsonEnc::UnSetDecCallBack()
{
    p_callback = NULL;
    return;
}
void JetsonEnc::AddFrame(unsigned char *data, int len)
{
    YUVData *node = new YUVData();
    // node->data=(unsigned char *)malloc(len);
    // memcpy(node->data,data,len);
    node->data = data;
    node->len = len;
    pthread_mutex_lock(&data_mutex);
#if 0
    // 编码丢帧
    if (data_list.size() > 5) {
        while (!data_list.empty()) {
            YUVData *tmp = data_list.front();
            data_list.pop_front();
            delete tmp;
        }
    }
#endif
    data_list.push_back(node);
    pthread_mutex_unlock(&data_mutex);
    pthread_cond_signal(&data_cond);
    return;
}
int JetsonEnc::GetQueueSize(){
    int size;
    pthread_mutex_lock(&data_mutex);
    size = data_list.size();
    pthread_mutex_unlock(&data_mutex);
    return size;
}
JetsonEnc::~JetsonEnc()
{
    m_abort = true;
    // ctx.got_error = true;
    // ctx.enc->abort();
    int ret = pthread_join(job_tid, NULL);
    if (ret != 0) {
        printf("pthread_join iob_tid error\n");
    }
    pthread_mutex_destroy(&data_mutex);
    pthread_cond_destroy(&data_cond);

    for (list<YUVData *>::iterator it = data_list.begin(); it != data_list.end(); ++it) {
        YUVData *node = *it;
        delete node;
    }
    data_list.clear();
    printf("~JetsonEnc()\n");
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

int JetsonEnc::read_video_frame_my(std::ifstream *stream, NvBuffer &buffer)
{
    uint32_t i, j;
    char *data;
    pthread_mutex_lock(&data_mutex);
    while (data_list.empty() && !m_abort) {
        struct timespec start_tm;
        struct timespec end_tm;
        int timeout_ms = 40;
        clock_gettime(CLOCK_REALTIME, &start_tm);
        end_tm = ns_to_tm(tm_to_ns(start_tm) + timeout_ms * 1000000);
        pthread_cond_timedwait(&data_cond, &data_mutex, &end_tm);
    }
    if (m_abort) {
        return -1;
    }
    YUVData *node = data_list.front();
    data_list.pop_front();
    pthread_mutex_unlock(&data_mutex);
    int pos = 0;
    for (i = 0; i < buffer.n_planes; i++) {
        NvBuffer::NvBufferPlane &plane = buffer.planes[i];
        std::streamsize bytes_to_read =
            plane.fmt.bytesperpixel * plane.fmt.width;
        data = (char *)plane.data;
        plane.bytesused = 0;
        for (j = 0; j < plane.fmt.height; j++) {
            memcpy(data, node->data + pos, bytes_to_read);
            pos += bytes_to_read;
            data += plane.fmt.stride;
        }
        plane.bytesused = plane.fmt.stride * plane.fmt.height;
    }
    delete node;
    return 0;
}

/**
 * Abort on error.
 *
 * @param ctx : Encoder context
 */
static void
abort(context_t_enc *ctx)
{
    ctx->got_error = true;
    ctx->enc->abort();
    return;
}
static const char *
get_pixfmt_string(uint32_t pixfmt)
{
    switch (pixfmt) {
    case V4L2_PIX_FMT_YUV420M:
        return "V4L2_PIX_FMT_YUV420M";
    case V4L2_PIX_FMT_NV12M:
        return "V4L2_PIX_FMT_NV12M";
    case V4L2_PIX_FMT_YUV444M:
        return "V4L2_PIX_FMT_YUV444M";
    case V4L2_PIX_FMT_NV24M:
        return "V4L2_PIX_FMT_NV24M";
    case V4L2_PIX_FMT_P010M:
        return "V4L2_PIX_FMT_P010M";
    case V4L2_PIX_FMT_NV24_10LE:
        return "V4L2_PIX_FMT_NV24_10LE";
    case V4L2_PIX_FMT_H264:
        return "V4L2_PIX_FMT_H264";
    case V4L2_PIX_FMT_H265:
        return "V4L2_PIX_FMT_H265";
    case V4L2_PIX_FMT_VP8:
        return "V4L2_PIX_FMT_VP8";
    case V4L2_PIX_FMT_VP9:
        return "V4L2_PIX_FMT_VP9";
    default:
        return "";
    }
    return "";
}
/**
 * Initialise CRC Rec and creates CRC Table based on the polynomial.
 *
 * @param CrcPolynomial : CRC Polynomial values
 */
static Crc *InitCrc(unsigned int CrcPolynomial)
{
    unsigned short int i;
    unsigned short int j;
    unsigned int tempcrc;
    Crc *phCrc;
    phCrc = (Crc *)malloc(sizeof(Crc));
    if (phCrc == NULL) {
        cerr << "Mem allocation failed for Init CRC" << endl;
        return NULL;
    }

    memset(phCrc, 0, sizeof(Crc));

    for (i = 0; i <= 255; i++) {
        tempcrc = i;
        for (j = 8; j > 0; j--) {
            if (tempcrc & 1) {
                tempcrc = (tempcrc >> 1) ^ CrcPolynomial;
            } else {
                tempcrc >>= 1;
            }
        }
        phCrc->CRCTable[i] = tempcrc;
    }

    phCrc->CrcValue = 0;
    return phCrc;
}

/**
 * Calculates CRC of data provided in by buffer.
 *
 * @param *phCrc : bitstream CRC
 * @param buffer : process buffer
 * @param count  : bytes used
 */
static void CalculateCrc(Crc *phCrc, unsigned char *buffer, uint32_t count)
{
    unsigned char *p;
    unsigned int temp1;
    unsigned int temp2;
    unsigned int crc = phCrc->CrcValue;
    unsigned int *CRCTable = phCrc->CRCTable;

    if (!count)
        return;

    p = (unsigned char *)buffer;
    while (count-- != 0) {
        temp1 = (crc >> 8) & 0x00FFFFFFL;
        temp2 = CRCTable[((unsigned int)crc ^ *p++) & 0xFF];
        crc = temp1 ^ temp2;
    }

    phCrc->CrcValue = crc;
}

/**
 * Closes CRC related handles.
 *
 * @param *phCrc : bitstream CRC
 */
static void CloseCrc(Crc **phCrc)
{
    if (*phCrc)
        free(*phCrc);
}

/**
 * Write encoded frame data.
 *
 * @param stream : output stream
 * @param buffer : output nvbuffer
 */
const char *enc_filename = "out.h264";
FILE *enc_fd = NULL;
static int
write_encoder_output_frame(ofstream *stream, NvBuffer *buffer, void *obj_arg)
{
#if 0
    stream->write((char *) buffer->planes[0].data, buffer->planes[0].bytesused);
#else
    /*
    if(enc_fd==NULL){
        enc_fd= fopen(enc_filename, "wb");
    }
    fwrite((char *) buffer->planes[0].data, 1, buffer->planes[0].bytesused, enc_fd);*/
    JetsonEnc *Obj = (JetsonEnc *)obj_arg;
    if (Obj->p_callback) {
        Obj->p_callback->OnJetsonEncData((unsigned char *)buffer->planes[0].data, buffer->planes[0].bytesused);
    }
#endif
    return 0;
}

/**
 * Encoder capture-plane deque buffer callback function.
 *
 * @param v4l2_buf      : v4l2 buffer
 * @param buffer        : NvBuffer
 * @param shared_buffer : shared NvBuffer
 * @param arg           : context pointer
 */
bool JetsonEnc::encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer,
                                                         NvBuffer *shared_buffer, void *arg)
{
    context_t_enc *ctx = (context_t_enc *)arg;
    NvVideoEncoder *enc = ctx->enc;
    pthread_setname_np(pthread_self(), "EncCapPlane");
    uint32_t frame_num = ctx->enc->capture_plane.getTotalDequeuedBuffers() - 1;
    uint32_t ReconRef_Y_CRC = 0;
    uint32_t ReconRef_U_CRC = 0;
    uint32_t ReconRef_V_CRC = 0;
    static uint32_t num_encoded_frames = 1;
    struct v4l2_event ev;
    int ret = 0;

    if (v4l2_buf == NULL) {
        cout << "Error while dequeing buffer from output plane" << endl;
        abort(ctx);
        return false;
    }

    if (ctx->b_use_enc_cmd) {
        if (v4l2_buf->flags & V4L2_BUF_FLAG_LAST) {
            memset(&ev, 0, sizeof(struct v4l2_event));
            ret = ctx->enc->dqEvent(ev, 1000);
            if (ret < 0)
                cout << "Error in dqEvent" << endl;
            if (ev.type == V4L2_EVENT_EOS)
                return false;
        }
    }

    /* Received EOS from encoder. Stop dqthread. */
    if (buffer->planes[0].bytesused == 0) {
        cout << "Got 0 size buffer in capture \n";
        return false;
    }

    /* Computing CRC with each frame */
    if (ctx->pBitStreamCrc)
        CalculateCrc(ctx->pBitStreamCrc, buffer->planes[0].data, buffer->planes[0].bytesused);

    if (!ctx->stats)
        write_encoder_output_frame(ctx->out_file, buffer, ctx->arg_obj);

    /* Accounting for the first frame as it is only sps+pps */
    if (ctx->gdr_out_frame_number != 0xFFFFFFFF)
        if ((ctx->enableGDR) && (ctx->GDR_out_file_path) && (num_encoded_frames >= ctx->gdr_out_frame_number + 1))
            write_encoder_output_frame(ctx->gdr_out_file, buffer, ctx->arg_obj);

    num_encoded_frames++;

    if (ctx->report_metadata) {
        v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
        if (ctx->enc->getMetadata(v4l2_buf->index, enc_metadata) == 0) {
            if (ctx->bReconCrc && enc_metadata.bValidReconCRC) {
                /* CRC for Recon frame */
                cout << "Frame: " << frame_num << endl;
                cout << "ReconFrame_Y_CRC " << enc_metadata.ReconFrame_Y_CRC << " ReconFrame_U_CRC " << enc_metadata.ReconFrame_U_CRC << " ReconFrame_V_CRC " << enc_metadata.ReconFrame_V_CRC << endl;

                if (!ctx->recon_Ref_file->eof()) {
                    string recon_ref_YUV_data[4];

                    parse_csv_recon_file(ctx->recon_Ref_file, recon_ref_YUV_data);

                    ReconRef_Y_CRC = stoul(recon_ref_YUV_data[0]);
                    ReconRef_U_CRC = stoul(recon_ref_YUV_data[1]);
                    ReconRef_V_CRC = stoul(recon_ref_YUV_data[2]);
                }

                if ((ReconRef_Y_CRC != enc_metadata.ReconFrame_Y_CRC) ||
                    (ReconRef_U_CRC != enc_metadata.ReconFrame_U_CRC) ||
                    (ReconRef_V_CRC != enc_metadata.ReconFrame_V_CRC)) {
                    cout << "Recon CRC FAIL" << endl;
                    cout << "ReconRef_Y_CRC " << ReconRef_Y_CRC << " ReconRef_U_CRC " << ReconRef_U_CRC << " ReconRef_V_CRC " << ReconRef_V_CRC << endl;
                    abort(ctx);
                    return false;
                }
                cout << "Recon CRC PASS for frame : " << frame_num << endl;
            } else if (ctx->externalRPS && enc_metadata.bRPSFeedback_status) {
                /* RPS Feedback */
                ctx->rps_par.nActiveRefFrames = enc_metadata.nActiveRefFrames;
                cout << "Frame: " << frame_num << endl;
                cout << "nCurrentRefFrameId " << enc_metadata.nCurrentRefFrameId << " nActiveRefFrames " << enc_metadata.nActiveRefFrames << endl;

                for (uint32_t i = 0; i < enc_metadata.nActiveRefFrames; i++) {
                    /* Update RPS List */
                    ctx->rps_par.rps_list[i].nFrameId = enc_metadata.RPSList[i].nFrameId;
                    ctx->rps_par.rps_list[i].bLTRefFrame = enc_metadata.RPSList[i].bLTRefFrame;

                    cout << "FrameId " << enc_metadata.RPSList[i].nFrameId << " IdrFrame " << (int)enc_metadata.RPSList[i].bIdrFrame << " LTRefFrame " << (int)enc_metadata.RPSList[i].bLTRefFrame << " PictureOrderCnt " << enc_metadata.RPSList[i].nPictureOrderCnt << " FrameNum " << enc_metadata.RPSList[i].nFrameNum << " LTFrameIdx " << enc_metadata.RPSList[i].nLTRFrameIdx << endl;
                }
            } else if (ctx->externalRCHints) {
                /* Rate Control Feedback */
                cout << "Frame: " << frame_num << endl;
                cout << "EncodedBits " << enc_metadata.EncodedFrameBits << " MinQP " << enc_metadata.FrameMinQP << " MaxQP " << enc_metadata.FrameMaxQP << endl;
            } else {
                cout << "Frame " << frame_num << ": isKeyFrame=" << (int)enc_metadata.KeyFrame << " AvgQP=" << enc_metadata.AvgQP << " MinQP=" << enc_metadata.FrameMinQP << " MaxQP=" << enc_metadata.FrameMaxQP << " EncodedBits=" << enc_metadata.EncodedFrameBits << endl;
            }
        }
    }
    if (ctx->dump_mv) {
        /* Get motion vector parameters of the frames from encoder */
        v4l2_ctrl_videoenc_outputbuf_metadata_MV enc_mv_metadata;
        if (ctx->enc->getMotionVectors(v4l2_buf->index, enc_mv_metadata) == 0) {
            uint32_t numMVs = enc_mv_metadata.bufSize / sizeof(MVInfo);
            MVInfo *pInfo = enc_mv_metadata.pMVInfo;

            cout << "Frame " << frame_num << ": Num MVs=" << numMVs << endl;

            for (uint32_t i = 0; i < numMVs; i++, pInfo++) {
                cout << i << ": mv_x=" << pInfo->mv_x << " mv_y=" << pInfo->mv_y << " weight=" << pInfo->weight << endl;
            }
        }
    }
    if (ctx->blocking_mode && ctx->RPS_threeLayerSvc) {
        sem_post(&ctx->rps_par.sema);
    }
    /* encoder qbuffer for capture plane */
    if (enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
        cerr << "Error while Qing buffer at capture plane" << endl;
        abort(ctx);
        return false;
    }

    return true;
}

/**
 * Parse runtime command stream.
 *
 * @param ctx   : Encoder context
 * @param id    : string id
 * @param value : Integer value
 */
static int
get_next_parsed_pair(context_t_enc *ctx, char *id, uint32_t *value)
{
    char charval;

    *ctx->runtime_params_str >> *id;
    if (ctx->runtime_params_str->eof()) {
        return -1;
    }

    charval = ctx->runtime_params_str->peek();
    if (!IS_DIGIT(charval)) {
        return -1;
    }

    *ctx->runtime_params_str >> *value;

    *ctx->runtime_params_str >> charval;
    if (ctx->runtime_params_str->eof()) {
        return 0;
    }

    return charval;
}

/**
 * Set Runtime Parameters.
 *
 * @param ctx : Encoder context
 */
static int
set_runtime_params(context_t_enc *ctx)
{
    char charval;
    uint32_t intval;
    int ret, next;

    cout << "Frame " << ctx->next_param_change_frame << ": Changing parameters" << endl;
    while (!ctx->runtime_params_str->eof()) {
        next = get_next_parsed_pair(ctx, &charval, &intval);
        TEST_PARSE_ERROR(next < 0, err);
        switch (charval) {
        case 'b':
            if (ctx->ratecontrol == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR &&
                ctx->peak_bitrate < intval) {
                uint32_t peak_bitrate = 1.2f * intval;
                cout << "Peak bitrate = " << peak_bitrate << endl;
                ret = ctx->enc->setPeakBitrate(peak_bitrate);
                if (ret < 0) {
                    cerr << "Could not set encoder peakbitrate" << endl;
                    goto err;
                }
            }
            cout << "Bitrate = " << intval << endl;
            ret = ctx->enc->setBitrate(intval);
            if (ret < 0) {
                cerr << "Could not set encoder bitrate" << endl;
                goto err;
            }
            break;
        case 'p':
            cout << "Peak bitrate = " << intval << endl;
            ret = ctx->enc->setPeakBitrate(intval);
            if (ret < 0) {
                cerr << "Could not set encoder peakbitrate" << endl;
                goto err;
            }
            break;
        case 'r': {
            int fps_num = intval;
            TEST_PARSE_ERROR(next != '/', err);

            ctx->runtime_params_str->seekg(-1, ios::cur);
            next = get_next_parsed_pair(ctx, &charval, &intval);
            TEST_PARSE_ERROR(next < 0, err);

            cout << "Framerate = " << fps_num << "/" << intval << endl;

            ret = ctx->enc->setFrameRate(fps_num, intval);
            if (ret < 0) {
                cerr << "Could not set framerate" << endl;
                goto err;
            }
            break;
        }
        case 'i':
            if (intval > 0) {
                ctx->enc->forceIDR();
                cout << "Forcing IDR" << endl;
            }
            break;
        default:
            TEST_PARSE_ERROR(true, err);
        }
        switch (next) {
        case 0:
            delete ctx->runtime_params_str;
            ctx->runtime_params_str = NULL;
            return 0;
        case '#':
            return 0;
        case ',':
            break;
        default:
            break;
        }
    }
    return 0;
err:
    cerr << "Skipping further runtime parameter changes" << endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

/**
 * Get the next runtime parameters change for frame.
 *
 * @param ctx : Encoder context
 */
static int
get_next_runtime_param_change_frame(context_t_enc *ctx)
{
    char charval;
    int ret;

    ret = get_next_parsed_pair(ctx, &charval, &ctx->next_param_change_frame);
    if (ret == 0) {
        return 0;
    }

    TEST_PARSE_ERROR((ret != ';' && ret != ',') || charval != 'f', err);

    return 0;

err:
    cerr << "Skipping further runtime parameter changes" << endl;
    delete ctx->runtime_params_str;
    ctx->runtime_params_str = NULL;
    return -1;
}

/**
 * Set encoder context defaults values.
 *
 * @param ctx : Encoder context
 */
static void
set_defaults(context_t_enc *ctx, void *arg)
{
    memset(ctx, 0, sizeof(context_t_enc));

    ctx->raw_pixfmt = V4L2_PIX_FMT_YUV420M;
    ctx->bitrate = 4 * 1024 * 1024;
    ctx->peak_bitrate = 0;
    ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
    ctx->ratecontrol = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
    ctx->iframe_interval = 30;
    ctx->externalRPS = false;
    ctx->enableGDR = false;
    ctx->enableROI = false;
    ctx->bnoIframe = false;
    ctx->bGapsInFrameNumAllowed = false;
    ctx->bReconCrc = false;
    ctx->enableLossless = false;
    ctx->nH264FrameNumBits = 0;
    ctx->nH265PocLsbBits = 0;
    ctx->idr_interval = 256;
    ctx->level = -1;
    ctx->fps_n = 30;
    ctx->fps_d = 1;
    ctx->gdr_start_frame_number = 0xffffffff;
    ctx->gdr_num_frames = 0xffffffff;
    ctx->gdr_out_frame_number = 0xffffffff;
    ctx->num_b_frames = (uint32_t)-1;
    ctx->nMinQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpI = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpP = (uint32_t)QP_RETAIN_VAL;
    ctx->nMinQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->nMaxQpB = (uint32_t)QP_RETAIN_VAL;
    ctx->use_gold_crc = false;
    ctx->pBitStreamCrc = NULL;
    ctx->externalRCHints = false;
    ctx->input_metadata = false;
    ctx->sMaxQp = 51;
    ctx->stats = false;
    ctx->stress_test = 1;
    ctx->output_memory_type = V4L2_MEMORY_DMABUF;
    ctx->cs = V4L2_COLORSPACE_SMPTE170M;
    ctx->copy_timestamp = false;
    ctx->sar_width = 0;
    ctx->sar_height = 0;
    ctx->start_ts = 0;
    ctx->max_perf = 0;
    ctx->blocking_mode = 1;
    ctx->startf = 0;
    ctx->endf = 0;
    ctx->num_output_buffers = 6;
    ctx->num_frames_to_encode = -1;
    ctx->poc_type = 0;
    ctx->chroma_format_idc = -1;
    ctx->bit_depth = 8;
    ctx->is_semiplanar = false;
    ctx->enable_initQP = false;
    ctx->IinitQP = 0;
    ctx->PinitQP = 0;
    ctx->BinitQP = 0;
    ctx->enable_ratecontrol = true;
    ctx->enable_av1tile = false;
    ctx->log2_num_av1rows = 0;
    ctx->log2_num_av1cols = 0;
    ctx->enable_av1ssimrdo = (uint8_t)-1;
    ctx->disable_av1cdfupdate = (uint8_t)-1;
    ctx->ppe_init_params.enable_ppe = false;
    ctx->ppe_init_params.wait_time_ms = -1;
    ctx->ppe_init_params.feature_flags = V4L2_PPE_FEATURE_NONE;
    ctx->ppe_init_params.enable_profiler = 0;
    ctx->ppe_init_params.width = 0;
    ctx->ppe_init_params.height = 0;
    ctx->ppe_init_params.taq_vic_downsampling = 0;
    ctx->ppe_init_params.taq_max_qp_delta = 5;
    ctx->arg_obj = arg;
}

/**
 * Populate the Region Of Interest(ROI) Parameters.
 *
 * @param stream          : input stream
 * @param VEnc_ROI_params : encoder ROI Parameters
 */
static void
populate_roi_Param(std::ifstream *stream, v4l2_enc_frame_ROI_params *VEnc_ROI_params)
{
    unsigned int ROIIndex = 0;

    if (!stream->eof()) {
        *stream >> VEnc_ROI_params->num_ROI_regions;
        while (ROIIndex < VEnc_ROI_params->num_ROI_regions) {
            if (ROIIndex == V4L2_MAX_ROI_REGIONS) {
                string skip_str;
                getline(*stream, skip_str);

                VEnc_ROI_params->num_ROI_regions = V4L2_MAX_ROI_REGIONS;

                cout << "Maximum of " << V4L2_MAX_ROI_REGIONS << "regions can be applied for a frame" << endl;
                break;
            }

            /* Populate Region Of Interest(ROI) coordinates and qpdelta value */
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].QPdelta;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.left;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.top;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.width;
            *stream >> VEnc_ROI_params->ROI_params[ROIIndex].ROIRect.height;
            ROIIndex++;
        }
    } else {
        cout << "EOF of ROI_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
}

/**
 * Populate the External Reference Picture Set(RPS) Parameters.
 *
 * @param stream                   : input stream
 * @param VEnc_ext_rps_ctrl_params : encoder RPS Parameters
 */
static void
populate_ext_rps_ctrl_Param(std::ifstream *stream, v4l2_enc_frame_ext_rps_ctrl_params *VEnc_ext_rps_ctrl_params)
{
    unsigned int RPSIndex = 0;
    unsigned int temp = 0;

    stream->peek();
restart:
    if (stream->eof()) {
        cout << "EOF of rps_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        /* Populate External Reference Picture Set(RPS) specific configuration */
        *stream >> VEnc_ext_rps_ctrl_params->nFrameId;
        if (stream->eof())
            goto restart;
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bRefFrame = ((temp) ? true : false);
        *stream >> temp;
        VEnc_ext_rps_ctrl_params->bLTRefFrame = ((temp) ? true : false);
        *stream >> VEnc_ext_rps_ctrl_params->nMaxRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nActiveRefFrames;
        *stream >> VEnc_ext_rps_ctrl_params->nCurrentRefFrameId;
        while (RPSIndex < VEnc_ext_rps_ctrl_params->nActiveRefFrames) {
            if (RPSIndex == V4L2_MAX_REF_FRAMES) {
                string skip_str;
                getline(*stream, skip_str);

                VEnc_ext_rps_ctrl_params->nActiveRefFrames = V4L2_MAX_REF_FRAMES;

                cout << "Maximum of " << V4L2_MAX_REF_FRAMES << "reference frames are valid" << endl;
                break;
            }

            *stream >> VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].nFrameId;
            *stream >> temp;
            VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].bLTRefFrame = ((temp) ? true : false);
            RPSIndex++;
        }
    }
}
static void
populate_ext_rps_threeLayerSvc_Param(context_t_enc *ctx, v4l2_enc_frame_ext_rps_ctrl_params *VEnc_ext_rps_ctrl_params)
{
    uint32_t RPSIndex = 0;
    uint32_t temporalCycle;
    uint32_t temporalID;
    int32_t previousTemporal0Id;
    bool bRefFrame;
    uint32_t nCurrentRefFrameId;
    uint32_t nFrameId = ctx->input_frames_queued_count;

    temporalCycle = (1 << (ctx->rps_par.m_numTemperalLayers - 1));
    temporalID = (nFrameId % 2);
    if (nFrameId % temporalCycle == 0) {
        previousTemporal0Id = nFrameId - temporalCycle;
        if (previousTemporal0Id < 0) {
            previousTemporal0Id = 0;
        }
    } else {
        previousTemporal0Id = (nFrameId >> 2) << 2;
    }
    if (ctx->rps_par.m_numTemperalLayers > 2) {
        temporalID += (nFrameId % temporalCycle) ? 1 : 0;
    }
    bRefFrame = (bool)(temporalID != ctx->rps_par.m_numTemperalLayers - 1);
    nCurrentRefFrameId = (temporalID == (ctx->rps_par.m_numTemperalLayers - 1) ? nFrameId - 1 : previousTemporal0Id);

    VEnc_ext_rps_ctrl_params->nFrameId = nFrameId;
    VEnc_ext_rps_ctrl_params->bRefFrame = bRefFrame;
    VEnc_ext_rps_ctrl_params->bLTRefFrame = false;
    VEnc_ext_rps_ctrl_params->nMaxRefFrames = ctx->num_reference_frames;
    VEnc_ext_rps_ctrl_params->nCurrentRefFrameId = nCurrentRefFrameId;
    VEnc_ext_rps_ctrl_params->nActiveRefFrames =
        ctx->rps_par.nActiveRefFrames;
    cout << "[Input]nFrameId:" << nFrameId << ",";
    cout << "bRefFrame:" << bRefFrame << ",";
    cout << "nMaxRefFrames:" << ctx->num_reference_frames << ",";
    cout << "nCurrentRefFrameId:" << nCurrentRefFrameId << endl;
    while (RPSIndex < VEnc_ext_rps_ctrl_params->nActiveRefFrames) {
        VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].nFrameId =
            ctx->rps_par.rps_list[RPSIndex].nFrameId;
        VEnc_ext_rps_ctrl_params->RPSList[RPSIndex].bLTRefFrame =
            ctx->rps_par.rps_list[RPSIndex].bLTRefFrame;
        RPSIndex++;
    }
}
/**
 * Setup output plane for DMABUF io-mode.
 *
 * @param ctx         : encoder context
 * @param num_buffers : request buffer count
 */
static int
setup_output_dmabuf(context_t_enc *ctx, uint32_t num_buffers)
{
    int ret = 0;
    NvBufSurf::NvCommonAllocateParams cParams;
    int fd;
    ret = ctx->enc->output_plane.reqbufs(V4L2_MEMORY_DMABUF, num_buffers);
    if (ret) {
        cerr << "reqbufs failed for output plane V4L2_MEMORY_DMABUF" << endl;
        return ret;
    }
    for (uint32_t i = 0; i < ctx->enc->output_plane.getNumBuffers(); i++) {
        cParams.width = ctx->width;
        cParams.height = ctx->height;
        cParams.layout = NVBUF_LAYOUT_PITCH;
        switch (ctx->cs) {
        case V4L2_COLORSPACE_REC709:
            cParams.colorFormat = ctx->enable_extended_colorformat ? NVBUF_COLOR_FORMAT_YUV420_709_ER : NVBUF_COLOR_FORMAT_YUV420_709;
            break;
        case V4L2_COLORSPACE_SMPTE170M:
        default:
            cParams.colorFormat = ctx->enable_extended_colorformat ? NVBUF_COLOR_FORMAT_YUV420_ER : NVBUF_COLOR_FORMAT_YUV420;
        }
        if (ctx->is_semiplanar) {
            cParams.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        }
        if (ctx->encoder_pixfmt == V4L2_PIX_FMT_H264) {
            if (ctx->enableLossless) {
                if (ctx->is_semiplanar)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24;
                else

                    cParams.colorFormat = NVBUF_COLOR_FORMAT_YUV444;
            }
        } else if (ctx->encoder_pixfmt == V4L2_PIX_FMT_H265) {
            if (ctx->chroma_format_idc == 3) {
                if (ctx->is_semiplanar)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24;
                else
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_YUV444;

                if (ctx->bit_depth == 10)
                    cParams.colorFormat = NVBUF_COLOR_FORMAT_NV24_10LE;
            }
            if (ctx->profile == V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10 && (ctx->bit_depth == 10)) {
                cParams.colorFormat = NVBUF_COLOR_FORMAT_NV12_10LE;
            }
        }
        cParams.memtag = NvBufSurfaceTag_VIDEO_ENC;
        cParams.memType = NVBUF_MEM_SURFACE_ARRAY;
        /* Create output plane fd for DMABUF io-mode */
        ret = NvBufSurf::NvAllocate(&cParams, 1, &fd);
        if (ret < 0) {
            cerr << "Failed to create NvBuffer" << endl;
            return ret;
        }
        ctx->output_plane_fd[i] = fd;
    }
    return ret;
}

/**
 * Populate the External Rate Control(RC) Parameters.
 *
 * @param stream                    : input stream
 * @param VEnc_ext_rate_ctrl_params : encoder external RC Parameters
 */
static void
populate_ext_rate_ctrl_Param(std::ifstream *stream, v4l2_enc_frame_ext_rate_ctrl_params *VEnc_ext_rate_ctrl_params)
{
    stream->peek();
restart:
    if (stream->eof()) {
        cout << "EOF of hints_param_file & rewind" << endl;
        stream->clear();
        stream->seekg(0);
    }
    if (!stream->eof()) {
        /* Populate External Rate Control specific configuration */
        *stream >> VEnc_ext_rate_ctrl_params->nTargetFrameBits;
        if (stream->eof())
            goto restart;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameQP;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMinQp;
        *stream >> VEnc_ext_rate_ctrl_params->nFrameMaxQp;
        *stream >> VEnc_ext_rate_ctrl_params->nMaxQPDeviation;
    }
}

/**
 * Populate the Gradual Decoder Refresh(GDR) Parameters.
 *
 * @param stream          : input stream
 * @param start_frame_num : start frame number
 * @param gdr_num_frames  : GDR frame number
 */
static void
populate_gdr_Param(std::ifstream *stream, uint32_t *start_frame_num, uint32_t *gdr_num_frames)
{
    if (stream->eof()) {
        *start_frame_num = 0xFFFFFFFF;
        cout << "GDR param EoF reached \n";
    }
    if (!stream->eof()) {
        *stream >> *start_frame_num;
        *stream >> *gdr_num_frames;
    }
}

/**
 * Encoder polling thread loop function.
 *
 * @param args : void arguments
 */
static void *encoder_pollthread_fcn(void *arg)
{

    context_t_enc *ctx = (context_t_enc *)arg;
    v4l2_ctrl_video_device_poll devicepoll;

    cout << "Starting Device Poll Thread " << endl;

    memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

    /* wait here until signalled to issue the Poll call.
       Check if the abort status is set , if so exit
       Else issue the Poll on the encoder and block.
       When the Poll returns, signal the encoder thread to continue. */

    while (!ctx->got_error && !ctx->enc->isInError()) {
        sem_wait(&ctx->pollthread_sema);

        if (ctx->got_eos) {
            cout << "Got eos, exiting poll thread \n";
            return NULL;
        }

        devicepoll.req_events = POLLIN | POLLOUT | POLLERR | POLLPRI;

        /* This call shall wait in the v4l2 encoder library */
        ctx->enc->DevicePoll(&devicepoll);

        /* Can check the devicepoll.resp_events bitmask to see which events are set. */
        sem_post(&ctx->encoderthread_sema);
    }
    cout << "encoder_pollthread_fcn exited\n";
    return NULL;
}

/**
 * Encode processing function for non-blocking mode.
 *
 * @param ctx : Encoder context
 * @param eos : end of stream
 */
int JetsonEnc::encoder_proc_nonblocking(context_t_enc &ctx, bool eos)
{
    /* NOTE: In non-blocking mode, we will have this function do below things:
           1) Issue signal to PollThread so it starts Poll and wait until signalled.
           2) After we are signalled, it means there is something to dequeue,
              either output plane or capture plane or there's an event.
           3) Try dequeuing from all three and then act appropriately.
           4) After enqueuing go back to the same loop. */

    /* Since all the output plane buffers have been queued, we first need to
       dequeue a buffer from output plane before we can read new data into it
       and queue it again. */
    int ret = 0;
    JetsonEnc *self = (JetsonEnc*)ctx.arg_obj;
    while (!ctx.got_error && !ctx.enc->isInError()) {
        /* Call SetPollInterrupt */
        ctx.enc->SetPollInterrupt();

        /* Since buffers have been queued, issue a post to start polling and
           then wait here */
        sem_post(&ctx.pollthread_sema);
        sem_wait(&ctx.encoderthread_sema);

        /* Already end of file, no more queue-dequeue for output plane */
        if (eos)
            goto check_capture_buffers;

        /* Check if can dequeue from output plane */
        while (1) {

            struct v4l2_buffer v4l2_output_buf;
            struct v4l2_plane output_planes[MAX_PLANES];
            NvBuffer *outplane_buffer = NULL;

            memset(&v4l2_output_buf, 0, sizeof(v4l2_output_buf));
            memset(output_planes, 0, sizeof(output_planes));
            v4l2_output_buf.m.planes = output_planes;

            /* Dequeue from output plane, fill the frame and enqueue it back again.
               NOTE: This could be moved out to a different thread as an optimization. */
            ret = ctx.enc->output_plane.dqBuffer(v4l2_output_buf, &outplane_buffer, NULL, 10);
            if (ret < 0) {
                if (errno == EAGAIN) {
                    goto check_capture_buffers;
                }
                cerr << "ERROR while DQing buffer at output plane" << endl;
                abort(&ctx);
                return -1;
            }

            /* Get the parsed encoder runtime parameters */
            if (ctx.runtime_params_str &&
                (ctx.enc->output_plane.getTotalQueuedBuffers() ==
                 ctx.next_param_change_frame)) {
                set_runtime_params(&ctx);
                if (ctx.runtime_params_str)
                    get_next_runtime_param_change_frame(&ctx);
            }

            /* Read yuv frame data from input file */
            if (read_video_frame_my(ctx.in_file, *outplane_buffer) < 0) // || ctx.num_frames_to_encode == 0
            {
                cerr << "Could not read complete frame from input file" << endl;
                v4l2_output_buf.m.planes[0].bytesused = 0;
                if (ctx.b_use_enc_cmd) {
                    ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                    eos = true;
                    break;
                } else {
                    eos = true;
                    v4l2_output_buf.m.planes[0].m.userptr = 0;
                    v4l2_output_buf.m.planes[0].bytesused = 0;
                    v4l2_output_buf.m.planes[1].bytesused = 0;
                    v4l2_output_buf.m.planes[2].bytesused = 0;
                }
            }

            /* Encoder supported input metadata specific configurations */
            if (ctx.input_metadata) {
                v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
                v4l2_enc_frame_ROI_params VEnc_ROI_params;
                v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
                v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
                v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
                v4l2_enc_gdr_params VEnc_gdr_params;
                VEnc_imeta_param.flag = 0;

                if (ctx.ROI_Param_file_path) {
                    if (ctx.enableROI) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;
                        /* Update Region of Intrest parameters from ROI params file */
                        populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                    }
                }

                if (ctx.bReconCrc) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                    VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                    VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                    VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                    VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                    /* Update reconstructed CRC Parameters */
                    VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
                }

                if (ctx.RPS_Param_file_path) {
                    if (ctx.externalRPS) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                        /* Update external reference picture set parameters from RPS params file */
                        populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                    }
                }

                if (ctx.GDR_Param_file_path) {
                    if (ctx.enableGDR) {
                        /* Update GDR parameters from GDR params file */
                        if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                            populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                               &ctx.gdr_num_frames);
                        if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number) {
                            ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                            VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                            VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                            VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                        }
                    }
                }

                if (ctx.hints_Param_file_path) {
                    if (ctx.externalRCHints) {
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                        /* Update external rate control parameters from hints params file */
                        populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);
                    }
                }

                if (VEnc_imeta_param.flag) {
                    /* Set encoder input metadatas */
                    ctx.enc->SetInputMetaParams(v4l2_output_buf.index, VEnc_imeta_param);
                    v4l2_output_buf.reserved2 = v4l2_output_buf.index;
                }
            }

            if (ctx.copy_timestamp) {
                /* Set user provided timestamp when copy timestamp is enabled */
                v4l2_output_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
                ctx.timestamp += ctx.timestampincr;
                v4l2_output_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
                v4l2_output_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
            }

            if (!ctx.num_frames_to_encode) {
                outplane_buffer->planes[0].bytesused = outplane_buffer->planes[1].bytesused = outplane_buffer->planes[2].bytesused = 0;
            }

            if (ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP) {
                for (uint32_t j = 0; j < outplane_buffer->n_planes; j++) {
                    NvBufSurface *nvbuf_surf = 0;
                    ret = NvBufSurfaceFromFd(outplane_buffer->planes[j].fd, (void **)(&nvbuf_surf));
                    if (ret < 0) {
                        cerr << "Error while NvBufSurfaceFromFd" << endl;
                        abort(&ctx);
                        return -1;
                    }

                    ret = NvBufSurfaceSyncForDevice(nvbuf_surf, 0, j);
                    if (ret < 0) {
                        cerr << "Error while NvBufSurfaceSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                        abort(&ctx);
                        return -1;
                    }
                }
            }

            if (ctx.output_memory_type == V4L2_MEMORY_DMABUF) {
                for (uint32_t j = 0; j < outplane_buffer->n_planes; j++) {
                    v4l2_output_buf.m.planes[j].bytesused = outplane_buffer->planes[j].bytesused;
                }
            }
            /* encoder qbuffer for output plane */
            ret = ctx.enc->output_plane.qBuffer(v4l2_output_buf, NULL);
            if (ret < 0) {
                cerr << "Error while queueing buffer at output plane" << endl;
                abort(&ctx);
                return -1;
            }
            if (ctx.num_frames_to_encode > 0) {
                ctx.num_frames_to_encode--;
            }
            ctx.input_frames_queued_count++;
            if (v4l2_output_buf.m.planes[0].bytesused == 0) {
                cerr << "File read complete." << endl;
                eos = true;
                goto check_capture_buffers;
            }
            goto check_capture_buffers;
        }

    check_capture_buffers:
        while (1) {
            struct v4l2_buffer v4l2_capture_buf;
            struct v4l2_plane capture_planes[MAX_PLANES];
            NvBuffer *capplane_buffer = NULL;
            bool capture_dq_continue = true;

            memset(&v4l2_capture_buf, 0, sizeof(v4l2_capture_buf));
            memset(capture_planes, 0, sizeof(capture_planes));
            v4l2_capture_buf.m.planes = capture_planes;
            v4l2_capture_buf.length = 1;

            /* Dequeue from output plane, fill the frame and enqueue it back again.
               NOTE: This could be moved out to a different thread as an optimization. */
            ret = ctx.enc->capture_plane.dqBuffer(v4l2_capture_buf, &capplane_buffer, NULL, 10);
            if (ret < 0) {
                if (errno == EAGAIN)
                    break;
                cerr << "ERROR while DQing buffer at capture plane" << endl;
                abort(&ctx);
                return -1;
            }

            /* Invoke encoder capture-plane deque buffer callback */
            capture_dq_continue = encoder_capture_plane_dq_callback(&v4l2_capture_buf, capplane_buffer, NULL,
                                                                    &ctx);
            if (!capture_dq_continue) {
                cout << "Capture plane dequeued 0 size buffer " << endl;
                ctx.got_eos = true;
                sem_post(&ctx.pollthread_sema);
                return 0;
            }
        }
    }
    return 0;
}

/**
 * Encode processing function for blocking mode.
 *
 * @param ctx : Encoder context
 * @param eos : end of stream
 */
int JetsonEnc::encoder_proc_blocking(context_t_enc &ctx, bool eos)
{
#if 1
    int ret = 0;
    /* Keep reading input till EOS is reached */
    while (!ctx.got_error && !ctx.enc->isInError() && !eos) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer;

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;

        /* dequeue buffer from encoder output plane */
        if (ctx.enc->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10) < 0) {
            cerr << "ERROR while DQing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }

        /* Get the parsed encoder runtime parameters */
        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
             ctx.next_param_change_frame)) {
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }

        /* Read yuv frame data from input file */
        if (read_video_frame_my(ctx.in_file, *buffer) < 0 || ctx.num_frames_to_encode == 0) {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
            if (ctx.b_use_enc_cmd) {
                ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                eos = true;
                ctx.got_eos = true;
                break;
            } else {
                eos = true;
                ctx.got_eos = true;
                v4l2_buf.m.planes[0].m.userptr = 0;
                v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
            }
        }

        /* Encoder supported input metadata specific configurations */
        if (ctx.input_metadata) {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            v4l2_enc_gdr_params VEnc_gdr_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path) {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;
                    /* Update Region of Intrest parameters from ROI params file */
                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc) {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                /* Update reconstructed CRC Parameters */
                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.externalRPS) {
                if (ctx.RPS_threeLayerSvc) {
                    if (ctx.input_frames_queued_count > 0) {
                        sem_wait(&ctx.rps_par.sema);
                    }
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                    populate_ext_rps_threeLayerSvc_Param(&ctx, VEnc_imeta_param.VideoEncRPSParams);
                } else if (ctx.RPS_Param_file_path) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                    /* Update external reference picture set parameters from RPS params file */
                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.GDR_Param_file_path) {
                if (ctx.enableGDR) {
                    /* Update GDR parameters from GDR params file */
                    if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                        populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                           &ctx.gdr_num_frames);
                    if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number) {
                        ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                        VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                    }
                }
            }

            if (ctx.hints_Param_file_path) {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    /* Update external rate control parameters from hints params file */
                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);
                }
            }

            if (VEnc_imeta_param.flag) {
                /* Set encoder input metadatas */
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        if (ctx.copy_timestamp) {
            /* Set user provided timestamp when copy timestamp is enabled */
            v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
            ctx.timestamp += ctx.timestampincr;
            v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
            v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if (ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP) {
            for (uint32_t j = 0; j < buffer->n_planes; j++) {
                NvBufSurface *nvbuf_surf = 0;
                ret = NvBufSurfaceFromFd(buffer->planes[j].fd, (void **)(&nvbuf_surf));
                if (ret < 0) {
                    cerr << "Error while NvBufSurfaceFromFd" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
                ret = NvBufSurfaceSyncForDevice(nvbuf_surf, 0, j);
                if (ret < 0) {
                    cerr << "Error while NvBufSurfaceSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
            }
        }
        if (!ctx.num_frames_to_encode) {
            buffer->planes[0].bytesused = buffer->planes[1].bytesused = buffer->planes[2].bytesused = 0;
        }

        if (ctx.output_memory_type == V4L2_MEMORY_DMABUF) {
            for (uint32_t j = 0; j < buffer->n_planes; j++) {
                v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
            }
        }
        /* encoder qbuffer for output plane */
        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
        if (ctx.num_frames_to_encode > 0) {
            ctx.num_frames_to_encode--;
        }
        ctx.input_frames_queued_count++;
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            cerr << "File read complete." << endl;
            eos = true;
            ctx.got_eos = true;
            return 0;
        }
    }
cleanup:
    return -1;
#endif
}
void *JetsonEnc::encode_proc(void *arg)
{
    JetsonEnc *self = (JetsonEnc *)arg;
    int ret = 0;
    int error = 0;
    bool eos = false;
    context_t_enc &ctx = self->ctx;
    /* Set default values for encoder context members. */
    set_defaults(&ctx, (void *)self);

    /* Parse application command line options. */
    /*ret = parse_csv_args(&ctx, argc, argv);
    TEST_ERROR(ret < 0, "Error parsing commandline arguments", cleanup);*/

    /* set your enc arg */
    ctx.encoder_pixfmt = V4L2_PIX_FMT_H264;
    ctx.max_perf = 1;
    // ctx.bGapsInFrameNumAllowed = true;
    ctx.fps_n = self->fps;
    ctx.fps_d = 1;
    ctx.peak_bitrate = 1000000;
    ctx.iframe_interval = self->fps * 2;
    ctx.idr_interval = self->fps * 2;
    ctx.insert_sps_pps_at_idr = true;
    ctx.insert_vui = true;
    // ctx.insert_aud = true;
    ctx.hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
    // ctx.alliframes = true;
    ctx.profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
    ctx.num_b_frames = 0;
    // ctx.input_metadata = true;
    ctx.blocking_mode = 0;
    ctx.width = self->image_width;
    ctx.height = self->image_height;

    /*not use*/
    const char *ptr = "../352_288.yuv";
    ctx.in_file_path = strdup(ptr);
    const char *tmp = "enc_out.h264";
    ctx.out_file_path = strdup(tmp);

    /* Set thread name for encoder Output Plane thread. */
    pthread_setname_np(pthread_self(), "EncOutPlane");

    /* Get the parsed encoder runtime parameters */
    if (ctx.runtime_params_str) {
        get_next_runtime_param_change_frame(&ctx);
    }

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {
        TEST_ERROR(ctx.width < 144 || ctx.height < 144, "Height/Width should be"
                                                        " > 144 for H.265",
                   cleanup);
    }

    if (ctx.endf) {
        TEST_ERROR(ctx.startf > ctx.endf, "End frame should be greater than start frame", cleanup);
        ctx.num_frames_to_encode = ctx.endf - ctx.startf + 1;
    }

    if (ctx.use_gold_crc) {
        /* CRC specific initializetion if gold_crc flag is set */
        ctx.pBitStreamCrc = InitCrc(CRC32_POLYNOMIAL);
        TEST_ERROR(!ctx.pBitStreamCrc, "InitCrc failed", cleanup);
    }

    /* Open input file  for raw yuv */
    // ctx.in_file = new ifstream(ctx.in_file_path);
    // TEST_ERROR(!ctx.in_file->is_open(), "Could not open input file", cleanup);

    if (!ctx.stats) {
        /* Open output file for encoded bitstream */
        // ctx.out_file = new ofstream(ctx.out_file_path);
        // TEST_ERROR(!ctx.out_file->is_open(), "Could not open output file", cleanup);
    }

    if (ctx.ROI_Param_file_path) {
        /* Open Region of Intreset(ROI) parameter file when ROI feature enabled */
        ctx.roi_Param_file = new ifstream(ctx.ROI_Param_file_path);
        TEST_ERROR(!ctx.roi_Param_file->is_open(), "Could not open roi param file", cleanup);
    }

    if (ctx.Recon_Ref_file_path) {
        /* Open Reconstructed CRC reference file when ReconCRC feature enabled */
        ctx.recon_Ref_file = new ifstream(ctx.Recon_Ref_file_path);
        TEST_ERROR(!ctx.recon_Ref_file->is_open(), "Could not open recon crc reference file", cleanup);
    }

    if (ctx.RPS_Param_file_path) {
        /* Open Reference Picture set(RPS) specififc reference file when Dynamic RPS feature enabled */
        ctx.rps_Param_file = new ifstream(ctx.RPS_Param_file_path);
        TEST_ERROR(!ctx.rps_Param_file->is_open(), "Could not open rps param file", cleanup);
    }

    if (ctx.GDR_Param_file_path) {
        /* Open Gradual Decoder Refresh(GDR) parameters reference file when GDR feature enabled */
        ctx.gdr_Param_file = new ifstream(ctx.GDR_Param_file_path);
        TEST_ERROR(!ctx.gdr_Param_file->is_open(), "Could not open GDR param file", cleanup);
    }

    if (ctx.GDR_out_file_path) {
        /* Open Gradual Decoder Refresh(GDR) output parameters reference file when GDR feature enabled */
        ctx.gdr_out_file = new ofstream(ctx.GDR_out_file_path);
        TEST_ERROR(!ctx.gdr_out_file->is_open(), "Could not open GDR Out file", cleanup);
    }

    if (ctx.hints_Param_file_path) {
        /* Open external hints parameters file for when external rate control feature enabled */
        ctx.hints_Param_file = new ifstream(ctx.hints_Param_file_path);
        TEST_ERROR(!ctx.hints_Param_file->is_open(), "Could not open hints param file", cleanup);
    }

    /* Create NvVideoEncoder object for blocking or non-blocking I/O mode. */
    if (ctx.blocking_mode) {
        cout << "Creating Encoder in blocking mode \n";
        ctx.enc = NvVideoEncoder::createVideoEncoder("enc0");
    } else {
        cout << "Creating Encoder in non-blocking mode \n";
        ctx.enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK);
    }
    TEST_ERROR(!ctx.enc, "Could not create encoder", cleanup);

    if (ctx.stats) {
        ctx.enc->enableProfiling();
    }
    if (log_level >= LOG_LEVEL_DEBUG)
        cout << "Encode pixel format :" << get_pixfmt_string(ctx.encoder_pixfmt) << endl;

    /* Set encoder capture plane format.
       NOTE: It is necessary that Capture Plane format be set before Output Plane
       format. It is necessary to set width and height on the capture plane as well */
    ret =
        ctx.enc->setCapturePlaneFormat(ctx.encoder_pixfmt, ctx.width,
                                       ctx.height, 2 * 1024 * 1024);
    TEST_ERROR(ret < 0, "Could not set capture plane format", cleanup);

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {
        switch (ctx.profile) {
        case V4L2_MPEG_VIDEO_H265_PROFILE_MAIN10: {
            ctx.raw_pixfmt = V4L2_PIX_FMT_P010M;
            ctx.is_semiplanar = true; /* To keep previous execution commands working */
            ctx.bit_depth = 10;
            break;
        }
        case V4L2_MPEG_VIDEO_H265_PROFILE_MAIN: {
            if (ctx.is_semiplanar)
                ctx.raw_pixfmt = V4L2_PIX_FMT_NV12M;
            else
                ctx.raw_pixfmt = V4L2_PIX_FMT_YUV420M;
            if (ctx.chroma_format_idc == 3) {
                if (ctx.bit_depth == 10 && ctx.is_semiplanar)
                    ctx.raw_pixfmt = V4L2_PIX_FMT_NV24_10LE;
                if (ctx.bit_depth == 8) {
                    if (ctx.is_semiplanar)
                        ctx.raw_pixfmt = V4L2_PIX_FMT_NV24M;
                    else
                        ctx.raw_pixfmt = V4L2_PIX_FMT_YUV444M;
                }
            }
        } break;
        default:
            ctx.raw_pixfmt = V4L2_PIX_FMT_YUV420M;
        }
    }
    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
        if (ctx.enableLossless &&
            ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE) {
            if (ctx.is_semiplanar)
                ctx.raw_pixfmt = V4L2_PIX_FMT_NV24M;
            else
                ctx.raw_pixfmt = V4L2_PIX_FMT_YUV444M;
        } else if ((ctx.enableLossless &&
                    ctx.profile != V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE) ||
                   (!ctx.enableLossless && ctx.profile == V4L2_MPEG_VIDEO_H264_PROFILE_HIGH_444_PREDICTIVE)) {
            cerr << "Lossless encoding is supported only for high444 profile\n";
            error = 1;
            goto cleanup;
        } else {
            if (ctx.is_semiplanar)
                ctx.raw_pixfmt = V4L2_PIX_FMT_NV12M;
            else
                ctx.raw_pixfmt = V4L2_PIX_FMT_YUV420M;
        }
    }

    if (log_level >= LOG_LEVEL_DEBUG)
        cout << "Raw pixel format :" << get_pixfmt_string(ctx.raw_pixfmt) << endl;

    /* Set encoder output plane format */
    ret =

        ctx.enc->setOutputPlaneFormat(ctx.raw_pixfmt, ctx.width,
                                      ctx.height);
    TEST_ERROR(ret < 0, "Could not set output plane format", cleanup);

    if (ctx.num_frames_to_encode) {
        ret = ctx.enc->setFramesToEncode(ctx.num_frames_to_encode);
        TEST_ERROR(ret < 0, "Could not set frames to encode", cleanup);
    }

    ret = ctx.enc->setBitrate(ctx.bitrate);
    TEST_ERROR(ret < 0, "Could not set encoder bitrate", cleanup);

    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
        /* Set encoder profile for H264 format */
        ret = ctx.enc->setProfile(ctx.profile);
        TEST_ERROR(ret < 0, "Could not set encoder profile", cleanup);

        if (ctx.level == (uint32_t)-1) {
            ctx.level = (uint32_t)V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
        }

        ret = ctx.enc->setLevel(ctx.level);
        TEST_ERROR(ret < 0, "Could not set encoder level", cleanup);
    } else if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {

        ret = ctx.enc->setProfile(ctx.profile);
        TEST_ERROR(ret < 0, "Could not set encoder profile", cleanup);

        if (ctx.level != (uint32_t)-1) {

            ret = ctx.enc->setLevel(ctx.level);
            TEST_ERROR(ret < 0, "Could not set encoder level", cleanup);
        }
        if (ctx.chroma_format_idc != (uint8_t)-1) {
            ret = ctx.enc->setChromaFactorIDC(ctx.chroma_format_idc);
            TEST_ERROR(ret < 0, "Could not set chroma_format_idc", cleanup);
        }
    }
    if (ctx.enable_initQP) {
        ret = ctx.enc->setInitQP(ctx.IinitQP, ctx.PinitQP, ctx.BinitQP);
        TEST_ERROR(ret < 0, "Could not set encoder init QP", cleanup);
    }

    if (ctx.enableLossless) {
        ret = ctx.enc->setLossless(ctx.enableLossless);
        TEST_ERROR(ret < 0, "Could not set lossless encoding", cleanup);
    } else if (!ctx.enable_ratecontrol) {
        /* Set constant QP configuration by disabling rate control */
        ret = ctx.enc->setConstantQp(ctx.enable_ratecontrol);
        TEST_ERROR(ret < 0, "Could not set encoder constant QP", cleanup);
    } else {
        /* Set rate control mode for encoder */
        ret = ctx.enc->setRateControlMode(ctx.ratecontrol);
        TEST_ERROR(ret < 0, "Could not set encoder rate control mode", cleanup);
        if (ctx.ratecontrol == V4L2_MPEG_VIDEO_BITRATE_MODE_VBR) {
            uint32_t peak_bitrate;
            if (ctx.peak_bitrate < ctx.bitrate)
                peak_bitrate = 1.2f * ctx.bitrate;
            else
                peak_bitrate = ctx.peak_bitrate;
            /* Set peak bitrate value for variable bitrate mode for encoder */
            ret = ctx.enc->setPeakBitrate(peak_bitrate);
            TEST_ERROR(ret < 0, "Could not set encoder peak bitrate", cleanup);
        }
    }

    if (ctx.poc_type) {
        ret = ctx.enc->setPocType(ctx.poc_type);
        TEST_ERROR(ret < 0, "Could not set Picture Order Count value", cleanup);
    }

    /* Set IDR frame interval for encoder */
    ret = ctx.enc->setIDRInterval(ctx.idr_interval);
    TEST_ERROR(ret < 0, "Could not set encoder IDR interval", cleanup);

    /* Set I frame interval for encoder */
    ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
    TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);

    /* Set framerate for encoder */
    ret = ctx.enc->setFrameRate(ctx.fps_n, ctx.fps_d);
    TEST_ERROR(ret < 0, "Could not set framerate", cleanup);

    if (ctx.temporal_tradeoff_level) {
        /* Set temporal tradeoff level value for encoder */
        ret = ctx.enc->setTemporalTradeoff(ctx.temporal_tradeoff_level);
        TEST_ERROR(ret < 0, "Could not set temporal tradeoff level", cleanup);
    }

    if (ctx.slice_length) {
        /* Set slice length value for encoder */
        ret = ctx.enc->setSliceLength(ctx.slice_length_type,
                                      ctx.slice_length);
        TEST_ERROR(ret < 0, "Could not set slice length params", cleanup);
    }

    if (ctx.enable_slice_level_encode) {
        /* Enable slice level encode for encoder */
        ret = ctx.enc->setSliceLevelEncode(true);
        TEST_ERROR(ret < 0, "Could not set slice level encode", cleanup);
    }

    if (ctx.hw_preset_type) {
        /* Set hardware preset value for encoder */
        ret = ctx.enc->setHWPresetType(ctx.hw_preset_type);
        TEST_ERROR(ret < 0, "Could not set encoder HW Preset Type", cleanup);
    }

    if (ctx.virtual_buffer_size) {
        /* Set virtual buffer size value for encoder */
        ret = ctx.enc->setVirtualBufferSize(ctx.virtual_buffer_size);
        TEST_ERROR(ret < 0, "Could not set virtual buffer size", cleanup);
    }

    if (ctx.slice_intrarefresh_interval) {
        /* Set slice intra refresh interval value for encoder */
        ret = ctx.enc->setSliceIntrarefresh(ctx.slice_intrarefresh_interval);
        TEST_ERROR(ret < 0, "Could not set slice intrarefresh interval", cleanup);
    }

    if (ctx.insert_sps_pps_at_idr) {
        /* Enable insert of SPSPPS at IDR frames */
        ret = ctx.enc->setInsertSpsPpsAtIdrEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertSPSPPSAtIDR", cleanup);
    }

    if (ctx.disable_cabac) {
        /* Disable CABAC entropy encoding */
        ret = ctx.enc->setCABAC(false);
        TEST_ERROR(ret < 0, "Could not set disable CABAC", cleanup);
    }
    if (ctx.sar_width) {
        /* Set SAR width */
        ret = ctx.enc->setSampleAspectRatioWidth(ctx.sar_width);
        TEST_ERROR(ret < 0, "Could not set Sample Aspect Ratio width", cleanup);
    }

    if (ctx.sar_height) {
        /* Set SAR width */
        ret = ctx.enc->setSampleAspectRatioHeight(ctx.sar_height);
        TEST_ERROR(ret < 0, "Could not set Sample Aspect Ratio height", cleanup);
    }
    if (ctx.insert_vui) {
        /* Enable insert of VUI parameters */
        ret = ctx.enc->setInsertVuiEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertVUI", cleanup);
    }

    if (ctx.enable_extended_colorformat) {
        /* Enable extnded colorformat for encoder */
        ret = ctx.enc->setExtendedColorFormat(true);
        TEST_ERROR(ret < 0, "Could not set extended color format", cleanup);
    }

    if (ctx.insert_aud) {
        /* Enable insert of AUD parameters */
        ret = ctx.enc->setInsertAudEnabled(true);
        TEST_ERROR(ret < 0, "Could not set insertAUD", cleanup);
    }

    if (ctx.alliframes) {
        /* Enable all I-frame encode */
        ret = ctx.enc->setAlliFramesEncode(true);
        TEST_ERROR(ret < 0, "Could not set Alliframes encoding", cleanup);
    }

    if (ctx.num_b_frames != (uint32_t)-1) {
        /* Set number of B-frames to to be used by encoder */
        ret = ctx.enc->setNumBFrames(ctx.num_b_frames);
        TEST_ERROR(ret < 0, "Could not set number of B Frames", cleanup);
    }

    if ((ctx.nMinQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpI != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpP != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMinQpB != (uint32_t)QP_RETAIN_VAL) ||
        (ctx.nMaxQpB != (uint32_t)QP_RETAIN_VAL)) {
        /* Set Min & Max qp range values for I/P/B-frames to be used by encoder */
        ret = ctx.enc->setQpRange(ctx.nMinQpI, ctx.nMaxQpI, ctx.nMinQpP,
                                  ctx.nMaxQpP, ctx.nMinQpB, ctx.nMaxQpB);
        TEST_ERROR(ret < 0, "Could not set quantization parameters", cleanup);
    }

    if (ctx.max_perf) {
        /* Enable maximum performance mode by disabling internal DFS logic.
           NOTE: This enables encoder to run at max clocks */
        ret = ctx.enc->setMaxPerfMode(ctx.max_perf);
        TEST_ERROR(ret < 0, "Error while setting encoder to max perf", cleanup);
    }

    if (ctx.dump_mv) {
        /* Enable dumping of motion vectors report from encoder */
        ret = ctx.enc->enableMotionVectorReporting();
        TEST_ERROR(ret < 0, "Could not enable motion vector reporting", cleanup);
    }

    if (ctx.bnoIframe) {
        ctx.iframe_interval = ((1 << 31) + 1); /* TODO: how can we do this properly */
        ret = ctx.enc->setIFrameInterval(ctx.iframe_interval);
        TEST_ERROR(ret < 0, "Could not set encoder I-Frame interval", cleanup);
    }

    if (ctx.enableROI) {
        v4l2_enc_enable_roi_param VEnc_enable_ext_roi_ctrl;

        VEnc_enable_ext_roi_ctrl.bEnableROI = ctx.enableROI;
        /* Enable region of intrest configuration for encoder */
        ret = ctx.enc->enableROI(VEnc_enable_ext_roi_ctrl);
        TEST_ERROR(ret < 0, "Could not enable ROI", cleanup);
    }

    if (ctx.bReconCrc) {
        v4l2_enc_enable_reconcrc_param VEnc_enable_recon_crc_ctrl;

        VEnc_enable_recon_crc_ctrl.bEnableReconCRC = ctx.bReconCrc;
        /* Enable reconstructed CRC configuration for encoder */
        ret = ctx.enc->enableReconCRC(VEnc_enable_recon_crc_ctrl);
        TEST_ERROR(ret < 0, "Could not enable Recon CRC", cleanup);
    }

    if (ctx.externalRPS) {
        v4l2_enc_enable_ext_rps_ctr VEnc_enable_ext_rps_ctrl;

        VEnc_enable_ext_rps_ctrl.bEnableExternalRPS = ctx.externalRPS;
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H264) {
            VEnc_enable_ext_rps_ctrl.bGapsInFrameNumAllowed = ctx.bGapsInFrameNumAllowed;
            VEnc_enable_ext_rps_ctrl.nH264FrameNumBits = ctx.nH264FrameNumBits;
        }
        if (ctx.encoder_pixfmt == V4L2_PIX_FMT_H265) {
            VEnc_enable_ext_rps_ctrl.nH265PocLsbBits = ctx.nH265PocLsbBits;
        }
        /* Enable external reference picture set configuration for encoder */
        ret = ctx.enc->enableExternalRPS(VEnc_enable_ext_rps_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RPS", cleanup);
    }

    if (ctx.num_reference_frames) {
        /* Set number of reference frame configuration value for encoder */
        ret = ctx.enc->setNumReferenceFrames(ctx.num_reference_frames);
        TEST_ERROR(ret < 0, "Could not set num reference frames", cleanup);
    }

    if (ctx.externalRCHints) {
        v4l2_enc_enable_ext_rate_ctr VEnc_enable_ext_rate_ctrl;

        VEnc_enable_ext_rate_ctrl.bEnableExternalPictureRC = ctx.externalRCHints;
        VEnc_enable_ext_rate_ctrl.nsessionMaxQP = ctx.sMaxQp;

        /* Enable external rate control configuration for encoder */
        ret = ctx.enc->enableExternalRC(VEnc_enable_ext_rate_ctrl);
        TEST_ERROR(ret < 0, "Could not enable external RC", cleanup);
    }
    if (ctx.encoder_pixfmt == V4L2_PIX_FMT_AV1) {
        if (ctx.enable_av1tile) {
            v4l2_enc_av1_tile_config VEnc_av1_tile_config;

            VEnc_av1_tile_config.bEnableTile = ctx.enable_av1tile;
            VEnc_av1_tile_config.nLog2RowTiles = ctx.log2_num_av1rows;
            VEnc_av1_tile_config.nLog2ColTiles = ctx.log2_num_av1cols;

            /* Enable tile configuration for encoder */
            ret = ctx.enc->enableAV1Tile(VEnc_av1_tile_config);
            TEST_ERROR(ret < 0, "Could not enable Tile Configuration", cleanup);
        }
        if (ctx.enable_av1ssimrdo != (uint8_t)-1) {
            ret = ctx.enc->setAV1SsimRdo(ctx.enable_av1ssimrdo);
            TEST_ERROR(ret < 0, "Could not set Ssim RDO", cleanup);
        }
        if (ctx.disable_av1cdfupdate != (uint8_t)-1) {
            ret = ctx.enc->setAV1DisableCDFUpdate(ctx.disable_av1cdfupdate);
            TEST_ERROR(ret < 0, "Could not set disable CDF update", cleanup);
        }
    }
    /* Query, Export and Map the output plane buffers so that we can read
       raw data into the buffers */
    switch (ctx.output_memory_type) {
    case V4L2_MEMORY_MMAP:
        ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
        TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
        break;

    case V4L2_MEMORY_USERPTR:
        ret = ctx.enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
        TEST_ERROR(ret < 0, "Could not setup output plane", cleanup);
        break;

    case V4L2_MEMORY_DMABUF:
        ret = setup_output_dmabuf(&ctx, 10);
        TEST_ERROR(ret < 0, "Could not setup plane", cleanup);
        break;
    default:
        TEST_ERROR(true, "Not a valid plane", cleanup);
    }

    /* Query, Export and Map the capture plane buffers so that we can write
       encoded bitstream data into the buffers */
    ret = ctx.enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, ctx.num_output_buffers,
                                            true, false);
    TEST_ERROR(ret < 0, "Could not setup capture plane", cleanup);

    /* Subscibe for End Of Stream event */
    ret = ctx.enc->subscribeEvent(V4L2_EVENT_EOS, 0, 0);
    TEST_ERROR(ret < 0, "Could not subscribe EOS event", cleanup);

    if (ctx.b_use_enc_cmd) {
        /* Send v4l2 command for encoder start */
        ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_START, 0);
        TEST_ERROR(ret < 0, "Error in start of encoder commands ", cleanup);
    } else {
        /* set encoder output plane STREAMON */
        ret = ctx.enc->output_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in output plane streamon", cleanup);

        /* set encoder capture plane STREAMON */
        ret = ctx.enc->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in capture plane streamon", cleanup);
    }

    if (ctx.blocking_mode) {
        if (ctx.RPS_threeLayerSvc) {
            sem_init(&ctx.rps_par.sema, 0, 0);
        }
        /* Set encoder capture plane dq thread callback for blocking io mode */
        ctx.enc->capture_plane.setDQThreadCallback(self->encoder_capture_plane_dq_callback);

        /* startDQThread starts a thread internally which calls the
           encoder_capture_plane_dq_callback whenever a buffer is dequeued
           on the plane */
        ctx.enc->capture_plane.startDQThread(&ctx);
    } else {
        sem_init(&ctx.pollthread_sema, 0, 0);
        sem_init(&ctx.encoderthread_sema, 0, 0);
        /* Set encoder poll thread for non-blocking io mode */
        pthread_create(&ctx.enc_pollthread, NULL, encoder_pollthread_fcn, &ctx);
        pthread_setname_np(ctx.enc_pollthread, "EncPollThread");
        cout << "Created the PollThread and Encoder Thread \n";
    }

    /* Enqueue all the empty capture plane buffers. */
    for (uint32_t i = 0; i < ctx.enc->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        ret = ctx.enc->capture_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error while queueing buffer at capture plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
    }

    if (ctx.copy_timestamp) {
        /* Set user provided timestamp when copy timestamp is enabled */
        ctx.timestamp = (ctx.start_ts * MICROSECOND_UNIT);
        ctx.timestampincr = (MICROSECOND_UNIT * 16) / ((uint32_t)(ctx.fps_n * 16));
    }
    if (ctx.ppe_init_params.enable_ppe) {
        ctx.ppe_init_params.width = ctx.width;
        ctx.ppe_init_params.height = ctx.height;
        ret = ctx.enc->setPPEInitParams(ctx.ppe_init_params);
        if (ret < 0) {
            cerr << "Error calling setPPEInitParams" << endl;
        }
    }
    /* Read video frame and queue all the output plane buffers. */
    for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer = ctx.enc->output_plane.getNthBuffer(i);

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        if (ctx.output_memory_type == V4L2_MEMORY_DMABUF) {
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF;
            /* Map output plane buffer for memory type DMABUF. */
            ret = ctx.enc->output_plane.mapOutputBuffers(v4l2_buf, ctx.output_plane_fd[i]);

            if (ret < 0) {
                cerr << "Error while mapping buffer at output plane" << endl;
                abort(&ctx);
                goto cleanup;
            }
        }

        if (ctx.startf) {
            uint32_t i = 0, frame_size = 0;

            for (i = 0; i < buffer->n_planes; i++) {
                frame_size += buffer->planes[i].fmt.bytesperpixel * buffer->planes[i].fmt.width * buffer->planes[i].fmt.height;
            }
            frame_size = frame_size * ctx.startf;
            ctx.in_file->seekg(frame_size, std::ios::cur);
            ctx.startf = 0;
        }

        /* Read yuv frame data from input file */
        if (self->read_video_frame_my(ctx.in_file, *buffer) < 0 || ctx.num_frames_to_encode == 0) {
            cerr << "Could not read complete frame from input file" << endl;
            v4l2_buf.m.planes[0].bytesused = 0;
            if (ctx.b_use_enc_cmd) {
                /* Send v4l2 command for encoder stop */
                ret = ctx.enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
                eos = true;
                break;
            } else {
                eos = true;
                v4l2_buf.m.planes[0].m.userptr = 0;
                v4l2_buf.m.planes[0].bytesused = v4l2_buf.m.planes[1].bytesused = v4l2_buf.m.planes[2].bytesused = 0;
            }
        }

        if (ctx.runtime_params_str &&
            (ctx.enc->output_plane.getTotalQueuedBuffers() ==
             ctx.next_param_change_frame)) {
            /* Set runtime configuration parameters */
            set_runtime_params(&ctx);
            if (ctx.runtime_params_str)
                get_next_runtime_param_change_frame(&ctx);
        }

        /* Encoder supported input metadata specific configurations */
        if (ctx.input_metadata) {
            v4l2_ctrl_videoenc_input_metadata VEnc_imeta_param;
            v4l2_enc_frame_ROI_params VEnc_ROI_params;
            v4l2_enc_frame_ReconCRC_params VEnc_ReconCRC_params;
            v4l2_enc_frame_ext_rps_ctrl_params VEnc_ext_rps_ctrl_params;
            v4l2_enc_frame_ext_rate_ctrl_params VEnc_ext_rate_ctrl_params;
            v4l2_enc_gdr_params VEnc_gdr_params;
            VEnc_imeta_param.flag = 0;

            if (ctx.ROI_Param_file_path) {
                if (ctx.enableROI) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_ROI_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncROIParams = &VEnc_ROI_params;

                    /* Update Region of Intrest parameters from ROI params file */
                    populate_roi_Param(ctx.roi_Param_file, VEnc_imeta_param.VideoEncROIParams);
                }
            }

            if (ctx.bReconCrc) {
                VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RECONCRC_PARAM_FLAG;

                VEnc_ReconCRC_params.ReconCRCRect.left = ctx.rl;
                VEnc_ReconCRC_params.ReconCRCRect.top = ctx.rt;
                VEnc_ReconCRC_params.ReconCRCRect.width = ctx.rw;
                VEnc_ReconCRC_params.ReconCRCRect.height = ctx.rh;

                /* Update reconstructed CRC Parameters */
                VEnc_imeta_param.VideoReconCRCParams = &VEnc_ReconCRC_params;
            }

            if (ctx.externalRPS) {
                if (ctx.RPS_threeLayerSvc) {
                    if (ctx.input_frames_queued_count > 0) {
                        sem_wait(&ctx.rps_par.sema);
                    }
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;
                    populate_ext_rps_threeLayerSvc_Param(&ctx, VEnc_imeta_param.VideoEncRPSParams);
                } else if (ctx.RPS_Param_file_path) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RPS_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncRPSParams = &VEnc_ext_rps_ctrl_params;

                    /* Update external reference picture set parameters from RPS params file */
                    populate_ext_rps_ctrl_Param(ctx.rps_Param_file, VEnc_imeta_param.VideoEncRPSParams);
                }
            }

            if (ctx.GDR_Param_file_path) {
                if (ctx.enableGDR) {
                    /* Update GDR parameters from GDR params file */
                    if (ctx.gdr_start_frame_number == 0xFFFFFFFF)
                        populate_gdr_Param(ctx.gdr_Param_file, &ctx.gdr_start_frame_number,
                                           &ctx.gdr_num_frames);
                    if (ctx.input_frames_queued_count == ctx.gdr_start_frame_number) {
                        ctx.gdr_out_frame_number = ctx.gdr_start_frame_number;
                        VEnc_gdr_params.nGDRFrames = ctx.gdr_num_frames;
                        VEnc_imeta_param.flag |= V4L2_ENC_INPUT_GDR_PARAM_FLAG;
                        VEnc_imeta_param.VideoEncGDRParams = &VEnc_gdr_params;
                    }
                }
            }

            if (ctx.hints_Param_file_path) {
                if (ctx.externalRCHints) {
                    VEnc_imeta_param.flag |= V4L2_ENC_INPUT_RC_PARAM_FLAG;
                    VEnc_imeta_param.VideoEncExtRCParams = &VEnc_ext_rate_ctrl_params;

                    /* Update external rate control parameters from hints params file */
                    populate_ext_rate_ctrl_Param(ctx.hints_Param_file, VEnc_imeta_param.VideoEncExtRCParams);
                }
            }

            if (VEnc_imeta_param.flag) {
                /* Set encoder input metadatas */
                ctx.enc->SetInputMetaParams(v4l2_buf.index, VEnc_imeta_param);
                v4l2_buf.reserved2 = v4l2_buf.index;
            }
        }

        if (ctx.copy_timestamp) {
            v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
            ctx.timestamp += ctx.timestampincr;
            v4l2_buf.timestamp.tv_sec = ctx.timestamp / (MICROSECOND_UNIT);
            v4l2_buf.timestamp.tv_usec = ctx.timestamp % (MICROSECOND_UNIT);
        }

        if (ctx.output_memory_type == V4L2_MEMORY_DMABUF || ctx.output_memory_type == V4L2_MEMORY_MMAP) {
            for (uint32_t j = 0; j < buffer->n_planes; j++) {
                NvBufSurface *nvbuf_surf = 0;
                ret = NvBufSurfaceFromFd(buffer->planes[j].fd, (void **)(&nvbuf_surf));
                if (ret < 0) {
                    cerr << "Error while NvBufSurfaceFromFd" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
                ret = NvBufSurfaceSyncForDevice(nvbuf_surf, 0, j);
                if (ret < 0) {
                    cerr << "Error while NvBufSurfaceSyncForDevice at output plane for V4L2_MEMORY_DMABUF" << endl;
                    abort(&ctx);
                    goto cleanup;
                }
            }
        }

        if (ctx.output_memory_type == V4L2_MEMORY_DMABUF) {
            for (uint32_t j = 0; j < buffer->n_planes; j++) {
                v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
            }
        }
        /* encoder qbuffer for output plane */
        ret = ctx.enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error while queueing buffer at output plane" << endl;
            abort(&ctx);
            goto cleanup;
        }
        if (ctx.num_frames_to_encode > 0) {
            ctx.num_frames_to_encode--;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            cerr << "File read complete." << endl;
            eos = true;
            break;
        }
        ctx.input_frames_queued_count++;
    }

    if (ctx.blocking_mode) {
        /* Wait till capture plane DQ Thread finishes
           i.e. all the capture plane buffers are dequeued. */
        if (self->encoder_proc_blocking(ctx, eos) != 0)
            goto cleanup;
        ctx.enc->capture_plane.waitForDQThread(-1);
    } else {
        if (self->encoder_proc_nonblocking(ctx, eos) != 0)
            goto cleanup;
    }

    if (ctx.stats) {
        ctx.enc->printProfilingStats(cout);
    }

cleanup:
    if (ctx.enc && ctx.enc->isInError()) {
        cerr << "Encoder is in error" << endl;
        error = 1;
    }
    if (ctx.got_error) {
        error = 1;
    }

    if (ctx.pBitStreamCrc) {
        char *pgold_crc = ctx.gold_crc;
        Crc *pout_crc = ctx.pBitStreamCrc;
        char StrCrcValue[20];
        snprintf(StrCrcValue, 20, "%u", pout_crc->CrcValue);
        /* Remove CRLF from end of CRC, if present */
        do {
            unsigned int len = strlen(pgold_crc);
            if (len == 0)
                break;
            if (pgold_crc[len - 1] == '\n')
                pgold_crc[len - 1] = '\0';
            else if (pgold_crc[len - 1] == '\r')
                pgold_crc[len - 1] = '\0';
            else
                break;
        } while (1);

        /* Check with golden CRC */
        if (strcmp(StrCrcValue, pgold_crc)) {
            cout << "======================" << endl;
            cout << "video_encode: CRC FAILED" << endl;
            cout << "======================" << endl;
            cout << "Encoded CRC: " << StrCrcValue << " Gold CRC: " << pgold_crc << endl;
            error = 1;
        } else {
            cout << "======================" << endl;
            cout << "video_encode: CRC PASSED" << endl;
            cout << "======================" << endl;
        }

        CloseCrc(&ctx.pBitStreamCrc);
    }

    if (ctx.output_memory_type == V4L2_MEMORY_DMABUF && ctx.enc) {
        for (uint32_t i = 0; i < ctx.enc->output_plane.getNumBuffers(); i++) {
            /* Unmap output plane buffer for memory type DMABUF. */
            ret = ctx.enc->output_plane.unmapOutputBuffers(i, ctx.output_plane_fd[i]);
            if (ret < 0) {
                cerr << "Error while unmapping buffer at output plane" << endl;
                goto cleanup;
            }

            ret = NvBufSurf::NvDestroy(ctx.output_plane_fd[i]);
            if (ret < 0) {
                cerr << "Failed to Destroy NvBuffer\n"
                     << endl;
                return NULL;
            }
        }
    }

    /* Release encoder configuration specific resources. */
    delete ctx.enc;
    // delete ctx.in_file;
    // delete ctx.out_file;
    delete ctx.roi_Param_file;
    delete ctx.recon_Ref_file;
    delete ctx.rps_Param_file;
    delete ctx.hints_Param_file;
    delete ctx.gdr_Param_file;
    delete ctx.gdr_out_file;

    free(ctx.in_file_path);
    free(ctx.out_file_path);
    free(ctx.ROI_Param_file_path);
    free(ctx.Recon_Ref_file_path);
    free(ctx.RPS_Param_file_path);
    free(ctx.hints_Param_file_path);
    free(ctx.GDR_Param_file_path);
    free(ctx.GDR_out_file_path);
    delete ctx.runtime_params_str;

    if (ctx.blocking_mode) {
        if (ctx.RPS_threeLayerSvc) {
            sem_destroy(&ctx.rps_par.sema);
        }
    } else {
        sem_destroy(&ctx.pollthread_sema);
        sem_destroy(&ctx.encoderthread_sema);
    }
    // return -error;
    return NULL;
}
