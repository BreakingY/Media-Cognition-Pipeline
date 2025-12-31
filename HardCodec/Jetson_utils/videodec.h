/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NvVideoDecoder.h"
#include "NvEglRenderer.h"
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include <queue>
#include <fstream>
#include <pthread.h>
#include "nvosd.h"
#include "NvBufSurface.h"

#define MAX_RECT_NUM 100
#define MAX_BUFFERS 32
#include <list>
typedef struct
{    
    NvVideoDecoder *dec;
    uint32_t decoder_pixfmt;

    void *nvosd_context;
    NvEglRenderer *renderer;

    char *in_file_path;
    std::ifstream *in_file;

    char *out_file_path;
    std::ofstream *out_file;

    EGLImageKHR egl_image;

    bool disable_rendering;
    bool fullscreen;
    uint32_t dec_width;
    uint32_t dec_height;
    uint32_t window_height;
    uint32_t window_width;
    uint32_t window_x;
    uint32_t window_y;
    uint32_t out_pixfmt;
    float fps;

    bool input_nalu;

    bool enable_osd;
    bool enable_osd_text;
    char *osd_file_path;
    std::ifstream *osd_file;
    NvOSD_RectParams g_rect[MAX_RECT_NUM];
    int  g_rect_num;

    char* osd_text;
    NvOSD_TextParams textParams;

    pthread_t dec_capture_loop;
    bool got_error;
    bool got_eos;
    int dst_dma_fd;
} context_t;

