#include <stdio.h>
#include <stdlib.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
#include <libavformat/avformat.h>
/*
typedef struct AVCodecParameters {
    enum AVMediaType codec_type;
    enum AVCodecID   codec_id;
    uint32_t         codec_tag;

    uint8_t *extradata;
    int      extradata_size;
   
    int format;

    int64_t bit_rate;
	
    int bits_per_coded_sample;

    
    int bits_per_raw_sample;

    
    int profile;
    int level;

    
    int width;
    int height;

    
    AVRational sample_aspect_ratio;
	
    enum AVFieldOrder                  field_order;

    
    enum AVColorRange                  color_range;
    enum AVColorPrimaries              color_primaries;
    enum AVColorTransferCharacteristic color_trc;
    enum AVColorSpace                  color_space;
    enum AVChromaLocation              chroma_location;

    
    int video_delay;

    
    uint64_t channel_layout;
    
    int      channels;
    
    int      sample_rate;
    
    int      block_align;
    
    int      frame_size;
	
    int initial_padding;
    
    int trailing_padding;
    
    int seek_preroll;
} AVCodecParameters;
*/
void printCodec(AVCodecParameters *in_codecpar){
        printf("**********************************\n");
        printf("codec_type:%d\n",in_codecpar->codec_type);
        printf("codec_id:%d\n",in_codecpar->codec_id);
        printf("codec_tag:%d\n",in_codecpar->codec_tag);
        printf("format:%d\n",in_codecpar->format);
        printf("bit_rate:%d\n",in_codecpar->bit_rate);
        printf("bits_per_coded_sample:%d\n",in_codecpar->bits_per_coded_sample);
        printf("bits_per_raw_sample:%d\n",in_codecpar->bits_per_raw_sample);
        printf("profile:%d\n",in_codecpar->profile);
        printf("level:%d\n",in_codecpar->level);
        printf("width:%d\n",in_codecpar->width);
        printf("height:%d\n",in_codecpar->height);
        printf("sample_aspect_ratio.num:%d\n",in_codecpar->sample_aspect_ratio.num);
        printf("sample_aspect_ratio.den:%d\n",in_codecpar->sample_aspect_ratio.den);
        printf("field_order:%d\n",in_codecpar->field_order);
        printf("color_range:%d\n",in_codecpar->color_range);
        printf("color_primaries:%d\n",in_codecpar->color_primaries);
        printf("color_trc:%d\n",in_codecpar->color_trc);
        printf("color_space:%d\n",in_codecpar->color_space);
        printf("chroma_location:%d\n",in_codecpar->chroma_location);
        printf("video_delay:%d\n",in_codecpar->video_delay);
        printf("channel_layout:%d\n",in_codecpar->channel_layout);
        printf("channels:%d\n",in_codecpar->channels);
        printf("sample_rate:%d\n",in_codecpar->sample_rate);
        printf("block_align:%d\n",in_codecpar->block_align);
        printf("frame_size:%d\n",in_codecpar->frame_size);
        printf("initial_padding:%d\n",in_codecpar->initial_padding);
        printf("trailing_padding:%d\n",in_codecpar->trailing_padding);
        printf("seek_preroll:%d\n",in_codecpar->seek_preroll);
}


int main(int argc, char** argv)
{
    AVOutputFormat *ofmt = NULL;
    //输入对应一个AVFormatContext，输出对应一个AVFormatContext
    //（Input AVFormatContext and Output AVFormatContext）
    AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx = NULL;
    AVPacket pkt;
    const char *in_filename, *out_filename;
    int ret, i;
    if (argc < 3) {
        printf("usage: %s input output\n", argv[0]);
        return 1;
    }
    in_filename  = argv[1];//输入文件名（Input file URL）
    out_filename = argv[2];//输出文件名（Output file URL）
    av_register_all();
    //输入（Input）
    if ((ret = avformat_open_input(&ifmt_ctx, in_filename, 0, 0)) < 0) {
        printf( "Could not open input file.");
        goto end;
    }
    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0) {
        printf( "Failed to retrieve input stream information");
        goto end;
    }
    av_dump_format(ifmt_ctx, 0, in_filename, 0);
    //输出（Output）
    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, out_filename);
    if (!ofmt_ctx) {
        printf( "Could not create output context\n");
        ret = AVERROR_UNKNOWN;
        goto end;
    }
    ofmt = ofmt_ctx->oformat;
    for (i = 0; i < ifmt_ctx->nb_streams; i++) {
        //根据输入流创建输出流（Create output AVStream according to input AVStream）
        AVStream *in_stream = ifmt_ctx->streams[i];
        AVStream *out_stream =avformat_new_stream(ofmt_ctx, NULL);
                AVCodecParameters *in_codecpar = in_stream->codecpar;//编解码参数
                printCodec(in_codecpar);
        if (!out_stream) {
            printf( "Failed allocating output stream\n");
            ret = AVERROR_UNKNOWN;
            goto end;
        }
        for(int i=0;i<in_stream->codec->extradata_size;i++){
                printf("%02x ",in_stream->codec->extradata[i]);
        }
        printf("\n****************\n");
        ret = avcodec_parameters_copy(out_stream->codecpar, in_codecpar);// avcodec_parameters_to_context(*dec_ctx, st->codecpar)是把参数拷贝到解码器Context里面，和avcodec_parameters_copy不同
        if (ret < 0) {
            fprintf(stderr, "Failed to copy codec parameters\n");
            goto end;
        }
        out_stream->codec->codec_tag = 0;
                if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER){
                        out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                }
    }
    //输出一下格式------------------
    av_dump_format(ofmt_ctx, 0, out_filename, 1);
    //打开输出文件（Open output file）
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, out_filename, AVIO_FLAG_WRITE);
        if (ret < 0) {
            printf( "Could not open output file '%s'", out_filename);
            goto end;
        }
    }
    //写文件头（Write file header）
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) {
        printf( "Error occurred when opening output file\n");
        goto end;
    }
    int frame_index=0;
    while (1) {
        AVStream *in_stream, *out_stream;
        //获取一个AVPacket（Get an AVPacket）
        ret = av_read_frame(ifmt_ctx, &pkt);
        if (ret < 0){
                        av_packet_unref(&pkt);
                        break;
                }

        in_stream  = ifmt_ctx->streams[pkt.stream_index];
        out_stream = ofmt_ctx->streams[pkt.stream_index];
        /* copy packet */
        //转换PTS/DTS（Convert PTS/DTS）
        pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
        pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
        pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
        pkt.pos = -1;
        //写入（Write）
        ret = av_interleaved_write_frame(ofmt_ctx, &pkt);
        if (ret < 0) {
            printf( "Error muxing packet\n");
                        av_packet_unref(&pkt);
            break;
        }
        //printf("Write %8d frames to output file\n",frame_index);
        av_packet_unref(&pkt);
        frame_index++;
    }
    //写文件尾（Write file trailer）
    av_write_trailer(ofmt_ctx);
end:
    avformat_close_input(&ifmt_ctx);
    /* close output */
    if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE))
        avio_close(ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    if (ret < 0 && ret != AVERROR_EOF) {
        printf( "Error occurred.\n");
        return -1;
    }
    return 0;
}
