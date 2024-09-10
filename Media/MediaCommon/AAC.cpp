#include "AAC.h"
int GetSampleRateIndex(int freq){

    int i = 0;
    int freq_arr[13] = {
        96000, 88200, 64000, 48000, 44100, 32000,
        24000, 22050, 16000, 12000, 11025, 8000, 7350
    };
    for(i = 0; i < 13; i++){
        if(freq == freq_arr[i]){
            return i;
        }
    }
    return 4;//默认是44100
}
void GenerateAdtsHeader(char *adts_header_buffer, int data_len, int profile, int sample_rate_index, int channels){
    int adts_len = data_len + 7;

    adts_header_buffer[0] = 0xff;         //syncword:0xfff                          高8bits
    adts_header_buffer[1] = 0xf0;         //syncword:0xfff                          低4bits
    adts_header_buffer[1] |= (0 << 3);    //MPEG Version:0 for MPEG-4,1 for MPEG-2  1bit
    adts_header_buffer[1] |= (0 << 1);    //Layer:0                                 2bits
    adts_header_buffer[1] |= 1;           //protection absent:1                     1bit

    adts_header_buffer[2] = (profile)<<6;            //profile:audio_object_type - 1                      2bits
    adts_header_buffer[2] |= (sample_rate_index & 0x0f)<<2; //sampling frequency index:sampling_frequency_index  4bits
    adts_header_buffer[2] |= (0 << 1);                             //private bit:0                                      1bit
    adts_header_buffer[2] |= (channels & 0x04)>>2;           //channel configuration:channel_config               高1bit

    adts_header_buffer[3] = (channels & 0x03)<<6;     //channel configuration:channel_config      低2bits
    adts_header_buffer[3] |= (0 << 5);                      //original：0                               1bit
    adts_header_buffer[3] |= (0 << 4);                      //home：0                                   1bit
    adts_header_buffer[3] |= (0 << 3);                      //copyright id bit：0                       1bit
    adts_header_buffer[3] |= (0 << 2);                      //copyright id start：0                     1bit
    adts_header_buffer[3] |= ((adts_len & 0x1800) >> 11);           //frame length：value   高2bits

    adts_header_buffer[4] = (uint8_t)((adts_len & 0x7f8) >> 3);     //frame length:value    中间8bits
    adts_header_buffer[5] = (uint8_t)((adts_len & 0x7) << 5);       //frame length:value    低3bits
    adts_header_buffer[5] |= 0x1f;                                 //buffer fullness:0x7ff 高5bits
    adts_header_buffer[6] = 0xfc;
    return;
}
int ParseAdtsHeader(uint8_t *in, struct AdtsHeader *res)
{
    static int frame_number = 0;
    memset(res, 0, sizeof(*res));

    if ((in[0] == 0xFF) && ((in[1] & 0xF0) == 0xF0)) // syncword
    {
        res->id = ((unsigned int)in[1] & 0x08) >> 3;
        log_debug("adts:id  {}", res->id);
        res->layer = ((unsigned int)in[1] & 0x06) >> 1;
        log_debug("adts:layer  {}", res->layer);
        // 这里需要判断如果protectionAbsent=1 表示没有CRC校验，即头部一共7个字节，如果protectionAbsent=0，表示含有crc校验，则还应该跳过两个字节才是acc数据，aac_frame_length = (protection_absent == 1 ? 7 : 9) + size(AACFrame)
        res->protectionAbsent = (unsigned int)in[1] & 0x01;
        log_debug("adts:protection_absent  {}", res->protectionAbsent);

        res->profile = ((unsigned int)in[2] & 0xc0) >> 6;
        log_debug("adts:profile  {}", res->profile);
        res->samplingFreqIndex = ((unsigned int)in[2] & 0x3c) >> 2;
        log_debug("adts:sf_index  {}", res->samplingFreqIndex);
        res->privateBit = ((unsigned int)in[2] & 0x02) >> 1;
        log_debug("adts:pritvate_bit  {}", res->privateBit);
        res->channelCfg = ((((unsigned int)in[2] & 0x01) << 2) | (((unsigned int)in[3] & 0xc0) >> 6));
        log_debug("adts:channel_configuration  {}", res->channelCfg);
        res->originalCopy = ((unsigned int)in[3] & 0x20) >> 5;
        log_debug("adts:original  {}", res->originalCopy);
        res->home = ((unsigned int)in[3] & 0x10) >> 4;
        log_debug("adts:home  {}", res->home);

        res->copyrightIdentificationBit = ((unsigned int)in[3] & 0x08) >> 3;
        log_debug("adts:copyright_identification_bit  {}", res->copyrightIdentificationBit);
        res->copyrightIdentificationStart = (unsigned int)in[3] & 0x04 >> 2;
        log_debug("adts:copyright_identification_start  {}", res->copyrightIdentificationStart);
        // ADTS从低地址到高地址存放长度数值的高2位 中间8位，低5位
        // ADTS中标明13bit存放，一般存放数据长度这种大多是由低地址到高地址分别存放数值的高位到低位，这样符合人类的习惯
        res->aacFrameLength = (((((unsigned int)in[3]) & 0x03) << 11) |
                               (((unsigned int)in[4] & 0xFF) << 3) |
                               ((unsigned int)in[5] & 0xE0) >> 5);
        log_debug("adts:aac_frame_length  {}", res->aacFrameLength);
        res->adtsBufferFullness = (((unsigned int)in[5] & 0x1f) << 6 |
                                   ((unsigned int)in[6] & 0xfc) >> 2);
        log_debug("adts:adts_buffer_fullness  {}", res->adtsBufferFullness);
        res->numberOfRawDataBlockInFrame = ((unsigned int)in[6] & 0x03);
        log_debug("adts:no_raw_data_blocks_in_frame  {}", res->numberOfRawDataBlockInFrame);

        return 0;
    } else {
        log_error("failed to parse adts header\n");
        return -1;
    }
    return 0;
}