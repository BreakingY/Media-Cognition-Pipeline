#include "aac_demuxer.h"

void AACDemuxer::InputData(const uint8_t* data, size_t size){
    struct RtpHeader *header = (struct RtpHeader *)data;
    int payload_type = header->payloadType;
    if(payload_type != payload_){
        return;
    }
    const uint8_t* payload = data + sizeof(struct RtpHeader);
    size_t payload_len = size - sizeof(struct RtpHeader);
    // RTP extension head
    if (header->extension){
        const uint8_t *extension_data = payload;
        size_t extension_length = 4 * (extension_data[2] << 8 | extension_data[3]);
        size_t payload_offset = 4 + extension_length;
        payload = payload + payload_offset;
        payload_len = payload_len - payload_offset;
    }
    if(call_back_){
        call_back_->OnAudioData(ntohl(header->timestamp),  payload + 4, payload_len - 4); // 4 for AU Header, A RTP packet usually only encapsulates one frame of AAC
    }
    return;
}