#include "h264_demuxer.h"

void H264Demuxer::InputData(const uint8_t* data, size_t size){
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
    struct H264NaluHeader *h264_header = (struct H264NaluHeader *)payload;
    if(h264_header->type == 28){ // Fragmentation
        struct H264FUIndicator *fu_indicator = (struct H264FUIndicator *)payload;
        struct H264FUHeader *fu_header = (struct H264FUHeader *)&payload[1];
        if(fu_header->s == 1){ // start
            find_start_ = true;
            if(pos_buffer_ == 0){
                struct H264NaluHeader header;
                header.f = fu_indicator->f;
                header.nri = fu_indicator->nri;
                header.type = fu_header->type;
                buffer_[0] = 0;
                buffer_[1] = 0;
                buffer_[2] = 0;
                buffer_[3] = 1;
                memcpy(buffer_ + 4, &header, sizeof(struct H264NaluHeader));
                pos_buffer_ += 4 + sizeof(struct H264NaluHeader);
            }
            memcpy(buffer_ + pos_buffer_, payload + 2, payload_len - 2);
            pos_buffer_ += payload_len - 2;
        }
        else if(fu_header->e == 1){ // end
            if(find_start_ == false){
                return;
            }
            memcpy(buffer_ + pos_buffer_, payload + 2, payload_len - 2);
            pos_buffer_ += payload_len - 2;
            if(call_back_){
                call_back_->OnVideoData(ntohl(header->timestamp),  buffer_, pos_buffer_);
            }
            find_start_ = false;
            pos_buffer_ = 0;
        }
        else { // Middle partition
            if (!find_start_) {
                return;
            }
            memcpy(buffer_ + pos_buffer_, payload + 2, payload_len - 2);
            pos_buffer_ += payload_len - 2;
        }
    }
    else{ // Single packet
        buffer_[0] = 0;
        buffer_[1] = 0;
        buffer_[2] = 0;
        buffer_[3] = 1;
        memcpy(buffer_ + 4, payload, payload_len);
        if(call_back_){
            call_back_->OnVideoData(ntohl(header->timestamp),  buffer_, payload_len + 4);
        }
    }
    return;
}