#ifndef H265_DEMUXER_
#define H265_DEMUXER_
#include <iostream>
#include <string.h>
#include "rtp_demuxer.h"
struct H265NaluHeader {
  uint16_t layer_hi : 1;
  uint16_t type : 6;
  uint16_t f : 1;
  uint16_t tid : 3;
  uint16_t layer_low : 5; 
};
struct H265FUHeader {
  uint8_t type : 6;
  uint8_t e : 1;
  uint8_t s : 1;
};

class H265Demuxer : public RTPDemuxer{
public:
    void InputData(const uint8_t* data, size_t size);
private:
    uint8_t buffer_[4 * 1024 * 1024];
    bool find_start_ = false;
    size_t pos_buffer_ = 0;
};
#endif