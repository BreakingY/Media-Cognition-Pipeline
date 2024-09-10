#ifndef H264_DEMUXER_
#define H264_DEMUXER_
#include <iostream>
#include <string.h>
#include "rtp_demuxer.h"
struct H264NaluHeader {
  uint8_t type : 5;
  uint8_t nri : 2;
  uint8_t f : 1;
};
struct H264FUIndicator {
  uint8_t type : 5;
  uint8_t nri : 2;
  uint8_t f : 1;
};
struct H264FUHeader {
  uint8_t type : 5;
  uint8_t r : 1;
  uint8_t e : 1;
  uint8_t s : 1;
}; 

class H264Demuxer : public RTPDemuxer{
public:
    void InputData(const uint8_t* data, size_t size);
private:
    uint8_t buffer_[4 * 1024 * 1024];
    bool find_start_ = false;
    size_t pos_buffer_ = 0;
};
#endif