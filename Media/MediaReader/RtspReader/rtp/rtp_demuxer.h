#ifndef RTP_DEMUXER_
#define RTP_DEMUXER_
#include <iostream>
#include <stdint.h>
#include <socket_io.h>
#include "rtsp_common.h"

class RTPDemuxerInterface {
public:
  virtual void OnVideoData(int64_t pts, const uint8_t* data, size_t size) = 0; //video demuxer only
  virtual void OnAudioData(int64_t pts,  const uint8_t* data, size_t size) = 0; //audio demuxer only
};

class RTPDemuxer{
public:
  virtual void InputData(const uint8_t* data, size_t size) = 0;
  void SetCallBack(RTPDemuxerInterface *call_back) {call_back_ = call_back; return;}
  void SetPayloadType(int payload){payload_ = payload; return;}
protected:
  RTPDemuxerInterface *call_back_ = NULL;
  int payload_;
};
#endif