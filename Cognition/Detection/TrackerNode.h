#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#ifndef TRACKER_NODE_H
#define TRACKER_NODE_H
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <memory>
#include "DataChannel.h"
#include "DetectionInfo.h"
#include "log_helpers.h"
#include "ByteTrack/BYTETracker.h"

class TrackerNode{
public:
    TrackerNode(int fps);
    ~TrackerNode();
    void SetDataNode(int64_t stream_id, std::shared_ptr<RelayNode> relayer_in = nullptr, std::shared_ptr<RelayNode> relayer_out = nullptr, std::shared_ptr<DistributorNode> distributor = nullptr);
    void TrackThreadLoop();
    void TrackUpdate(ImgPacket *packet);
private:
    std::unique_ptr<byte_track::BYTETracker> tracker_;
    std::thread worker_;
    bool abort_ = false;
    bool thread_run_flag_ = false;
    // input
    int64_t stream_id_;
    std::shared_ptr<RelayNode> relayer_in_ = nullptr;
    // output
    std::shared_ptr<RelayNode> relayer_out_ = nullptr;
    std::shared_ptr<DistributorNode> distributor_ = nullptr;
};

#endif // TRACKER_NODE_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND