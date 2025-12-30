#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#include "TrackerNode.h"
TrackerNode::TrackerNode(int fps){
    tracker_ = std::make_unique<byte_track::BYTETracker>(fps, fps * 2);
}
TrackerNode::~TrackerNode(){
    abort_ = true;
    if(thread_run_flag_)
        worker_.join();

}
void TrackerNode::SetDataNode(int64_t stream_id, std::shared_ptr<RelayNode> relayer_in, std::shared_ptr<RelayNode> relayer_out, std::shared_ptr<DistributorNode> distributor){
    stream_id_ = stream_id;
    relayer_in_ = relayer_in; 
    relayer_out_ = relayer_out; 
    distributor_ = distributor;
    worker_ = std::thread(&TrackerNode::TrackThreadLoop, this);
}
static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}
void TrackerNode::TrackUpdate(ImgPacket *packet){
    DetectionInfo info = packet->info;
    std::vector<Detection> &dets = info.dets;
    std::vector<byte_track::Object> objects;
    for (auto& d : dets) {
        d.track_id = 0;
        byte_track::Object object(byte_track::Rect(d.box.x, d.box.y, d.box.width, d.box.height), d.class_id, d.score);
        objects.push_back(object);
    }
    const std::vector<byte_track::BYTETracker::STrackPtr> outputs = tracker_->update(objects);
    std::vector<int> allocate_idx;
    for(int k = 0; k < outputs.size(); k++){
        const auto &outputs_per_frame = outputs[k];
        const byte_track::Rect<float> &rect = outputs_per_frame->getRect();
        const size_t &track_id = outputs_per_frame->getTrackId();
        float max_iou = 0.0f;
        int best_match_idx = -1;
        for (size_t i = 0; i < dets.size(); ++i) {
            if(dets[i].track_id != 0) 
                continue;
            cv::Rect2f rect_original = dets[i].box;
            float iou = IoU(rect_original, cv::Rect2f(rect.x(), rect.y(), rect.width(), rect.height()));
            if(iou > max_iou && iou > 0.3f){
                max_iou = iou;
                best_match_idx = i;
            }
        }
        if((best_match_idx != -1) && (std::find(allocate_idx.begin(), allocate_idx.end(), k) == allocate_idx.end())){
            allocate_idx.push_back(k);
            dets[best_match_idx].track_id = track_id;
        }
    }
    packet->info = info;
}
void TrackerNode::TrackThreadLoop(){
    thread_run_flag_ = true;
    while (!abort_) {
        if(!relayer_in_){
            log_error("No data source available");
            return;
        }
        ImgPacket *packet = relayer_in_->Get(stream_id_);
        if(packet == nullptr){
            continue;
        }
        TrackUpdate(packet);
        if(relayer_out_){
            relayer_out_->Push(packet);
        }
        if(distributor_){
            distributor_->Push(packet);
        }

    }
}
#endif // DETECTION_NVIDIA DETECTION_ASCEND