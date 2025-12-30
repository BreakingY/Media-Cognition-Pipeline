#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
#include "NodeFlow.h"

std::unique_ptr<YoloDetectionNode> detect = nullptr;
std::shared_ptr<CollectorNode> collector = nullptr;
std::shared_ptr<RelayNode> relayer1 = nullptr;
std::shared_ptr<DistributorNode> distributor = nullptr;
std::once_flag detect_init_flag;
std::map<QueueContext*, TrackerNode*> stream_map;
std::mutex flow_mutex_g;
std::string eng_path_g;
int device_id_g;
void DetectModelInit(std::string eng_path, int device_id){
    std::lock_guard<std::mutex> guard(flow_mutex_g);
    eng_path_g = eng_path;
    device_id_g = device_id;
    if(detect == nullptr) {
        detect = std::make_unique<YoloDetectionNode>(eng_path, device_id);
        collector = std::make_shared<CollectorNode>();
        relayer1 = std::make_shared<RelayNode>();
        distributor = std::make_shared<DistributorNode>();
        detect->SetDataNode(collector, relayer1, nullptr);
    };
}
void* AddStream(InferDataListner* listener, int width, int height, int fps){
    if(detect == nullptr){
        DetectModelInit(eng_path_g, device_id_g);
    }
    QueueContext* context = CreateContext(listener, width, height);
    TrackerNode *tracker_node = new TrackerNode(fps);
    tracker_node->SetDataNode(context->stream_id, relayer1, nullptr, distributor);
    std::lock_guard<std::mutex> guard(flow_mutex_g);
    stream_map[context] = tracker_node;
    return context;
}
void EndStream(void* context){
    if(!context){
        return;
    }
    QueueContext* ctx = (QueueContext*)context;
    std::lock_guard<std::mutex> guard(flow_mutex_g);
    relayer1->MarkStreamEOS(ctx->stream_id);
    delete stream_map[ctx];
    stream_map.erase(ctx);
    DestroyContext(ctx);
    if(stream_map.empty()){
        detect.reset();
        detect = nullptr;
        collector.reset();
        collector = nullptr;
        relayer1.reset();
        relayer1 = nullptr;
        distributor.reset();
        distributor = nullptr;
    }
}
void StreamPushData(cv::Mat &img, void* context){
    QueueContext* ctx = (QueueContext*)context;
    collector->Push(std::move(img), ctx);
}
#endif // DETECTION_NVIDIA DETECTION_ASCEND
