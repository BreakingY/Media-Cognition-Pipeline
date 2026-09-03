#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND) || defined(DETECTION_HYGON)
#ifndef DATA_CHANNEL_H
#define DATA_CHANNEL_H

#include <atomic>
#include <cstdint>
#include <list>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
#include <cuda_runtime.h>
#endif
#if defined(DETECTION_ASCEND)
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#include <ops/acl_dvpp.h>
#endif
#include "DetectionInfo.h"
#include "log_helpers.h"
#include "TimeMetrics.h"

/// =======================
/// Infer Data Listener
/// =======================
class InferDataListner {
public:
    virtual void OnInferData(cv::Mat& img, DetectionInfo& info, int64_t timestamp) = 0;
};
/// =======================
/// GPUMemPool
/// =======================
class GPUMemPool {
public:
    GPUMemPool(int oneBlockSize, int blocks, const std::string& name){
        logName=name;
#if defined(DETECTION_ASCEND)
        CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &addr, oneBlockSize * blocks));
#elif defined(DETECTION_NVIDIA)
        CHECK_CUDA(cudaMalloc((void**)&addr, oneBlockSize * blocks));
#else
        log_error("Macro error")
#endif
        auto* base = static_cast<unsigned char*>(addr);
        for (unsigned char* ptr = base; ptr < base + oneBlockSize * blocks; ptr += oneBlockSize) {
            dataList_.push_back(ptr);
        }
    }

    ~GPUMemPool() {
#if defined(DETECTION_ASCEND)
        if (addr) {
            CHECK_DVPP_MPI(hi_mpi_dvpp_free(addr));
            addr = nullptr;
        }
#elif defined(DETECTION_NVIDIA)
        if (addr) {
            CHECK_CUDA(cudaFree(addr));
            addr = nullptr;
        }
#else
        log_error("Macro error")
#endif 
    }

    void* getAddr() {
        std::unique_lock<std::mutex> guard(mutex_);
        if (!dataList_.empty()) {
            void* addr = dataList_.front();
            dataList_.pop_front();
            return addr;
        }
        log_debug("get {} empty !!!!!!!!!!!!!!!!!!", logName);

        auto now = std::chrono::system_clock::now();
        cond_.wait_until(guard, now + std::chrono::milliseconds(10));
        if (!dataList_.empty()) {
            void* addr = dataList_.front();
            dataList_.pop_front();
            return addr;
        }
        return nullptr;
    }

    void putAddr(void* addr) {
        if (addr == nullptr) {
            return;
        }
        {
            std::unique_lock<std::mutex> guard(mutex_);
            dataList_.push_back(addr);
        }

        cond_.notify_one();
    }

private:
    std::list<void*> dataList_;
    std::mutex mutex_;
    std::condition_variable cond_;

    void* addr = nullptr;
    std::string logName;
};
/// =======================
/// Queue Context
/// =======================
struct QueueContext {
    int64_t stream_id{0};
    InferDataListner* listener{nullptr};
    int width;
    int height;
    TimeMetrics time_for_log;
#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
    void *img_buffer{nullptr};
    void *pu8_resized{nullptr}; // yolo  Letterbox_resize_GPU
    std::mutex mutex_buffer;
#endif
#if defined(DETECTION_ASCEND)
    std::mutex mutex_buffer;
    void *img_buffer{nullptr};
#endif
#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
    GPUMemPool *mempool{nullptr};
    std::mutex mutex_mempool;
    void *image_ptr{nullptr};
#endif
};

inline QueueContext* CreateContext(InferDataListner* listener, int width, int height) {
    static std::atomic<int64_t> stream_id_init{-1};
    auto* ctx = new QueueContext;
    ctx->stream_id = ++stream_id_init;
    ctx->listener = listener;
    ctx->width = width;
    ctx->height = height;
    return ctx;
}
#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
inline void MemAllocate(QueueContext* ctx, int pu8_resized_w, int pu8_resized_h, int channel){
    std::unique_lock<std::mutex> guard(ctx->mutex_buffer);
    if(ctx->img_buffer == nullptr){
        CHECK_CUDA(cudaMalloc(&ctx->img_buffer, ctx->width * ctx->height * 3));
    }
    if(ctx->pu8_resized == nullptr){
        CHECK_CUDA(cudaMalloc(&ctx->pu8_resized, pu8_resized_w * pu8_resized_h * channel));
    }
}
#endif
#if defined(DETECTION_ASCEND)
inline void MemAllocate(QueueContext* ctx){
    std::unique_lock<std::mutex> guard(ctx->mutex_buffer);
    if(ctx->img_buffer == nullptr){
        CHECK_ACL(hi_mpi_dvpp_malloc(0, &ctx->img_buffer, ctx->width * ctx->height * 3));
    }
}
#endif

#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
inline void MemPoolAllocate(QueueContext* ctx){
    std::unique_lock<std::mutex> guard(ctx->mutex_mempool);
    int dec_image_size = ctx->width * ctx->height * 3;
    if(ctx->mempool == nullptr){
        ctx->mempool = new GPUMemPool(dec_image_size, 8, "device memory");
    }
    if(!ctx->image_ptr){
        ctx->image_ptr = (unsigned char *)malloc(dec_image_size);
    }
}
#endif   
inline void DestroyContext(QueueContext* ctx) {
    if(ctx){
#if defined(DETECTION_NVIDIA) || defined(DETECTION_HYGON)
        if(ctx->img_buffer){
            CHECK_CUDA(cudaFree(ctx->img_buffer));
        }
        if(ctx->pu8_resized){
            CHECK_CUDA(cudaFree(ctx->pu8_resized));
        }
#endif
#if defined(DETECTION_ASCEND)
        if(ctx->img_buffer){
            CHECK_DVPP_MPI(hi_mpi_dvpp_free(ctx->img_buffer));
            ctx->img_buffer = NULL;
        }
#endif
        if(ctx->mempool){
            delete ctx->mempool;
            ctx->mempool = nullptr;
        }
        if(ctx->image_ptr){
            free(ctx->image_ptr);
            ctx->image_ptr = nullptr;
        }
        delete ctx;
    }
}
/// =======================
/// Image Packet
/// =======================
struct ImgPacket{
    cv::Mat img;
    void* device_image_ptr;
    bool use_ptr;
    DetectionInfo info;
    TimeMetrics timer;
    int64_t timestamp/*图像时间戳ms*/;
    QueueContext* context{nullptr};
};
/// =======================
/// Collector Node
/// =======================
class CollectorNode {
public:
    CollectorNode() = default;

    ~CollectorNode() {
        std::unique_lock<std::mutex> guard(mutex_);
        while (!img_list_.empty()) {
            ImgPacket *packet = img_list_.front();
            img_list_.pop_front();
            if(packet){
                delete packet;
                packet = nullptr;
            }
        }
        log_debug("~CollectorNode");
    }

    /// Prefer this: move cv::Mat
    inline void Push(cv::Mat&& img, int64_t timestamp/*ms*/, QueueContext* context) {
        std::unique_lock<std::mutex> guard(mutex_);
        ImgPacket *packet = new ImgPacket();
        packet->img = std::move(img);
        packet->use_ptr = false;
        packet->timer.startTimer();
        packet->timestamp = timestamp;
        packet->context = context;
        img_list_.push_back(packet);
        cond_.notify_one();
    }

    /// Fallback: copy cv::Mat
    inline void Push(const cv::Mat& img, int64_t timestamp/*ms*/, QueueContext* context) {
        std::unique_lock<std::mutex> guard(mutex_);
        ImgPacket *packet = new ImgPacket();
        packet->img = img;
        packet->use_ptr = false;
        packet->timer.startTimer();
        packet->timestamp = timestamp;
        packet->context = context;
        img_list_.push_back(packet);
        cond_.notify_one();
    }

    inline void Push(void* device_image_ptr, int64_t timestamp/*ms*/, QueueContext* context) {
        MemPoolAllocate(context);
        void *addr = context->mempool->getAddr();
        if(addr == nullptr){ // 任务忙，丢帧
            return;
        }
        ImgPacket *packet = new ImgPacket();
        #if defined(DETECTION_NVIDIA)
            CHECK_CUDA(cudaMemcpy(addr, device_image_ptr, context->width * context->height * 3, cudaMemcpyDeviceToDevice));

            CHECK_CUDA(cudaMemcpy(context->image_ptr, device_image_ptr, context->width * context->height * 3, cudaMemcpyDeviceToHost));
            cv::Mat frame_mat(context->height, context->width, CV_8UC3, context->image_ptr);
            packet->img = frame_mat.clone();
        #elif defined(DETECTION_ASCEND)
            CHECK_ACL(aclrtMemcpy(addr, context->width * context->height * 3, device_image_ptr, context->width * context->height * 3, ACL_MEMCPY_DEVICE_TO_DEVICE));

            CHECK_ACL(aclrtMemcpy(context->image_ptr, context->width * context->height * 3, device_image_ptr, context->width * context->height * 3, ACL_MEMCPY_DEVICE_TO_HOST));
            cv::Mat frame_mat(context->height, context->width, CV_8UC3, context->image_ptr);
            packet->img = frame_mat.clone();
        #else
            log_error("Macro error")
            return;
        #endif 
        packet->device_image_ptr = addr;
        packet->use_ptr = true;
        packet->timer.startTimer();
        packet->timestamp = timestamp;
        packet->context = context;
        std::unique_lock<std::mutex> guard(mutex_);
        img_list_.push_back(packet);
        cond_.notify_one();
    }

    inline std::vector<ImgPacket*> GetBatch(size_t batch_size, int &list_size) {
        std::vector<ImgPacket*> batch;
        std::unique_lock<std::mutex> guard(mutex_);
        if (!img_list_.empty()) {
            size_t real_size = std::min(batch_size, img_list_.size());
            for (size_t i = 0; i < real_size; ++i) {
                batch.push_back(img_list_.front());
                img_list_.pop_front();
            }
        } else {
            auto now = std::chrono::system_clock::now();
            cond_.wait_until(guard, now + std::chrono::milliseconds(10));
            guard.unlock();
        }
        list_size = img_list_.size();
        return batch;
    }

private:
    std::list<ImgPacket*> img_list_;
    std::mutex mutex_;
    std::condition_variable cond_;
};

/// =======================
/// RelayNode Node
/// =======================
class RelayNode {
public:
    enum class StreamState {
        ACTIVE,
        EOS,
        REMOVED
    };

    struct StreamEntry {
        std::list<ImgPacket*> queue;
        StreamState state{StreamState::ACTIVE};
    };

public:
    RelayNode() {
        worker_ = std::thread(&RelayNode::CheckThreadLoop, this);
    }
    ~RelayNode() {
        abort_ = true;
        worker_.join();
        std::unique_lock<std::mutex> guard(mutex_);
        for (auto& kv : streams_) {
            auto& q = kv.second.queue;
            while (!q.empty()) {
                ImgPacket *packet = q.front();
                q.pop_front();
                if(packet){
                    delete packet;
                    packet = nullptr;
                }
            }
        }
        streams_.clear();
        log_debug("~RelayNode");
    }


    inline void Push(ImgPacket* packet) {
        if (!packet || !packet->context) 
            return;

        int64_t sid = packet->context->stream_id;
        std::unique_lock<std::mutex> guard(mutex_);
        auto& entry = streams_[sid];
        if(entry.state != StreamState::ACTIVE){
            delete packet;
            return;
        }
        entry.queue.push_back(packet);
        cond_.notify_all();
    }
    inline ImgPacket* Get(int64_t stream_id) {
        ImgPacket* packet = nullptr;
        std::unique_lock<std::mutex> guard(mutex_);

        auto it = streams_.find(stream_id);
        if (it == streams_.end()) {
            return packet;
        }

        auto& entry = it->second;

        if (!entry.queue.empty() && entry.state == StreamState::ACTIVE) {
            packet = entry.queue.front();
            entry.queue.pop_front();
        }
        else{
            auto now = std::chrono::system_clock::now();
            cond_.wait_until(guard, now + std::chrono::milliseconds(10));
            guard.unlock();
        }
        return packet;
    }


    inline std::vector<ImgPacket*> GetBatch(int64_t stream_id, size_t batch_size) {
        std::vector<ImgPacket*> batch;
        std::unique_lock<std::mutex> guard(mutex_);

        auto it = streams_.find(stream_id);
        if (it == streams_.end()) {
            return batch;
        }

        auto& entry = it->second;

        if (!entry.queue.empty() && entry.state == StreamState::ACTIVE) {
            size_t real = std::min(batch_size, entry.queue.size());
            for (size_t i = 0; i < real; ++i) {
                batch.push_back(entry.queue.front());
                entry.queue.pop_front();
            }
        }
        else{
            auto now = std::chrono::system_clock::now();
            cond_.wait_until(guard, now + std::chrono::milliseconds(10));
            guard.unlock();
        }
        return batch;
    }
    inline ImgPacket* Get() {
        ImgPacket* packet = nullptr;
        std::unique_lock<std::mutex> guard(mutex_);
        if(!streams_.empty()){
            auto it = last_iter_;
            if (it == streams_.end()) it = streams_.begin();
            size_t count = 0;

            while (count < streams_.size()) {
                if (it == streams_.end()) it = streams_.begin();
                auto& entry = it->second;
                if (!entry.queue.empty() && entry.state == StreamState::ACTIVE) {
                    packet = entry.queue.front();
                    entry.queue.pop_front();
                    last_iter_ = std::next(it);
                    if (last_iter_ == streams_.end()){
                        last_iter_ = streams_.begin();
                    }
                    break;
                }
                ++it;
                ++count;
            }
        } else {
            auto now = std::chrono::system_clock::now();
            cond_.wait_until(guard, now + std::chrono::milliseconds(10));
            guard.unlock();
        }

        return packet;
    }

    inline std::vector<ImgPacket*> GetBatch(size_t batch_size) {
        std::unique_lock<std::mutex> guard(mutex_);
        std::vector<ImgPacket*> batch;
        if(!streams_.empty()){
            auto it = last_iter_;
            if (it == streams_.end()) it = streams_.begin();

            size_t count = 0;
            size_t visited = 0;

            while (count < batch_size && visited < streams_.size()) {
                if (it == streams_.end()){
                    it = streams_.begin();
                }
                auto& entry = it->second;

                while (!entry.queue.empty() && entry.state == StreamState::ACTIVE && count < batch_size) {
                    batch.push_back(entry.queue.front());
                    entry.queue.pop_front();
                    ++count;
                }

                ++it;
                ++visited;
            }

            last_iter_ = it;
        } else {
            auto now = std::chrono::system_clock::now();
            cond_.wait_until(guard, now + std::chrono::milliseconds(10));
            guard.unlock();
        }
        return batch;
    }
    inline void MarkStreamEOS(int64_t stream_id) {
        std::unique_lock<std::mutex> guard(mutex_);
        auto it = streams_.find(stream_id);
        if (it == streams_.end()) {
            return;
        }
        it->second.state = StreamState::EOS;
        cond_.notify_all();
    }
    inline bool IsStreamAlive(int64_t stream_id) {
        std::unique_lock<std::mutex> guard(mutex_);
        auto it = streams_.find(stream_id);
        return it != streams_.end() && it->second.state == StreamState::ACTIVE;
    }
    inline void CheckThreadLoop() {
        while (!abort_) {
            std::vector<int64_t> to_remove;
            std::unique_lock<std::mutex> guard(mutex_);
            for (auto& kv : streams_) {
                auto& entry = kv.second;
                if (entry.state != StreamState::ACTIVE && entry.queue.empty()) {
                    entry.state = StreamState::REMOVED;
                    to_remove.push_back(kv.first);
                }
            }

            for (auto sid : to_remove) {
                auto& q = streams_[sid].queue;
                while (!q.empty()) {
                    ImgPacket *packet = q.front();
                    q.pop_front();
                    if(packet){
                        delete packet;
                        packet = nullptr;
                    }
                }
                streams_.erase(sid);
            }
            guard.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
private:
    std::unordered_map<int64_t, StreamEntry> streams_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread worker_;
    bool abort_ = false;
    std::unordered_map<int64_t, StreamEntry>::iterator last_iter_ = streams_.end();
};


/// =======================
/// Distributor Node
/// =======================
class DistributorNode {
public:
    DistributorNode() {
        worker_ = std::thread(&DistributorNode::ThreadLoop, this);
    }

    ~DistributorNode() {
        abort_ = true;
        worker_.join();
        std::unique_lock<std::mutex> guard(mutex_);
        while (!img_list_.empty()) {
            ImgPacket *packet = img_list_.front();
            img_list_.pop_front();
            if(packet){
                delete packet;
                packet = nullptr;
            }
        }
        log_debug("~DistributorNode");
    }

    inline void Push(ImgPacket* packet) {
        std::unique_lock<std::mutex> guard(mutex_);
        img_list_.push_back(packet);
        cond_.notify_one();
    }

private:
    inline void ThreadLoop() {
        while (!abort_) {
            ImgPacket* packet = nullptr;
            std::unique_lock<std::mutex> guard(mutex_);
            if (!img_list_.empty()) {
                packet = img_list_.front();
                img_list_.pop_front();
                if (packet) {
                    QueueContext* ctx = packet->context;
                    if(ctx->time_for_log.stopTimer() >= 1000) {
                        ctx->time_for_log.startTimer();
                        int flow_time = packet->timer.stopTimer();
                        log_debug("stream id:{} flow time:{}", ctx->stream_id, flow_time);
                    }
                    if (ctx && ctx->listener) {
                        if(packet->use_ptr && packet->device_image_ptr != nullptr && ctx->mempool != nullptr){
                            ctx->mempool->putAddr(packet->device_image_ptr);
                        }
                        ctx->listener->OnInferData(packet->img, packet->info, packet->timestamp);
                    }
                    delete packet;
                }
            } else {
                auto now = std::chrono::system_clock::now();
                cond_.wait_until(guard, now + std::chrono::milliseconds(10));
                guard.unlock();
            }
            
        }
    }

private:
    std::list<ImgPacket*> img_list_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread worker_;
    bool abort_ = false;
};

#endif // DATA_CHANNEL_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND DETECTION_HYGON