#if defined(DETECTION_NVIDIA) || defined(DETECTION_ASCEND)
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
#include "DetectionInfo.h"
#include "log_helpers.h"

/// =======================
/// Infer Data Listener
/// =======================
class InferDataListner {
public:
    virtual void OnInferData(cv::Mat& img, DetectionInfo& info) = 0;
};

/// =======================
/// Queue Context
/// =======================
struct QueueContext {
    int64_t stream_id{0};
    InferDataListner* listener{nullptr};
};

inline QueueContext* CreateContext(InferDataListner* listener) {
    static std::atomic<int64_t> stream_id_init{-1};
    auto* ctx = new QueueContext;
    ctx->stream_id = ++stream_id_init;
    ctx->listener = listener;
    return ctx;
}

inline void DestroyContext(QueueContext* ctx) {
    if(ctx)
        delete ctx;
}

/// =======================
/// Image Packet
/// =======================
class ImgPacket {
public:
    /// rvalue cv::Mat (zero-copy)
    ImgPacket(cv::Mat&& img, QueueContext* context)
        : img_(std::move(img)), context_(context) {}

    /// lvalue cv::Mat
    ImgPacket(const cv::Mat& img, QueueContext* context)
        : img_(img), context_(context) {}

    ~ImgPacket() = default;

    void SetDetectionInfo(DetectionInfo& info) {
        info_ = info;
    }

    cv::Mat& GetImg() { return img_; }
    DetectionInfo& GetDetectionInfo() { return info_; }
    QueueContext* GetContext() { return context_; }

private:
    cv::Mat img_;
    DetectionInfo info_;
    QueueContext* context_{nullptr};
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
            delete img_list_.front();
            img_list_.pop_front();
        }
        log_debug("~CollectorNode");
    }

    /// Prefer this: move cv::Mat
    inline void Push(cv::Mat&& img, QueueContext* context) {
        std::unique_lock<std::mutex> guard(mutex_);
        img_list_.push_back(new ImgPacket(std::move(img), context));
        cond_.notify_one();
    }

    /// Fallback: copy cv::Mat
    inline void Push(const cv::Mat& img, QueueContext* context) {
        std::unique_lock<std::mutex> guard(mutex_);
        img_list_.push_back(new ImgPacket(img, context));
        cond_.notify_one();
    }
    inline std::vector<ImgPacket*> GetBatch(size_t batch_size) {
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
                delete q.front();
                q.pop_front();
            }
        }
        streams_.clear();
        log_debug("~RelayNode");
    }


    inline void Push(ImgPacket* packet) {
        if (!packet || !packet->GetContext()) 
            return;

        int64_t sid = packet->GetContext()->stream_id;
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
            delete img_list_.front();
            img_list_.pop_front();
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
                    QueueContext* ctx = packet->GetContext();
                    if (ctx && ctx->listener) {
                        ctx->listener->OnInferData(
                            packet->GetImg(),
                            packet->GetDetectionInfo());
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
#endif // DETECTION_NVIDIA DETECTION_ASCEND