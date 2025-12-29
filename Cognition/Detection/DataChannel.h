#define DETECTION_NVIDIA
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

/// =======================
/// Infer Data Listener
/// =======================
class InferDataListner {
public:
    virtual ~InferDataListner() = default;

    virtual void OnInferData(const cv::Mat& img,
                             const std::vector<DetectionInfo>& info) = 0;
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

    void SetDetectionInfo(const std::vector<DetectionInfo>& info) {
        info_ = info;
    }

    cv::Mat& GetImg() const { return img_; }
    std::vector<DetectionInfo>& GetDetectionInfo() const { return info_; }
    QueueContext* GetContext() const { return context_; }

private:
    cv::Mat img_;
    std::vector<DetectionInfo> info_;
    QueueContext* context_{nullptr};
};

/// =======================
/// Collector Node
/// =======================
class CollectorNode {
public:
    CollectorNode() = default;

    ~CollectorNode() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!img_list_.empty()) {
            delete img_list_.front();
            img_list_.pop_front();
        }
    }

    /// Prefer this: move cv::Mat
    inline void Push(cv::Mat&& img, QueueContext* context) {
        std::lock_guard<std::mutex> lock(mutex_);
        img_list_.push_back(new ImgPacket(std::move(img), context));
        cond_.notify_one();
    }

    /// Fallback: copy cv::Mat
    inline void Push(const cv::Mat& img, QueueContext* context) {
        std::lock_guard<std::mutex> lock(mutex_);
        img_list_.push_back(new ImgPacket(img, context));
        cond_.notify_one();
    }

    inline std::vector<ImgPacket*> GetBatch(size_t batch_size) {
        std::vector<ImgPacket*> batch;
        std::unique_lock<std::mutex> lock(mutex_);

        cond_.wait_for(lock, std::chrono::milliseconds(100),
                       [&] { return !img_list_.empty(); });

        size_t real_size = std::min(batch_size, img_list_.size());
        for (size_t i = 0; i < real_size; ++i) {
            batch.push_back(img_list_.front());
            img_list_.pop_front();
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
    RelayNode() = default;
    ~RelayNode() { Clear(); }


    inline void Push(ImgPacket* packet) {
        if (!packet || !packet->GetContext()) return;

        int64_t sid = packet->GetContext()->stream_id;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& entry = streams_[sid];
            if (entry.state != StreamState::ACTIVE) {
                delete packet;
                return;
            }
            entry.queue.push_back(packet);
        }
        cond_.notify_all();
    }

    inline ImgPacket* Get(int64_t stream_id) {
        std::unique_lock<std::mutex> lock(mutex_);

        cond_.wait(lock, [&] {
            auto it = streams_.find(stream_id);
            if (it == streams_.end()) return true;
            return !it->second.queue.empty() ||
                   it->second.state != StreamState::ACTIVE;
        });

        auto it = streams_.find(stream_id);
        if (it == streams_.end()) return nullptr;

        auto& entry = it->second;

        if (!entry.queue.empty()) {
            ImgPacket* pkt = entry.queue.front();
            entry.queue.pop_front();
            return pkt;
        }

        return nullptr;
    }


    inline std::vector<ImgPacket*> GetBatch(int64_t stream_id, size_t batch_size) {
        std::vector<ImgPacket*> batch;
        std::unique_lock<std::mutex> lock(mutex_);

        cond_.wait(lock, [&] {
            auto it = streams_.find(stream_id);
            if (it == streams_.end()) return true;
            return !it->second.queue.empty() ||
                   it->second.state != StreamState::ACTIVE;
        });

        auto it = streams_.find(stream_id);
        if (it == streams_.end()) return batch;

        auto& entry = it->second;
        size_t real = std::min(batch_size, entry.queue.size());
        for (size_t i = 0; i < real; ++i) {
            batch.push_back(entry.queue.front());
            entry.queue.pop_front();
        }
        return batch;
    }

    inline void MarkStreamEOS(int64_t stream_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& entry = streams_[stream_id];
        entry.state = StreamState::EOS;
        cond_.notify_all();
    }

    inline void RemoveStream(int64_t stream_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = streams_.find(stream_id);
        if (it == streams_.end()) return;

        auto& entry = it->second;
        while (!entry.queue.empty()) {
            delete entry.queue.front();
            entry.queue.pop_front();
        }
        entry.state = StreamState::REMOVED;
        streams_.erase(it);
        cond_.notify_all();
    }

    inline bool IsStreamAlive(int64_t stream_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = streams_.find(stream_id);
        return it != streams_.end() &&
               it->second.state == StreamState::ACTIVE;
    }

    inline void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& kv : streams_) {
            auto& q = kv.second.queue;
            while (!q.empty()) {
                delete q.front();
                q.pop_front();
            }
        }
        streams_.clear();
    }

private:
    std::unordered_map<int64_t, StreamEntry> streams_;
    std::mutex mutex_;
    std::condition_variable cond_;
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
        abort_.store(true);
        cond_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        while (!img_list_.empty()) {
            delete img_list_.front();
            img_list_.pop_front();
        }
    }

    inline void Push(ImgPacket* packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        img_list_.push_back(packet);
        cond_.notify_one();
    }

private:
    inline void ThreadLoop() {
        while (!abort_.load()) {
            ImgPacket* packet = nullptr;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_.wait_for(lock, std::chrono::milliseconds(100),
                               [&] { return abort_.load() || !img_list_.empty(); });

                if (!img_list_.empty()) {
                    packet = img_list_.front();
                    img_list_.pop_front();
                }
            }

            if (packet) {
                QueueContext* ctx = packet->GetContext();
                if (ctx && ctx->listener) {
                    ctx->listener->OnInferData(
                        packet->GetImg(),
                        packet->GetDetectionInfo());
                }
                delete packet;
            }
        }
    }

private:
    std::list<ImgPacket*> img_list_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::thread worker_;
    std::atomic<bool> abort_{false};
};

#endif // DATA_CHANNEL_H
#endif // DETECTION_NVIDIA DETECTION_ASCEND