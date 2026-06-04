#ifndef TIMEMETRICS_H
#define TIMEMETRICS_H
#include <chrono>

class TimeMetrics {
public:
    TimeMetrics() {
        startTimer();
    }

    ~TimeMetrics() = default;

    void startTimer() {
        start_time_ = std::chrono::steady_clock::now();
    }

    int stopTimer() {
        end_time_ = std::chrono::steady_clock::now();
        return calculateElapsedTime();
    }

private:
    int calculateElapsedTime() const {
        return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_).count());
    }

    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
};
#endif