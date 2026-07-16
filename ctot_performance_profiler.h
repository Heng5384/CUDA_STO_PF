#ifndef CTOT_PERFORMANCE_PROFILER_H
#define CTOT_PERFORMANCE_PROFILER_H

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

// Profiling is runtime opt-in. When disabled, scopes perform one branch and
// create no CUDA events, NVTX ranges, files, or device synchronization.
class CtotPerformanceProfiler {
  public:
    CtotPerformanceProfiler() = default;
    ~CtotPerformanceProfiler() { close(); }

    bool open(bool enabled, const char *path) {
        enabled_ = enabled;
        if (!enabled_) return true;
        fp_ = std::fopen(path, "w");
        if (!fp_) {
            enabled_ = false;
            return false;
        }
        std::fprintf(fp_,
                     "step,attempt,outer,stage,gpu_ms,host_ms\n");
        return true;
    }

    bool enabled() const { return enabled_; }

    int begin(const char *stage, int step, int attempt, int outer) {
        if (!enabled_) return -1;
        Record record;
        record.stage = stage;
        record.step = step;
        record.attempt = attempt;
        record.outer = outer;
        record.host_start = Clock::now();
        if (cudaEventCreate(&record.start) != cudaSuccess ||
            cudaEventCreate(&record.stop) != cudaSuccess ||
            cudaEventRecord(record.start, 0) != cudaSuccess) {
            if (record.start) cudaEventDestroy(record.start);
            if (record.stop) cudaEventDestroy(record.stop);
            return -1;
        }
        records_.push_back(record);
        nvtxRangePushA(stage);
        return static_cast<int>(records_.size() - 1);
    }

    void end(int token) {
        if (!enabled_ || token < 0 ||
            token >= static_cast<int>(records_.size())) return;
        Record &record = records_[static_cast<size_t>(token)];
        if (record.ended) return;
        cudaEventRecord(record.stop, 0);
        record.host_stop = Clock::now();
        record.ended = true;
        nvtxRangePop();
    }

    // Flush only at an attempt boundary. One synchronization on the last
    // recorded event makes all earlier same-stream events queryable.
    bool flush() {
        if (!enabled_ || records_.empty()) return true;
        for (Record &record : records_) {
            if (!record.ended) {
                cudaEventRecord(record.stop, 0);
                record.host_stop = Clock::now();
                record.ended = true;
            }
        }
        if (cudaEventSynchronize(records_.back().stop) != cudaSuccess)
            return false;
        for (Record &record : records_) {
            float gpu_ms = 0.0f;
            if (cudaEventElapsedTime(&gpu_ms, record.start, record.stop) !=
                cudaSuccess) return false;
            const double host_ms = std::chrono::duration<double, std::milli>(
                record.host_stop - record.host_start).count();
            std::fprintf(fp_, "%d,%d,%d,%s,%.9g,%.9g\n",
                         record.step, record.attempt, record.outer,
                         record.stage.c_str(), static_cast<double>(gpu_ms),
                         host_ms);
            cudaEventDestroy(record.start);
            cudaEventDestroy(record.stop);
        }
        records_.clear();
        std::fflush(fp_);
        return true;
    }

    void close() {
        if (!enabled_) return;
        flush();
        if (fp_) std::fclose(fp_);
        fp_ = nullptr;
        enabled_ = false;
    }

  private:
    using Clock = std::chrono::steady_clock;
    struct Record {
        std::string stage;
        int step = 0;
        int attempt = 0;
        int outer = -1;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        Clock::time_point host_start;
        Clock::time_point host_stop;
        bool ended = false;
    };

    bool enabled_ = false;
    FILE *fp_ = nullptr;
    std::vector<Record> records_;
};

class CtotPerformanceScope {
  public:
    CtotPerformanceScope(CtotPerformanceProfiler *profiler,
                         const char *stage, int step, int attempt, int outer)
        : profiler_(profiler), token_(profiler ? profiler->begin(
              stage, step, attempt, outer) : -1),
          flush_on_exit_(stage && std::strcmp(stage, "attempt.total") == 0) {}
    ~CtotPerformanceScope() {
        if (profiler_) profiler_->end(token_);
        if (profiler_ && flush_on_exit_) profiler_->flush();
    }

  private:
    CtotPerformanceProfiler *profiler_;
    int token_;
    bool flush_on_exit_;
};

#define CTOT_PERF_JOIN_IMPL(a, b) a##b
#define CTOT_PERF_JOIN(a, b) CTOT_PERF_JOIN_IMPL(a, b)
#define CTOT_PERF_SCOPE(profiler, stage, step, attempt, outer) \
    CtotPerformanceScope CTOT_PERF_JOIN(ctot_perf_scope_, __COUNTER__)( \
        (profiler), (stage), (step), (attempt), (outer))

#endif
