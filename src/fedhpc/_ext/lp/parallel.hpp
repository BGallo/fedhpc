// fedlp: a small persistent thread pool for the simplex's data-parallel
// kernels (the column-wise pivot-row pass over large working sets).
//
// run(n, f) calls f(k) for k in [0, n) on the workers and the caller and
// returns when all are done. Between calls workers spin briefly (a simplex
// issues one call per pivot, microseconds apart), then sleep on a condition
// variable. One process-wide pool, created on first use; calls from several
// solvers are serialised by a mutex.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fedlp {

class ThreadPool {
public:
    explicit ThreadPool(int workers) {
        for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    int size() const { return (int)threads_.size() + 1; }   // workers + caller

    void run(int n, const std::function<void(int)>& f) {
        std::lock_guard<std::mutex> serial(run_m_);
        {
            std::lock_guard<std::mutex> g(m_);
            task_ = &f;
            n_ = n;
            next_.store(0);
            done_.store(0);
            ++gen_;
        }
        cv_.notify_all();
        work();
        while (done_.load(std::memory_order_acquire) < n) std::this_thread::yield();
        {
            std::lock_guard<std::mutex> g(m_);
            task_ = nullptr;   // a worker waking late finds nothing to do
        }
        // f must outlive every worker still inside work()
        while (active_.load(std::memory_order_acquire) > 0) std::this_thread::yield();
    }

    static ThreadPool& global(int workers) {
        static ThreadPool pool(workers);
        return pool;
    }

private:
    std::vector<std::thread> threads_;
    std::mutex m_, run_m_;
    std::condition_variable cv_;
    const std::function<void(int)>* task_ = nullptr;
    int n_ = 0;
    unsigned long long gen_ = 0;
    bool stop_ = false;
    std::atomic<int> next_{0}, done_{0}, active_{0};

    void work() {
        const std::function<void(int)>* f;
        int n;
        {
            std::lock_guard<std::mutex> g(m_);
            f = task_, n = n_;
            if (!f) return;
            active_.fetch_add(1);   // registered under the lock that reads task_
        }
        for (int k; (k = next_.fetch_add(1)) < n;) {
            (*f)(k);
            done_.fetch_add(1, std::memory_order_release);
        }
        active_.fetch_sub(1, std::memory_order_release);
    }

    void loop() {
        unsigned long long seen = 0;
        while (true) {
            // spin a little: the next call usually comes microseconds later
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            bool ready = false;
            while (std::chrono::steady_clock::now() < until) {
                {
                    std::lock_guard<std::mutex> g(m_);
                    if (gen_ != seen) { ready = true; break; }
                }
                std::this_thread::yield();
            }
            if (!ready) {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
            }
            {
                std::lock_guard<std::mutex> g(m_);
                seen = gen_;
                if (stop_) return;
            }
            work();
        }
    }
};

}  // namespace fedlp
