/*
 * thread_pool_wasm.h -- Sequential replacement for xLearn's ThreadPool
 *
 * For WASM builds, we replace the multi-threaded ThreadPool with a
 * synchronous version that executes tasks immediately in enqueue().
 * This avoids std::thread/mutex/condition_variable which require
 * pthreads support in Emscripten.
 */

#ifndef XLEARN_BASE_THREAD_POOL_H_
#define XLEARN_BASE_THREAD_POOL_H_

#include <functional>
#include <future>
#include <stdexcept>

#include "src/base/common.h"

class ThreadPool {
 public:
  ThreadPool(size_t threads) : thread_count_(threads > 0 ? threads : 1) {}
  ~ThreadPool() {}

  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    std::packaged_task<return_type()> task(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task.get_future();
    task();  // Execute immediately (synchronous)
    return res;
  }

  void Sync(int wait_count) {
    // No-op: all tasks already completed synchronously
    (void)wait_count;
  }

  size_t ThreadNumber() { return thread_count_; }

 private:
  size_t thread_count_;
};

inline size_t getStart(size_t count, size_t total, size_t id) {
  size_t gap = count / total;
  return id * gap;
}

inline size_t getEnd(size_t count, size_t total, size_t id) {
  size_t gap = count / total;
  size_t remain = count % total;
  size_t end_index = (id + 1) * gap;
  if (id == total - 1) {
    end_index += remain;
  }
  return end_index;
}

#endif  // XLEARN_BASE_THREAD_POOL_H_
